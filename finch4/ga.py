"""THE universal genetic algorithm — Daniel's specification, 2026-07-21.

One solve function and no others: solve(fitness_fns, output_shape, epochs).

The two ideas that make this GA unusual are the only two it keeps. First,
no individual ever IS a solution: an individual is genes (the input the
shared decoder network reads) plus latents (a vector that bends the shared
network's behavior for that individual alone), and solutions are always
computed by decoding. Second, there is exactly ONE decoder network for the
whole run, and on multi-function runs it LEARNS: the base is periodically
distilled — gradient-trained toward each function's best-ever phenotype
from its genes, with every per-individual modifier decaying afterward — so
the environment itself absorbs discoveries over time. Evolution vets;
gradients consolidate. (The original arithmetic fold — apply a bending
directly, no training — was searched for at short and long budgets, both
substrates, alone and alongside distillation, and no configuration was
found where it helps; removed 2026-07-30 at Daniel's direction.)

Genes and latents are different concepts and are never conflated: they are
stored separately, crossed by different functions, mutated by different
functions, and no operator treats the pair as one string of numbers.

Fitness is organized as SHARES (Daniel's environment rule): the combined
fitness mass of the whole environment is always 1, every fitness function's
population collectively owns an equal slice of it, and individuals split
their function's slice by within-function rank. Selection and survival both
run on shares, so a function whose population swells dilutes its members
and self-corrects, and a function down to one struggling member concentrates
its whole slice there. Overtaking is impossible by construction. Extinction
is still allowed (an empty function owns nothing until speciation
re-seeds it).

Every operator is a replaceable function:

  selection          — which two parents breed each child
  gene_crossover     — how two parents' genes combine
  latent_inheritance — which parent's latents a child receives (whole)
  gene_mutation / latent_mutation — how each space is perturbed
  speciation         — how individuals move onto other fitness functions
  consolidation      — how the shared decoder absorbs discoveries
                       (default: the Distillation operator)
  directions         — the SUBSTRATE: how latents modify the decoder;
                       a registered choice (register_substrate), like
                       architecture (register_architecture)

Population starts from `founders` random individuals per fitness function
(default 16 — measured 2026-07-27: every individual descends from the
founding set, so founding count IS the run's coverage of the space; two
founders left plateau problems unsolvable that sixteen solve 10/10, at a
0.6% budget cost, and images are neutral-to-better). There are no
champions, no reserved slots per problem, and no fixed generation size. A
best-ever record per function is kept as pure bookkeeping (never bred
from) so the solver returns an answer for every function even after
extinctions.

Since 2026-07-31 the epoch is not a monolithic loop body: TensorGA holds
the state and the laws, and each stage of the epoch is one engine method,
run in the canonical order TensorGA.STAGES. solve() drives that order
directly; the layer preset (finch4.layers.tensor_environment) drives the
SAME methods through the Finch layer grammar — one source of truth, so the
two paths are bit-identical by construction, not by testing alone. The
laws (fitness shares, culling, the best-ever archive, the one-decoder
invariant) are engine-owned; layers decide WHEN a stage runs, never HOW.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .conditional import build_conditional_decoder


# ------------------------------------------------------- decoder substrates
#
# The substrate — HOW an individual's latents modify the one shared decoder
# — is a registered choice, exactly like `architecture` (Daniel, 2026-07-30:
# "ensure that decoder choice ... modular like Finch, then we can get to new
# types of decoders"). A builder returns (decoder, capabilities); the
# decoder needs decode(genes, latents) [+ decode_seeded/absorb_seeded when
# seeded], get_params/set_params, and — to support consolidation —
# training_logits(genes) plus optionally sync_base(). Capabilities:
#   seeded        each individual carries an integer basis seed
#   shared_sites  all individuals share one coordinate system (lets
#                 consolidation run on a seeded substrate)

_SUBSTRATES: dict = {}


def register_substrate(name, builder):
    """Make `name` usable as the `directions` argument of solve()."""
    _SUBSTRATES[name] = builder


def _conditional_substrate(architecture, genes, output_shape, latents, device):
    return (build_conditional_decoder(architecture, genes, output_shape,
                                      latents, device),
            {"seeded": False, "shared_sites": False})


def _individual_substrate(architecture, genes, output_shape, latents, device):
    from .conditional import attach_seeded_directions
    decoder, _ = _conditional_substrate(architecture, genes, output_shape,
                                        latents, device)
    attach_seeded_directions(decoder)
    return decoder, {"seeded": True, "shared_sites": False}


def _sparse_substrate(shared):
    def build(architecture, genes, output_shape, latents, device):
        from .sparse import build_sparse_decoder
        return (build_sparse_decoder(architecture, genes, output_shape,
                                     latents, device),
                {"seeded": True, "shared_sites": shared})
    return build


register_substrate("frozen", _conditional_substrate)
register_substrate("evolve", _conditional_substrate)
register_substrate("individual", _individual_substrate)
register_substrate("sparse", _sparse_substrate(False))
register_substrate("sparse-shared", _sparse_substrate(True))


class Distillation:
    """The consolidation operator (replaceable via `consolidation=`).
    Evolution vets; this trains the base toward what the vetted champions
    achieved, then the loop decays every modifier (the discovery lives in
    the base now; decay 1.0 measured +49% worse — double counting). Tuned
    10/10 at t=+13.2: every=64, decay=0.7."""

    def __init__(self, every=64, steps=40, decay=0.7, lr=1e-3,
                 buffer_cap=256):
        self.every = int(every)
        self.steps = int(steps)
        self.decay = float(decay)
        self.lr = float(lr)
        self.buffer_cap = int(buffer_cap)
        self.replay_z: list = []
        self.replay_p: list = []
        self._opt = None

    def due(self, epoch):
        return (epoch + 1) % self.every == 0

    def run(self, decoder, best_genes, best_pheno):
        for genes_f, pheno_f in zip(best_genes, best_pheno):
            if pheno_f is not None:
                self.replay_z.append(genes_f.copy())
                self.replay_p.append(pheno_f.reshape(-1).copy())
        del self.replay_z[:-self.buffer_cap]
        del self.replay_p[:-self.buffer_cap]
        if not self.replay_z:
            return
        if self._opt is None:
            trainable = [q for name, q in decoder.net.named_parameters()
                         if "down" not in name and "up" not in name]
            self._opt = torch.optim.Adam(trainable, lr=self.lr)
        Z = torch.as_tensor(np.stack(self.replay_z), device=decoder.device)
        P = torch.as_tensor(np.stack(self.replay_p), device=decoder.device)
        for _ in range(self.steps):
            idx = torch.randint(0, len(Z), (min(64, len(Z)),),
                                device=decoder.device)
            self._opt.zero_grad()
            out = torch.sigmoid(
                decoder.training_logits(Z[idx])).reshape(len(idx), -1)
            ((out - P[idx]) ** 2).mean().backward()
            self._opt.step()
        if hasattr(decoder, "sync_base"):
            decoder.sync_base()


# ---------------------------------------------------------------- results

@dataclass
class ProblemResult:
    best_phenotype: np.ndarray | None   # None if the function was never tried
    best_fitness: float
    initial_fitness: float
    evaluations: int


@dataclass
class GAResult:
    problems: list[ProblemResult]
    evaluations: int
    epochs: int
    history: list[dict] = field(repr=False, default_factory=list)
    decoder: np.ndarray | None = field(repr=False, default=None)

    @property
    def best_phenotype(self) -> np.ndarray:
        if len(self.problems) != 1:
            raise ValueError("best_phenotype is ambiguous for "
                             f"{len(self.problems)} problems; use .problems")
        return self.problems[0].best_phenotype

    @property
    def best_fitness(self) -> float:
        if len(self.problems) != 1:
            raise ValueError("best_fitness is ambiguous for "
                             f"{len(self.problems)} problems; use .problems")
        return self.problems[0].best_fitness


# --------------------------------------------------------------- shares

def fitness_shares(scores: np.ndarray, fn_idx: np.ndarray) -> np.ndarray:
    """Each living function owns 1/(functions alive) of the total fitness
    mass; its members split that slice by within-function rank (linear
    ranking on raw scores — raw comparison is legal inside one function).
    Returns one weight per individual; the weights sum to 1."""
    weights = np.zeros(len(scores))
    alive = np.unique(fn_idx)
    slice_mass = 1.0 / len(alive)
    for f in alive:
        members = np.flatnonzero(fn_idx == f)
        order = np.argsort(np.argsort(-scores[members]))   # 0 = best
        rank_weight = (len(members) - order).astype(np.float64)
        weights[members] = slice_mass * rank_weight / rank_weight.sum()
    return weights


# ------------------------------------------------------- default operators

def make_species_selection(outcross_rate=0.05):
    """The default: parent one is drawn proportionally to fitness share;
    parent two comes from the SAME function (species breed within
    themselves), also share-proportional, except a rare outcross draws it
    from the whole population instead. Rare cross-species mixing is the
    same law the earlier campaign measured for crossover generally: the
    partner must be compatible most of the time, genuinely different only
    rarely. Fully-mixed pairing was measured to floor both mutation dials
    (cross-species chimeras almost never beat their parents) and to double
    the evaluation bill (every mixed child is scored twice)."""
    def select(weights, fn_idx, rng, n_pairs):
        pop = len(weights)
        p = weights / weights.sum()
        a = rng.choice(pop, size=n_pairs, p=p)
        b = np.empty(n_pairs, dtype=np.int64)
        outcross = rng.random(n_pairs) < outcross_rate
        for i in range(n_pairs):
            kin = np.flatnonzero(fn_idx == fn_idx[a[i]])
            pool = np.arange(pop) if (outcross[i] or len(kin) < 2) else kin
            if len(pool) == 1:
                b[i] = pool[0]
                continue
            pw = weights[pool] / weights[pool].sum()
            b[i] = rng.choice(pool, p=pw)
            while b[i] == a[i] and len(pool) > 1:
                b[i] = rng.choice(pool, p=pw)
        return a, b
    return select


def share_selection(weights, fn_idx, rng, n_pairs):
    """Fully-mixed share-proportional pairing — kept as a research arm;
    see make_species_selection for why it is not the default."""
    pop = len(weights)
    p = weights / weights.sum()
    a = rng.choice(pop, size=n_pairs, p=p)
    b = rng.choice(pop, size=n_pairs, p=p)
    if pop > 1:
        clash = a == b
        while clash.any():
            b[clash] = rng.choice(pop, size=int(clash.sum()), p=p)
            clash = a == b
    return a, b


def uniform_selection(weights, fn_idx, rng, n_pairs):
    """No selection pressure at reproduction — kept as the control arm."""
    pop = len(weights)
    a = rng.integers(0, pop, n_pairs)
    b = rng.integers(0, pop, n_pairs)
    if pop > 1:
        clash = a == b
        while clash.any():
            b[clash] = rng.integers(0, pop, int(clash.sum()))
            clash = a == b
    return a, b


def one_point_gene_crossover(genes_a, genes_b, rng):
    """One cut per child across the gene vector only."""
    n, dim = genes_a.shape
    cuts = rng.integers(1, dim, n)
    take = np.arange(dim)[None, :] >= cuts[:, None]
    return np.where(take, genes_b, genes_a).astype(np.float32)


def coin_flip_latent_inheritance(latents_a, latents_b, rng):
    """Each child receives one parent's latents whole — latents are never
    spliced (Daniel's rule: a latent vector is a coherent bending of the
    shared decoder and half of one is not half as useful)."""
    pick_b = rng.random(len(latents_a)) < 0.5
    return np.where(pick_b[:, None], latents_b, latents_a).astype(np.float32)


def make_gaussian_mutation(rate=0.1, sigma=0.12):
    """Masked gaussian noise scaled by a self-tuning step dial. The dial is
    owned by the engine (one per space, updated by success rate); the
    operator just applies it."""
    def mutate(values, rng, dial):
        mask = rng.random(values.shape) < rate
        noise = rng.normal(0.0, sigma * dial, values.shape)
        return (values + mask * noise).astype(np.float32)
    return mutate


def make_random_speciation(rate=0.02):
    """Each living individual is re-assigned to a uniformly random function
    with probability `rate` per epoch. Re-assignment triggers an honest
    re-scoring on the new function."""
    def speciate(fn_idx, n_functions, epoch, rng):
        moves = rng.random(len(fn_idx)) < rate
        new = fn_idx.copy()
        new[moves] = rng.integers(0, n_functions, int(moves.sum()))
        return new
    return speciate


# ------------------------------------------------------------ the wave

@dataclass
class Wave:
    """One epoch's children in flight, with the bookkeeping later stages
    need. This object IS the contract between the breeding stages — it
    travels explicitly (engine.wave, mirrored to env.state["wave"] by the
    layer preset) instead of as loose dict keys, so a custom layer can
    inspect or amend a wave without guessing at hidden state.

    a/b are parent indices into the population arrays as they stood at
    breeding time; gene_rows/latent_rows say which children mutated which
    space (each child mutates exactly one, so each dial's feedback stays
    uncontaminated); pre_genes/pre_latents are the pre-mutation values the
    mutation-memory accumulator differences against; fa/fb/mixed/fn track
    function adoption; score/success feed the dials."""
    a: np.ndarray
    b: np.ndarray
    genes: np.ndarray
    latents: np.ndarray
    basis: np.ndarray
    fa: np.ndarray | None = None
    fb: np.ndarray | None = None
    mixed: np.ndarray | None = None
    fn: np.ndarray | None = None
    score: np.ndarray | None = None
    success: np.ndarray | None = None
    gene_rows: np.ndarray | None = None
    latent_rows: np.ndarray | None = None
    pre_genes: np.ndarray | None = None
    pre_latents: np.ndarray | None = None


# ------------------------------------------------------------- the engine

class TensorGA:
    """The tensor engine: population state plus the laws, with each stage
    of the epoch as one method. STAGES is the canonical order — solve()
    runs it directly via step(); the layer preset runs the same methods
    one Layer per stage. Reordering or replacing stages is a NEW mechanism
    under the ledger rule: it runs fine, and it inherits no evidence.

    The laws live here and only here: fitness shares (fitness_shares),
    culling by shares with extinction allowed (integrate), the best-ever
    archive as pure bookkeeping (_score), and the one-decoder invariant
    (self.decoder is built once and never per-individual). Stage methods
    decide nothing about those; they only sequence them.

    Parameters are solve()'s — see solve.__doc__ for the full story;
    construction founds and scores the initial population."""

    STAGES = ("immigrate", "breed", "mutate", "score_and_adopt",
              "tune_dials", "integrate", "speciate", "consolidate",
              "evolve_directions", "record")

    def __init__(self, fitness_fns, output_shape, epochs=1_000,
                 architecture="auto", genes=64, latents=None, children=16,
                 population_cap=32, device="auto", selection=None,
                 gene_crossover=one_point_gene_crossover,
                 latent_inheritance=coin_flip_latent_inheritance,
                 gene_mutation=None, latent_mutation=None,
                 speciation=None,
                 directions="sparse-shared", direction_every=16,
                 direction_sigma=0.1, fresh_basis_rate=0.1,
                 win_target=0.2, dial_step=1.15,
                 mutation_memory="off", memory_drift=0.5,
                 distill="auto", distill_every=64,
                 distill_steps=40, distill_decay=0.7, distill_lr=1e-3,
                 consolidation=None,
                 founding="per_function", founders=16,
                 immigrants="off", immigrant_patience=32,
                 progress=None, progress_every=None,
                 init_decoder=None, seed=None):
        fns = [fitness_fns] if callable(fitness_fns) else list(fitness_fns)
        if not fns:
            raise ValueError("at least one fitness function is required")
        self.fns = fns
        self.n_fns = len(fns)
        if latents is None:
            latents = 2048 if directions in ("sparse", "sparse-shared") else 64
        self.genes_dim = int(genes)
        self.latents_dim = int(latents)
        self.output_shape = tuple(int(s) for s in output_shape)
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(int(self.rng.integers(0, 2 ** 31)))
        if directions not in _SUBSTRATES:
            raise ValueError(f"unknown substrate {directions!r}; "
                             f"registered: {sorted(_SUBSTRATES)}")
        self.directions = directions
        self.decoder, caps = _SUBSTRATES[directions](
            architecture, genes, self.output_shape, latents, device)
        if init_decoder is not None:
            # Warm start: the shared decoder begins as a previous run's, so a
            # new problem inherits the family's structure instead of starting
            # from a random prior. `GAResult.decoder` is the vector to pass in.
            self.decoder.set_params(init_decoder)

        self.selection = selection or make_species_selection()
        self.gene_crossover = gene_crossover
        self.latent_inheritance = latent_inheritance
        self.gene_mutation = gene_mutation or make_gaussian_mutation()
        self.latent_mutation = latent_mutation or make_gaussian_mutation()
        # Speciation (migration between functions) defaults OFF: measured
        # 2026-07-21, seeding every function plus NO background migration beat
        # both the grown-coverage start and seeded-with-migration on all seeds
        # (migration churn pays re-scorings to disrupt working species). The
        # designed successor is EVENT-DRIVEN recolonization on extinction,
        # relevant when the function count dwarfs the population; random
        # migration remains available for research via make_random_speciation.
        self.speciation = speciation
        self.children = int(children)
        self.win_target = float(win_target)
        self.dial_step = float(dial_step)
        self.mutation_memory = mutation_memory
        self.memory_drift = float(memory_drift)
        self.distill = distill
        self.direction_every = int(direction_every)
        self.direction_sigma = float(direction_sigma)
        self.immigrants = immigrants
        self.immigrant_patience = int(immigrant_patience)
        self.progress = progress
        self.progress_every = progress_every
        self.total_epochs = int(epochs)

        # Best-ever bookkeeping per function (never bred from).
        self.best_score = np.full(self.n_fns, -np.inf)
        self.founder_score = np.full(self.n_fns, np.nan)
        self.best_pheno: list[np.ndarray | None] = [None] * self.n_fns
        self.best_genes: list[np.ndarray | None] = [None] * self.n_fns
        self.fn_evals = np.zeros(self.n_fns, dtype=np.int64)
        self.spent = 0

        # Founders, random in BOTH spaces. "two" (the spec's default): a
        # single pair on the first function, coverage grows through
        # speciation. "per_function": founders on every function from the
        # start — the comparison arm for whether seeding every niche beats
        # growing into them.
        # `founders` sets how many of them per function (Daniel, 2026-07-26).
        # Two was the spec's number and it is a real limit, not a detail:
        # every individual that ever exists is descended from that pair, so
        # the run's entire coverage of the space is fixed at founding.
        # Raising the population cap does NOT fix this — it keeps a wider
        # cloud of the same two lineages' descendants. Where the score gives
        # no gradient to climb (MountainCarContinuous pays 0 until the goal),
        # fresh draws are the only thing that finds anything, which is why
        # blind sampling matched the GA there (FINDINGS sixteen, corrected).
        n_founders = max(2, int(founders))
        if founding == "per_function":
            population_cap = max(population_cap, n_founders * self.n_fns)
            n0 = n_founders * self.n_fns
            self.pop_fn = np.repeat(np.arange(self.n_fns), n_founders)
        else:
            population_cap = max(population_cap, n_founders)
            n0 = n_founders
            self.pop_fn = np.zeros(n0, dtype=np.int64)
        self.population_cap = int(population_cap)
        self.pop_genes = self.rng.standard_normal(
            (n0, self.genes_dim)).astype(np.float32)
        self.pop_latents = self.rng.standard_normal(
            (n0, self.latents_dim)).astype(np.float32)
        self.seeded = bool(caps.get("seeded"))
        # "sparse-shared" (round seven's designed arm): free placement and
        # full reach, but ONE run-level site set every individual shares — so
        # species' folds land in the same coordinates and compose instead of
        # colliding, and population-combining fold rules (sign vote, mean)
        # are expressible because coordinates mean the same thing for
        # everyone.
        self.shared_sites = bool(caps.get("shared_sites"))
        if self.shared_sites:
            self.pop_basis = np.full(
                n0, int(self.rng.integers(0, 2 ** 31)), dtype=np.int64)
            fresh_basis_rate = 0.0
        else:
            self.pop_basis = (self.rng.integers(0, 2 ** 31, n0) if self.seeded
                              else np.zeros(n0, dtype=np.int64))
        self.fresh_basis_rate = float(fresh_basis_rate)
        self.pop_score = self._score(
            self.pop_genes, self.pop_latents, self.pop_fn,
            self.pop_basis if self.seeded else None)

        # The step dials for the two mutation spaces are global (dense
        # feedback every epoch), start at 1.0, and self-tune by success rate
        # with ties counted as successes.
        self.gene_dial, self.latent_dial = 1.0, 1.0
        # Round-50 mutation memory: one accumulator per space. Every child
        # is a gradient sample — its birth delta, signed and scaled by the
        # fitness change it caused, failures included.
        self.mem = {
            "g": [np.zeros(self.genes_dim), np.zeros(self.genes_dim), 0, None],
            "l": [np.zeros(self.latents_dim), np.zeros(self.latents_dim),
                  0, None]}
        if consolidation is None:
            consolidation = Distillation(every=distill_every,
                                         steps=distill_steps,
                                         decay=distill_decay, lr=distill_lr)
        self.consolidation = consolidation
        # Direction evolution ((1+1)-ES on the shared low-rank vocabulary,
        # with an Adam memory over ACCEPTED changes so proposals drift along
        # what has historically worked — the round-50 pattern one level
        # deeper).
        self.dir_dial = 1.0
        self.dir_dim = (len(self.decoder.direction_vector())
                        if directions == "evolve" else 0)
        self.dir_m = np.zeros(self.dir_dim, dtype=np.float64)
        self.dir_v = np.zeros(self.dir_dim, dtype=np.float64)
        self.dir_t = 0
        self.history: list[dict] = []
        self.last_improved = np.zeros(self.n_fns, dtype=np.int64)
        self.prev_best = self.best_score.copy()
        self.epoch = 0
        self.wave: Wave | None = None

    # -------------------------------------------------------- engine laws

    def _score(self, genes_arr, latents_arr, fn_of, seeds=None) -> np.ndarray:
        """Decode once, score each phenotype on its own function, update
        the best-ever records, count evaluations."""
        decoder = self.decoder
        phenos = (decoder.decode_seeded(genes_arr, latents_arr, seeds)
                  if seeds is not None
                  else decoder.decode(genes_arr, latents_arr))
        values = np.empty(len(fn_of), dtype=np.float64)
        for f in np.unique(fn_of):
            picks = np.flatnonzero(fn_of == f)
            out = self.fns[int(f)](phenos[picks])
            v = np.asarray(out.detach().cpu().numpy()
                           if torch.is_tensor(out) else out, dtype=np.float64)
            values[picks] = v
            self.fn_evals[f] += len(picks)
            if np.isnan(self.founder_score[f]):
                self.founder_score[f] = float(v[0])
            top = int(np.argmax(v))
            if v[top] > self.best_score[f]:
                self.best_score[f] = float(v[top])
                self.best_pheno[int(f)] = (phenos[picks[top]]
                                           .detach().cpu().numpy().copy())
                self.best_genes[int(f)] = genes_arr[picks[top]].copy()
        self.spent += len(fn_of)
        return values

    def _memory_direction(self, key):
        m, v, steps, _ = self.mem[key]
        if steps == 0:
            return None
        m_hat = m / (1 - 0.9 ** steps)
        v_hat = v / (1 - 0.999 ** steps)
        return m_hat / (np.sqrt(v_hat) + 1e-8)

    def _memory_update(self, key, deltas, df):
        m, v, steps, df_scale = self.mem[key]
        mag = float(np.abs(df).mean())
        df_scale = mag if df_scale is None else 0.9 * df_scale + 0.1 * mag
        g = ((df / max(df_scale, 1e-12))[:, None] * deltas).mean(axis=0)
        self.mem[key] = [0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g,
                         steps + 1, df_scale]

    # ------------------------------------------------------------- stages

    def immigrate(self):
        """immigrants="stall" keeps founding-style fresh random draws
        flowing AFTER epoch zero: a function whose best-ever has not
        improved in `immigrant_patience` epochs receives one fresh random
        individual per epoch (scored honestly, competing on shares like
        anyone), and a function that has gone EXTINCT is re-founded the
        same way — the event-driven recolonization the speciation notes
        designed. Even an immigrant culled immediately has already
        contributed its evaluation to the best-ever record, which on
        plateau objectives is the entire value of a fresh draw."""
        if self.immigrants != "stall":
            return
        improved = self.best_score > self.prev_best + 1e-12
        self.last_improved[improved] = self.epoch
        self.prev_best = np.maximum(self.prev_best, self.best_score)
        alive = set(np.unique(self.pop_fn).tolist())
        stalled = [f for f in range(self.n_fns)
                   if f not in alive
                   or self.epoch - self.last_improved[f]
                   >= self.immigrant_patience]
        if not stalled:
            return
        n_new = len(stalled)
        im_genes = self.rng.standard_normal(
            (n_new, self.genes_dim)).astype(np.float32)
        im_latents = self.rng.standard_normal(
            (n_new, self.latents_dim)).astype(np.float32)
        im_fn = np.asarray(stalled, dtype=np.int64)
        im_basis = (self.rng.integers(0, 2 ** 31, n_new) if self.seeded
                    else np.zeros(n_new, dtype=np.int64))
        im_score = self._score(im_genes, im_latents, im_fn,
                               im_basis if self.seeded else None)
        self.pop_genes = np.concatenate([self.pop_genes, im_genes])
        self.pop_latents = np.concatenate([self.pop_latents, im_latents])
        self.pop_basis = np.concatenate([self.pop_basis, im_basis])
        self.pop_fn = np.concatenate([self.pop_fn, im_fn])
        self.pop_score = np.concatenate([self.pop_score, im_score])

    def breed(self):
        """Selection on fitness shares, gene crossover, latent
        inheritance. On seeded substrates (basis, latents) travel as one
        unit — the latents only mean anything relative to their basis, so
        seeded mode owns this choice; custom latent operators apply in
        the shared-vocabulary modes."""
        weights = fitness_shares(self.pop_score, self.pop_fn)
        a, b = self.selection(weights, self.pop_fn, self.rng, self.children)
        child_genes = self.gene_crossover(self.pop_genes[a],
                                          self.pop_genes[b], self.rng)
        if (self.seeded
                or self.latent_inheritance is coin_flip_latent_inheritance):
            pick_b = self.rng.random(self.children) < 0.5
            child_latents = np.where(pick_b[:, None], self.pop_latents[b],
                                     self.pop_latents[a]).astype(np.float32)
            child_basis = np.where(pick_b, self.pop_basis[b],
                                   self.pop_basis[a])
        else:
            child_latents = self.latent_inheritance(self.pop_latents[a],
                                                    self.pop_latents[b],
                                                    self.rng)
            child_basis = self.pop_basis[a].copy()
        if self.seeded and self.fresh_basis_rate > 0:
            fresh = self.rng.random(self.children) < self.fresh_basis_rate
            n_fresh = int(fresh.sum())
            if n_fresh:
                child_basis[fresh] = self.rng.integers(0, 2 ** 31, n_fresh)
                child_latents[fresh] = 0.0
        self.wave = Wave(a=a, b=b, genes=child_genes, latents=child_latents,
                         basis=child_basis)

    def mutate(self):
        """The two spaces are perturbed by different operators with
        independent self-tuning dials; each child mutates exactly one
        space so every dial's feedback is uncontaminated by the other."""
        w = self.wave
        mutates_latents = self.rng.random(self.children) < 0.5
        gi = np.flatnonzero(~mutates_latents)
        li = np.flatnonzero(mutates_latents)
        w.gene_rows, w.latent_rows = gi, li
        w.pre_genes = w.genes[gi].copy() if len(gi) else None
        w.pre_latents = w.latents[li].copy() if len(li) else None
        if len(gi):
            w.genes[gi] = self.gene_mutation(w.genes[gi], self.rng,
                                             self.gene_dial)
            if self.mutation_memory == "shared":
                direction = self._memory_direction("g")
                if direction is not None:
                    w.genes[gi] += (self.memory_drift * 0.12 * self.gene_dial
                                    * direction).astype(np.float32)
        if len(li):
            w.latents[li] = self.latent_mutation(w.latents[li], self.rng,
                                                 self.latent_dial)
            if self.mutation_memory == "shared":
                direction = self._memory_direction("l")
                if direction is not None:
                    w.latents[li] += (self.memory_drift * 0.12
                                      * self.latent_dial
                                      * direction).astype(np.float32)

    def score_and_adopt(self):
        """Scoring and function adoption. Same-function parents pass the
        function down; mixed parents have the child scored on both and it
        adopts the function on which it beats its parent by more."""
        w = self.wave
        w.fa, w.fb = self.pop_fn[w.a], self.pop_fn[w.b]
        w.fn = w.fa.copy()
        w.mixed = w.fa != w.fb
        w.score = np.empty(self.children, dtype=np.float64)
        same = np.flatnonzero(~w.mixed)
        if len(same):
            w.score[same] = self._score(
                w.genes[same], w.latents[same], w.fn[same],
                w.basis[same] if self.seeded else None)
        for i in np.flatnonzero(w.mixed):
            s_a = self._score(w.genes[i:i + 1], w.latents[i:i + 1],
                              w.fa[i:i + 1],
                              w.basis[i:i + 1] if self.seeded else None)[0]
            s_b = self._score(w.genes[i:i + 1], w.latents[i:i + 1],
                              w.fb[i:i + 1],
                              w.basis[i:i + 1] if self.seeded else None)[0]
            take_a = (s_a - self.pop_score[w.a[i]]) >= \
                (s_b - self.pop_score[w.b[i]])
            w.fn[i] = w.fa[i] if take_a else w.fb[i]
            w.score[i] = s_a if take_a else s_b

    def tune_dials(self):
        """Dial updates: the reference is the parent on the child's
        adopted function (raw scores compare legally within one
        function); ties count as successes."""
        w = self.wave
        ref = self.pop_score[w.a].copy()
        adopted_b = w.mixed & (w.fn == w.fb)
        ref[adopted_b] = self.pop_score[w.b][adopted_b]
        w.success = w.score >= ref - 1e-12
        gi, li = w.gene_rows, w.latent_rows
        if self.mutation_memory == "shared":
            improvement = w.score - ref
            if len(gi):
                self._memory_update("g", w.genes[gi] - w.pre_genes,
                                    improvement[gi])
            if len(li):
                self._memory_update("l", w.latents[li] - w.pre_latents,
                                    improvement[li])
        if len(gi):
            rate = float(w.success[gi].mean())
            self.gene_dial *= (self.dial_step if rate > self.win_target
                               else 1 / self.dial_step)
            self.gene_dial = float(np.clip(self.gene_dial, 1e-3, 1e4))
        if len(li):
            rate = float(w.success[li].mean())
            self.latent_dial *= (self.dial_step if rate > self.win_target
                                 else 1 / self.dial_step)
            self.latent_dial = float(np.clip(self.latent_dial, 1e-3, 1e4))

    def integrate(self):
        """Population cap: everyone competes on fitness share; the lowest
        shares are removed; a function may go extinct."""
        w = self.wave
        all_genes = np.concatenate([self.pop_genes, w.genes])
        all_latents = np.concatenate([self.pop_latents, w.latents])
        all_basis = np.concatenate([self.pop_basis, w.basis])
        all_fn = np.concatenate([self.pop_fn, w.fn])
        all_score = np.concatenate([self.pop_score, w.score])
        keep = np.argsort(
            -fitness_shares(all_score, all_fn))[:self.population_cap]
        self.pop_genes, self.pop_latents = all_genes[keep], all_latents[keep]
        self.pop_basis = all_basis[keep]
        self.pop_fn, self.pop_score = all_fn[keep], all_score[keep]

    def speciate(self):
        """Individuals drift onto other functions over time; a move is
        honest (re-scored on the new function immediately)."""
        if self.speciation is None:
            return
        new_fn = self.speciation(self.pop_fn, self.n_fns, self.epoch,
                                 self.rng)
        moved = np.flatnonzero(new_fn != self.pop_fn)
        if len(moved):
            self.pop_fn = new_fn
            self.pop_score[moved] = self._score(
                self.pop_genes[moved], self.pop_latents[moved],
                self.pop_fn[moved],
                self.pop_basis[moved] if self.seeded else None)

    def consolidate(self):
        """Consolidation (2026-07-30, Daniel: "remove folding if it's
        never been helpful and instead iterate on distillation"). The
        arithmetic fold was searched for at short and long budgets, both
        substrates, alone and alongside distillation (~70 paired runs)
        and no configuration was found where it helps; it is gone.
        Consolidation is DISTILLATION alone, on this event's cadence:
        each function's best-ever (genes -> phenotype) pair joins a
        replay buffer, the BASE decoder takes Adam steps toward it with
        zero per-individual modifier, every modifier then decays (the
        discovery lives in the base now), and the population is honestly
        re-scored. Multi-function only (measured: 10/10 seeds, t=+16.7;
        single-function t=-1.38) and only where all individuals share
        one coordinate system."""
        distill_on = (self.distill == "on"
                      or (self.distill == "auto" and self.n_fns >= 2))
        if not (distill_on and self.consolidation.due(self.epoch)
                and ((not self.seeded) or self.shared_sites)
                and hasattr(self.decoder, "training_logits")):
            return
        self.consolidation.run(self.decoder, self.best_genes, self.best_pheno)
        # The absorbed discoveries live in the base now, so every bending
        # shrinks; decay 1.0 (no shrink) measured +49% worse.
        self.pop_latents *= float(self.consolidation.decay)
        self.pop_score = self._score(
            self.pop_genes, self.pop_latents, self.pop_fn,
            self.pop_basis if self.seeded else None)

    def evolve_directions(self):
        """Direction evolution: trial a perturbation of the shared
        vocabulary itself. Accept if the living population, re-scored
        under the new directions, is on average better (per-individual
        relative change, so no function's scale dominates); reject
        restores the old vocabulary and the cached scores exactly. The
        trial's re-scoring is charged to the budget either way."""
        if (self.directions != "evolve"
                or (self.epoch + 1) % self.direction_every != 0):
            return
        base_dir = self.decoder.direction_vector()
        scale = max(float(base_dir.std()), 1e-4)
        noise = self.rng.standard_normal(self.dir_dim)
        step = self.direction_sigma * scale * self.dir_dial * noise
        if self.dir_t > 0:
            drift = (self.dir_m / (1 - 0.9 ** self.dir_t)) / (
                np.sqrt(self.dir_v / (1 - 0.999 ** self.dir_t)) + 1e-8)
            step = step + (0.5 * self.direction_sigma * scale
                           * self.dir_dial * drift)
        self.decoder.set_direction_vector(
            (base_dir + step).astype(np.float32))
        old_scores = self.pop_score.copy()
        trial_scores = self._score(self.pop_genes, self.pop_latents,
                                   self.pop_fn)
        gain_rel = np.mean((trial_scores - old_scores)
                           / np.maximum(np.abs(old_scores), 1e-12))
        if gain_rel >= 0:
            self.pop_score = trial_scores
            self.dir_dial = min(self.dir_dial * self.dial_step, 1e3)
            self.dir_t += 1
            self.dir_m = 0.9 * self.dir_m + 0.1 * step
            self.dir_v = 0.999 * self.dir_v + 0.001 * step * step
        else:
            self.decoder.set_direction_vector(base_dir)
            self.pop_score = old_scores
            self.dir_dial = max(self.dir_dial / self.dial_step, 1e-3)

    def record(self):
        """Progress reporting and the run's own history; closes the
        epoch."""
        if self.progress is not None and (self.epoch + 1) % (
                self.progress_every
                or max(1, self.total_epochs // 50)) == 0:
            # phenotypes reach the callback in output_shape (same bytes as
            # the flat storage), so consumers like live_progress can
            # render them without knowing the run's shape
            self.progress(self.epoch + 1, self.total_epochs, int(self.spent),
                          [None if b is None else
                           b.reshape(self.output_shape).copy()
                           for b in self.best_pheno],
                          self.best_score.copy())
        self.history.append({
            "epoch": self.epoch,
            "evaluations": int(self.spent),
            "population": int(len(self.pop_fn)),
            "functions_alive": int(len(np.unique(self.pop_fn))),
            "functions_tried": int(np.isfinite(self.best_score).sum()),
            "gene_dial": float(self.gene_dial),
            "latent_dial": float(self.latent_dial),
            "direction_dial": float(self.dir_dial),
            "mean_score": float(self.pop_score.mean()),
        })
        self.epoch += 1

    # ------------------------------------------------------------ driving

    def step(self):
        """One epoch: every stage, in the canonical order."""
        for stage in self.STAGES:
            getattr(self, stage)()

    def result(self) -> GAResult:
        problems = [ProblemResult(
            best_phenotype=(None if self.best_pheno[f] is None
                            else self.best_pheno[f].reshape(
                                self.output_shape)),
            best_fitness=float(self.best_score[f]),
            initial_fitness=(float(self.founder_score[f])
                             if np.isfinite(self.best_score[f])
                             else float("nan")),
            evaluations=int(self.fn_evals[f]),
        ) for f in range(self.n_fns)]
        return GAResult(problems=problems, evaluations=int(self.spent),
                        epochs=self.total_epochs, history=self.history,
                        decoder=self.decoder.get_params())

    # ----------------------------------------------------- engine protocol
    # The minimal surface Environment reads from any engine, duck-typed:
    # best_summary() -> {name: score}, best_record() -> best-ever object,
    # default_generations -> how many generations one evolve() means.

    @property
    def default_generations(self) -> int:
        return self.total_epochs

    def best_summary(self) -> dict:
        return {f"fn{i}": float(s) for i, s in enumerate(self.best_score)
                if np.isfinite(s)}

    def best_record(self):
        if not np.isfinite(self.best_score).any():
            return None
        f = int(np.argmax(self.best_score))
        return ProblemResult(
            best_phenotype=self.best_pheno[f].reshape(self.output_shape),
            best_fitness=float(self.best_score[f]),
            initial_fitness=float(self.founder_score[f]),
            evaluations=int(self.fn_evals[f]))


# --------------------------------------------------------------- the loop

def solve(fitness_fns, output_shape, epochs=1_000, architecture="auto",
          genes=64, latents=None, children=16, population_cap=32,
          device="auto", selection=None,
          gene_crossover=one_point_gene_crossover,
          latent_inheritance=coin_flip_latent_inheritance,
          gene_mutation=None, latent_mutation=None,
          speciation=None,
          directions="sparse-shared", direction_every=16,
          direction_sigma=0.1, fresh_basis_rate=0.1,
          win_target=0.2, dial_step=1.15,
          mutation_memory="off", memory_drift=0.5,
          distill="auto", distill_every=64,
          distill_steps=40, distill_decay=0.7, distill_lr=1e-3,
          consolidation=None,
          founding="per_function", founders=16,
          immigrants="off", immigrant_patience=32,
          progress=None, progress_every=None,
          init_decoder=None, seed=None) -> GAResult:
    """Maximize every fitness function over phenotypes of `output_shape`.

    This is sugar: it builds a TensorGA engine and runs its canonical
    stage order for `epochs` epochs. finch4.layers.tensor_environment
    drives the same engine through the layer grammar instead.

    fitness_fns: one callable or a list. Each takes a torch tensor
        (B, *output_shape) with values in [0, 1] and returns B fitness
        values, higher better. `epochs` counts loop iterations; each epoch
        breeds `children` children. Evaluation counts (one per scoring) are
        tracked and reported for honest cross-method comparison.
    genes / latents: sizes of the two spaces. Genes are the decoder's
        input; latents bend the shared decoder per individual. `latents`
        means PATCH SIZE K on the sparse paths and gate count on the
        low-rank paths, so its default resolves per substrate (2048 /
        64, the measured-best of each).
    directions: "sparse-shared" (DEFAULT since 2026-07-27, by round
        seven's pre-registered rule: keeps the apple win — 0.0113 vs
        frozen 0.0177, 3/3 paired seeds — and matches frozen on
        multi-function, 10 paired seeds, t=-0.32): each individual's
        latents are values added at K weight coordinates drawn ONCE per
        run, so every species edits the same coordinates and
        consolidation has one coordinate system to train. "frozen" (the
        prior default,
        low-rank gating) reproduces all benchmarks recorded before
        2026-07-27. "sparse" replaces low-rank bending
        with a per-individual SPARSE WEIGHT PATCH (Daniel, 2026-07-22):
        the individual's seed picks `latents` coordinates of the decoder's
        weight vector and its latents are the values added there, so
        edits can reach any weight instead of being trapped in a frozen
        random subspace forever. Locations inherit with the latents; a
        `fresh_basis_rate` fraction of children draw new ones. "evolve" trials perturbations of the
        shared low-rank vocabulary as a (1+1) evolution strategy —
        FALSIFIED as built (apple, 171k evals: 0.01268 vs frozen 0.01222;
        ~560 trials, essentially all rejected, because a random
        perturbation of the whole vocabulary almost never survives a
        population-mean vote over 32 co-adapted individuals — it froze
        itself and paid a 12% trial tax). Kept for iteration; the
        designed refinements are one-direction-at-a-time proposals, a
        share-weighted acceptance signal, and trialing right after
        consolidation when the vocabulary is least load-bearing.
    mutation_memory: "shared" pools every child's birth delta SIGNED BY
        ITS FITNESS CHANGE — failures included — into one Adam-style
        accumulator per space (genes and latents separately, never mixed),
        and later mutations drift along the accumulated direction
        (`memory_drift` as a fraction of the mutation step). This is round
        50's mechanism (the legacy engine's strongest result: image 1.38x
        at t=3.88, the first sub-0.002 apple) ported to the redesign.
        "off" disables it.
    distill: "on" adds GRADIENT DISTILLATION on the consolidation cadence
        (Daniel, 2026-07-27: the fold's arithmetic path measured unproven
        while the gradient path carried every decoder-learning win —
        "maybe we need a combo of both"; 2026-07-30: the fold is gone and
        distillation IS consolidation). The decoder's BASE weights (never
        the shared direction vocabulary — training that would re-define
        every individual's latents mid-run) take `distill_steps` Adam
        steps toward a replay buffer of each function's best-ever
        (genes -> phenotype) pair. The fitness function is never
        differentiated — the black-box constraint is on fitness only.
        Costs ZERO fitness evaluations (targets are already scored) plus
        one honest population re-score per event. "auto" (default)
        enables it on multi-function runs only.
    immigrants: "stall" keeps founding-style fresh random draws flowing
        AFTER epoch zero: a function whose best-ever has not improved in
        `immigrant_patience` epochs receives one fresh random individual
        per epoch (scored honestly, competing on shares like anyone), and a
        function that has gone EXTINCT is re-founded the same way — the
        event-driven recolonization the speciation notes designed.
        `founders` fixes coverage at epoch zero; this is the same medicine
        at every later epoch. Even an immigrant culled immediately has
        already contributed its evaluation to the best-ever record, which
        on plateau objectives is the entire value of a fresh draw. "off"
        disables (default pending measurement).
    init_decoder: a previous run's `GAResult.decoder` vector to warm-start
        the shared decoder from (transfer). Measured: helps related image
        families at every checkpoint; does NOT transfer on locomotion.
    The step dials for the two mutation spaces are global (dense feedback
    every epoch), start at 1.0, and self-tune by success rate with ties
    counted as successes. All other operators are the module-level defaults
    and are replaceable via the parameters above.
    """
    engine = TensorGA(
        fitness_fns, output_shape, epochs=epochs, architecture=architecture,
        genes=genes, latents=latents, children=children,
        population_cap=population_cap, device=device, selection=selection,
        gene_crossover=gene_crossover, latent_inheritance=latent_inheritance,
        gene_mutation=gene_mutation, latent_mutation=latent_mutation,
        speciation=speciation, directions=directions,
        direction_every=direction_every, direction_sigma=direction_sigma,
        fresh_basis_rate=fresh_basis_rate, win_target=win_target,
        dial_step=dial_step, mutation_memory=mutation_memory,
        memory_drift=memory_drift, distill=distill,
        distill_every=distill_every, distill_steps=distill_steps,
        distill_decay=distill_decay, distill_lr=distill_lr,
        consolidation=consolidation, founding=founding, founders=founders,
        immigrants=immigrants, immigrant_patience=immigrant_patience,
        progress=progress, progress_every=progress_every,
        init_decoder=init_decoder, seed=seed)
    for _ in range(int(epochs)):
        engine.step()
    return engine.result()
