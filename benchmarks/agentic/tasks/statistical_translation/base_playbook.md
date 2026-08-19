# Statistical translation methodology — version 2

Build a deterministic, generic translator that learns only from the complete
parallel training set and held-out source queries supplied in the current
batch. Treat retrieval, bilingual induction,
target-side sequence modeling, and decoding as separate mechanisms whose
errors can be inspected independently.

1. Decide how much of the supplied training set to use. Retrieval is optional
   internal modeling, not an externally imposed ten-example stage.
2. Infer bilingual word, character, or phrase relationships from the supplied
   aligned pairs. Never assume a language-specific dictionary.
3. Use a non-neural target-side model derived only from supplied targets to
   prefer plausible sequences and ordering.
4. Decode a complete hypothesis while accounting for source coverage,
   insertions, deletions, reordering, and uncertainty.
5. Work in both natural and independently word-coded text. A useful mechanism
   should transfer when language identity and pretrained priors disappear.
6. Treat the frozen position-aware retrieval-copy baseline as a comparison
   control, never an inherited default. A viable model must combine evidence
   from separate pairs on novel-conjunction episodes.
7. Be creative about representation, segmentation, alignment, candidate
   generation, decoding, and uncertainty. Every new mechanism needs a reachable
   activation condition, fallback guardrail, causal prediction, and falsifier.
   When a lineage plateaus, challenge a shared assumption or inject a new
   statistical framing instead of spending the campaign on coefficient polish.

The scorer, data boundary, no-neural rule, episode reset, and fitness formula
are campaign laws in `CONTRACT.md`; they are not evolvable methodology.
