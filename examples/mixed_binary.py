from Finch.environmental import environments
from Finch.environmental.layers import standard_layers as layers
from Finch.environmental.layers import mutation_layers
from Finch.genetics import genepools
from Finch.tools.fitness_functions import MeanSquaredErrorLoss as MSE
from Finch.functions import selection
from Finch.genetics.population import NPCP
from Finch.tools import rates

rate = rates.Rate(300, 1, 8000, 1)
print(rate.next())
inner = MSE().func

binary_array = NPCP.random.randint(2, size=10000)


def fit(individual):
    return inner(individual, binary_array)


rank = selection.RankBasedSelection(1).select
strict_rank = selection.RankBasedSelection(100).select

pool = genepools.BinaryPool(10000, fit)

strict_env = environments.Sequential(layers=[
    layers.Populate(pool, 100),
    mutation_layers.MutateAmount(2, 300, pool, 1, rank),
    layers.ParentSinglePointCrossover(4, 2, selection_function=strict_rank),
    layers.SortByFitness(),
    layers.CapPopulation(99),
], name="B: parents only best")
strict_env.compile(verbose_every=500)
pure_strict_env = environments.Sequential(layers=[
    layers.Populate(pool, 100),
    mutation_layers.MutateAmount(2, 300, pool, 1, rank),
    layers.ParentSinglePointCrossover(4, 2, selection_function=strict_rank),
    layers.SortByFitness(),
    layers.CapPopulation(99),
], name="B: parents only best")
pure_strict_env.compile(verbose_every=500)

v_low_c = environments.Sequential(layers=[
    layers.Populate(pool, 2),
    mutation_layers.MutateAmount(2, 5, pool, 1, rank),
    layers.ParentSinglePointCrossover(4, 2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(3),
], name="A: low population but parents everyone")
v_low_c.compile(verbose_every=500)

default = environments.Sequential(layers=[
    layers.Populate(pool, 2),
    mutation_layers.MutateAmount(2, 5, pool, 1, rank),
    layers.ParentSinglePointCrossover(4, 2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(3),
], name="A: low population but parents everyone")
default.compile(verbose_every=500)

env = environments.ChronologicalEnvironment([(strict_env, 150), (v_low_c, 1850)], name="Switches from A to B at 150")
env.compile(verbose_every=500)

adversarial = environments.Adversarial([env, default, pure_strict_env])
adversarial.evolve(2000)
