from Finch.environmental import environments, layers
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
    layers.MutateAmount(2, 300, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=strict_rank),
    layers.SortByFitness(),
    layers.CapPopulation(99),
], name="B: parents only best")
pure_strict_env = environments.Sequential(layers=[
    layers.Populate(pool, 100),
    layers.MutateAmount(2, 300, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=strict_rank),
    layers.SortByFitness(),
    layers.CapPopulation(99),
], name="B: parents only best")
v_low_c = environments.Sequential(layers=[
    layers.Populate(pool, 2),
    layers.MutateAmount(2, 5, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(3),
], name="A: low population but parents everyone")
default = environments.Sequential(layers=[
    layers.Populate(pool, 2),
    layers.MutateAmount(2, 5, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(3),
], name="A: low population but parents everyone")
env = environments.MixedEnvironment([(strict_env, 150), (v_low_c, 1850)], verbose_every=700, name="Switches from A to B at 150")

adversarial = environments.Adversarial([env, default, pure_strict_env], 2000)
adversarial.evolve()