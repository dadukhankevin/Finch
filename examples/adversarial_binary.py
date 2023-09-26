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


no_m = environments.Sequential(layers=[
    layers.Populate(pool, 100),
    #layers.MutateAmount(1, 300, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(99),
], name="no mutation")

standard = environments.Sequential(layers=[
    layers.Populate(pool, 100),
    mutation_layers.MutateAmount(2, 300, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(99),
], name="standard")

duplicate = environments.Sequential(layers=[
    layers.Populate(pool, 100),
    mutation_layers.MutateAmount(2, 300, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=strict_rank),
    layers.SortByFitness(),
    layers.CapPopulation(99),
], name="strict rank towards high fitness (parenting)")

no_p = environments.Sequential(layers=[
    layers.Populate(pool, 10),
    mutation_layers.MutateAmount(2, 10, pool, 1, rank),
    #layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(9),
], name="no parenting")

low_c = environments.Sequential(layers=[
    layers.Populate(pool, 5),
    mutation_layers.MutateAmount(2, 5, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(4),
], name="low population (5)")

freeze_layer = environments.Sequential(layers=[
    layers.Populate(pool, 5),
    mutation_layers.MutateAmount(2, 5, pool, 1, rank),
    layers.FreezeRandom(1, 500),
    layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(4),
], name="Same but freeze")

env = environments.Adversarial([low_c, freeze_layer],"adversarial")
env.compile(verbose_every=1000)
env.evolve(2000)
# print(env.history[-1])
# print(env.individuals[-1].genes)
# plt.plot(env.history)
# plt.show()
