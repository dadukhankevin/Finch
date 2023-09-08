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


a = environments.Sequential(layers=[
    layers.Populate(pool, 100),
    #layers.MutateAmount(1, 300, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(99),
], name="no mutation")

b = environments.Sequential(layers=[
    layers.Populate(pool, 100),
    layers.MutateAmount(2, 300, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(99),
], name="standard")

c = environments.Sequential(layers=[
    layers.Populate(pool, 100),
    layers.MutateAmount(2, 300, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=strict_rank),
    layers.SortByFitness(),
    layers.CapPopulation(99),
], name="strict rank towards high fitness (parenting)")

d = environments.Sequential(layers=[
    layers.Populate(pool, 10),
    layers.MutateAmount(2, 10, pool, 1, rank),
    #layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(9),
], name="no parenting")

e = environments.Sequential(layers=[
    layers.Populate(pool, 5),
    layers.MutateAmount(2, 5, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(4),
], name="low population (5)")

f = environments.Sequential(layers=[
    layers.Populate(pool, 2),
    layers.MutateAmount(2, 5, pool, 1, rank),
    layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    layers.SortByFitness(),
    layers.CapPopulation(3),
], name="ultra low population (3)")

env = environments.AdaptiveEnvironment([a, b, c, d], go_for=1, switch_every=1)
env.compile(verbose_every=500)

env.evolve(2000)
env.plot()
# print(env.history[-1])
# print(env.individuals[-1].genes)
# plt.plot(env.history)
# plt.show()
