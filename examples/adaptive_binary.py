from matplotlib import pyplot as plt
from Finch.environmental import environments
from Finch.environmental.layers import *
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
    Populate(pool, 100),
    #layers.MutateAmount(1, 300, pool, 1, rank),
    ParentSinglePointCrossover(4,2, selection_function=rank),
    SortByFitness(),
    CapPopulation(99),
], name="no mutation")

b = environments.Sequential(layers=[
    Populate(pool, 100),
    mutation_layers.MutateAmount(2, 300, pool, 1, rank),
    ParentSinglePointCrossover(4,2, selection_function=rank),
    SortByFitness(),
    CapPopulation(99),
], name="standard")

c = environments.Sequential(layers=[
    Populate(pool, 100),
    mutation_layers.MutateAmount(2, 300, pool, 1, rank),
    ParentSinglePointCrossover(4,2, selection_function=strict_rank),
    SortByFitness(),
    CapPopulation(99),
], name="strict rank towards high fitness (parenting)")

d = environments.Sequential(layers=[
    Populate(pool, 100),
    mutation_layers.MutateAmount(2, 10, pool, 1, rank),
    #layers.ParentSinglePointCrossover(4,2, selection_function=rank),
    SortByFitness(),
    CapPopulation(99),
], name="no parenting")

e = environments.Sequential(layers=[
    Populate(pool, 5),
    mutation_layers.MutateAmount(2, 5, pool, 1, rank),
    ParentSinglePointCrossover(4,2, selection_function=rank),
    SortByFitness(),
    CapPopulation(4),
], name="low population (5)")

f = environments.Sequential(layers=[
    Populate(pool, 2),
    mutation_layers.MutateAmount(2, 5, pool, 1, rank),
    ParentSinglePointCrossover(4,2, selection_function=rank),
    SortByFitness(),
    CapPopulation(3),
], name="ultra low population (3)")

env = environments.AdaptiveEnvironment([a, b, c], go_for=100, switch_every=30)
env.compile(verbose_every=500)

env.evolve(1000)
env.plot()
for envs in env.environments:
    plt.title(envs.name)
    plt.plot(envs.layer_history)
    plt.show()

# print(env.history[-1])
# print(env.individuals[-1].genes)
# plt.plot(env.history)
# plt.show()
