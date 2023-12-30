from Finch.environments import Sequential, AdaptiveEnvironment
from Finch.genepools import FloatPool
from Finch.layers import *


def fit(individual):
    return sum(individual.genes)  # -(individual.genes[-1]*4)  <-- You can add something like that to make it
    # interesting


# Config section
length = 100
pool_minimum = 0
pool_maximum = 10
population_size = 100
amount_to_mutate = 10
gene_selection = 5
parent_count = 20
children_count = 2
max_population = 99
evolution_steps = 1000
min_mutation = -2
max_mutation = 2

# Creating the FloatPool
pool = FloatPool(length=length, minimum=pool_minimum, maximum=pool_maximum)

# Creating the Sequential environment
environment = Sequential(layers=[
    Populate(pool, population=population_size),
    FloatMutateRange(individual_selection=amount_to_mutate, gene_selection=gene_selection,
                     min_mutation=min_mutation, max_mutation=max_mutation, keep_within_genepool_bounds=True),
    ParentSimple(parent_count, children=children_count),
    SortByFitness(),
    CapPopulation(max_population=max_population),
], name="standard")

environment2 = Sequential(layers=[
    Populate(pool, population=population_size),
    FloatMutateRange(individual_selection=amount_to_mutate, gene_selection=gene_selection,
                     min_mutation=min_mutation, max_mutation=max_mutation, keep_within_genepool_bounds=True),
    #ParentSimple(parent_count, children=children_count*2),
    SortByFitness(),
    CapPopulation(max_population=max_population),
], name="no children")

environment3 = Sequential(layers=[
    Populate(pool, population=population_size),
    #FloatMutateRange(individual_selection=amount_to_mutate, gene_selection=gene_selection,
    #                 min_mutation=min_mutation, max_mutation=max_mutation, keep_within_genepool_bounds=True),
    ParentSimple(parent_count, children=children_count*2),
    SortByFitness(),
    CapPopulation(max_population=max_population),
], name="no mutation")

adversary = AdaptiveEnvironment([environment, environment2, environment3])

if __name__ == "__main__":
    # Compiling the environment with the fitness function
    adversary.compile(fitness_function=fit)
    adversary.evolve(evolution_steps)

    # Printing the best individual
    print(environment.fitness)
    print(environment2.fitness)
    print(environment3.fitness)

    # Plotting the environment
    adversary.plot()
