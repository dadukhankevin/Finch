from Finch.generic import Environment, Individual
from Finch.layers.universal_layers import Populate, SortByFitness, CapPopulation
from Finch.selectors import RandomSelection, RankBasedSelection
from Finch.layers.array_layers import ParentNPoint, InsertionDeletionMutation, ArrayPool, SwapMutation, ReplaceMutation
from difflib import SequenceMatcher
import string

# Define the target sentence
TARGET = "genetic algos are lit"

# Define the character set (lowercase letters and space)
CHAR_SET = string.ascii_lowercase + " "

# Define the fitness function
def fitness_function(individual):
    return SequenceMatcher(None, TARGET, ''.join(individual.item)).ratio() * 100

# Create the gene pool
pool = ArrayPool(gene_array=list(CHAR_SET), fitness_function=fitness_function, length=len(TARGET))

# Define the layers
layers = [
    Populate(population=100, gene_pool=pool),
    ParentNPoint(selection_function=RankBasedSelection(amount_to_select=2, factor=2).select, families=8, children=2, refit=True),
    # InsertionDeletionMutation(gene_pool=pool, selection_function=RandomSelection(percent_to_select=0.2).select, overpowered=True, refit=False), # If overpowered is True ALWAYS set refit to False.
    SwapMutation(selection_function=RandomSelection(percent_to_select=0.2).select, overpowered=True,
                              refit=False),  # If overpowered is True ALWAYS set refit to False.
    ReplaceMutation(mutation_rate=.1, possible_values=list(CHAR_SET),
                    selection_function=RandomSelection(percent_to_select=0.2).select, overpowered=True, refit=False),
    SortByFitness(),
    CapPopulation(max_population=900)  # Kill 1 so that 1 is generated randomly again, can help diversity
]

# Create the environment

env = Environment(layers=layers, verbose_every=10)

# Compile the environment
env.compile()

# Evolve the population
env.evolve(generations=2000)

# Print the best individual
best = env.best_ever
print(f"Best solution: '{''.join(best.item)}'")
print(f"Fitness: {best.fitness}")

# Plot the fitness history
env.plot()