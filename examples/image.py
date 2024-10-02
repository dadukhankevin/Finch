"""
Note: This example does not work well yet.
"""
import numpy as np
from PIL import Image
from Finch.generic import Environment, Individual
from Finch.layers.universal_layers import Populate, SortByFitness, CapPopulation
from Finch.selectors import RankBasedSelection, RandomSelection
from Finch.layers.float_arrays import FloatPool, GaussianMutation, UniformMutation, ParentBlendFloat

# Load the target image

height, width, channels = 250, 250, 3
# Define the fitness functions
def fitness_function(individual):
    e = sum(individual.item)
    return e


# Create the float pool
pool = FloatPool(ranges=[[0, 0]] * (height * width * channels),
                 length=height * width * channels,
                 fitness_function=fitness_function)

# Define the layers
layers = [
    Populate(population=50, gene_pool=pool),
    UniformMutation(
        mutation_rate=0.1,
        upper_bound=.1,
        lower_bound=-.1,
        selection_function=RandomSelection(amount_to_select=16).select,
    ),
    ParentBlendFloat(selection_function=RankBasedSelection(amount_to_select=2, factor=4).select, families=4, children=2),
    SortByFitness(),

    CapPopulation(max_population=50)
]

# Create the environment
env = Environment(layers=layers, verbose_every=10)

# Compile the environment
env.compile()

# Evolve the population
env.evolve(generations=2000)

# Get the best individual
best = env.best_ever
# Reshape and denormalize the best individual
best_image = (best.item.reshape(height, width, channels) * 255).astype(np.uint8)

# Save the best image
Image.fromarray(best_image).save("evolved_image.png")

# Print the final fitness
print(f"Final fitness: {best.fitness}")

# Plot the fitness history
env.plot()