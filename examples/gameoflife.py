import numpy as np
from Finch.generic import Environment, Individual
from Finch.layers.universal_layers import Populate, SortByFitness, CapPopulation
from Finch.selectors import RandomSelection, RankBasedSelection
from Finch.layers.array_layers import ArrayPool, ParentNPoint, SwapMutation
import matplotlib.pyplot as plt

# Define the grid size
GRID_SIZE = 50

# Define the Game of Life rules
def apply_rules(grid):
    new_grid = grid.copy()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            total = int((grid[i, (j-1)%GRID_SIZE] + grid[i, (j+1)%GRID_SIZE] +
                         grid[(i-1)%GRID_SIZE, j] + grid[(i+1)%GRID_SIZE, j] +
                         grid[(i-1)%GRID_SIZE, (j-1)%GRID_SIZE] + grid[(i-1)%GRID_SIZE, (j+1)%GRID_SIZE] +
                         grid[(i+1)%GRID_SIZE, (j-1)%GRID_SIZE] + grid[(i+1)%GRID_SIZE, (j+1)%GRID_SIZE]))
            if grid[i, j] == 1:
                if (total < 2) or (total > 3):
                    new_grid[i, j] = 0
            else:
                if total == 3:
                    new_grid[i, j] = 1
    return new_grid

# Define the fitness function
def fitness_function(individual):
    grid = individual.item.reshape((GRID_SIZE, GRID_SIZE))
    next_gen = apply_rules(grid)
    return np.sum(next_gen)  # Return the number of live cells in the next generation

# Create the gene pool
pool = ArrayPool(gene_array=np.array([0, 1]), fitness_function=fitness_function, length=GRID_SIZE**2)

# Define the layers
layers = [
    Populate(population=100, gene_pool=pool),
    SwapMutation(selection_function=RandomSelection(amount_to_select=10).select),
    ParentNPoint(selection_function=RankBasedSelection(amount_to_select=2, factor=1).select, families=8, children=2, n_points=3),
    SortByFitness(),
    CapPopulation(max_population=100)
]

# Create and compile the environment
env = Environment(layers=layers, verbose_every=10)
env.compile()

# Evolve the population
env.evolve(generations=500)

# Get the best individual
best = env.best_ever
best_grid = best.item.reshape((GRID_SIZE, GRID_SIZE))

# Visualize the best grid
plt.imshow(apply_rules(best_grid), cmap='binary')
plt.title("Best Game of Life Grid")
plt.show()
