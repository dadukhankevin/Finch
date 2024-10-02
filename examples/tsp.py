import numpy as np
from Finch.generic import Environment, Individual, Layer
from Finch.layers.universal_layers import Populate, SortByFitness, CapPopulation
from Finch.selectors import RandomSelection, RankBasedSelection
from Finch.layers.array_layers import ArrayPool
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(10000)

# Define the number of cities and their coordinates
NUM_CITIES = 200
CITIES = np.random.rand(NUM_CITIES, 2)  # Random 2D coordinates for cities

# Calculate distances between all pairs of cities
DISTANCES = np.sqrt(((CITIES[:, np.newaxis, :] - CITIES[np.newaxis, :, :]) ** 2).sum(axis=2))

def calculate_total_distance(route):
    return sum(DISTANCES[route[i], route[(i + 1) % NUM_CITIES]] for i in range(NUM_CITIES))

# Define the fitness function (we want to minimize the total distance)
def fitness_function(individual):
    return -calculate_total_distance(individual.item)

# Create the gene pool
pool = ArrayPool(gene_array=np.arange(NUM_CITIES), fitness_function=fitness_function, length=NUM_CITIES, unique=True)

# Custom crossover operator for TSP (Order Crossover - OX)
class OrderCrossover(Layer):
    def __init__(self, selection_function, families, children):
        super().__init__(application_function=self.crossover, selection_function=selection_function, repeat=families)
        self.children = children

    def crossover(self, parents):
        for _ in range(self.children):
            parent1, parent2 = parents
            # Choose two random crossover points
            a, b = sorted(np.random.choice(NUM_CITIES, 2, replace=False))
            # Create a child with a segment from parent1
            child = [-1] * NUM_CITIES
            child[a:b] = parent1.item[a:b]
            # Fill the remaining positions with cities from parent2
            parent2_cities = [city for city in parent2.item if city not in child[a:b]]
            for i in range(NUM_CITIES):
                if child[i] == -1:
                    child[i] = parent2_cities.pop(0)
            # Add the child to the population
            self.environment.add_individuals([Individual(item=np.array(child), fitness_function=fitness_function)])

# Custom mutation operator for TSP (2-opt mutation)
class TwoOptMutation(Layer):
    def __init__(self, selection_function, mutation_rate=0.1):
        super().__init__(application_function=self.mutate_all, selection_function=selection_function)
        self.mutation_rate = mutation_rate

    def mutate_all(self, individuals):
        for individual in individuals:
            if np.random.random() < self.mutation_rate:
                self.mutate(individual)

    def mutate(self, individual):
        route = individual.item
        # Choose two random points
        i, j = sorted(np.random.choice(NUM_CITIES, 2, replace=False))
        # Reverse the segment between i and j
        route[i:j+1] = route[i:j+1][::-1]
        individual.item = route

# Define the layers
layers = [
    Populate(population=100, gene_pool=pool),
    OrderCrossover(selection_function=RankBasedSelection(amount_to_select=2, factor=2).select, families=8, children=2),
    TwoOptMutation(selection_function=RandomSelection(percent_to_select=0.1).select, mutation_rate=0.1),
    SortByFitness(),
    CapPopulation(max_population=100)
]

# Create and compile the environment
env = Environment(layers=layers, verbose_every=100)
env.compile()

# Evolve the population
env.evolve(generations=10000)

# Get the best solution
best = env.best_ever
best_route = best.item
best_distance = -best.fitness

print(f"Best route found: {best_route}")
print(f"Total distance: {best_distance}")

# Plot the best route
plt.figure(figsize=(10, 10))
plt.scatter(CITIES[:, 0], CITIES[:, 1], c='red', s=50)
for i in range(NUM_CITIES):
    plt.annotate(str(i), (CITIES[i, 0], CITIES[i, 1]))
for i in range(NUM_CITIES):
    start = best_route[i]
    end = best_route[(i + 1) % NUM_CITIES]
    plt.plot([CITIES[start, 0], CITIES[end, 0]], [CITIES[start, 1], CITIES[end, 1]], 'b-')

plt.title(f"Best TSP Route (Distance: {best_distance:.2f})")
plt.show()

# Plot the fitness history
env.plot()