from Finch.generic import Environment, Individual
from Finch.layers.universal_layers import Populate, SortByFitness, CapPopulation
from Finch.selectors import RankBasedSelection, RandomSelection
from Finch.layers.array_layers import ArrayPool, ParentNPoint, SwapMutation, ReplaceMutation, InversionMutation
import numpy as np

# Define the backpack problem parameters
items = [
    {"name": "Laptop", "weight": 3, "value": 10},
    {"name": "Headphones", "weight": 0.5, "value": 3},
    {"name": "Book", "weight": 1, "value": 1},
    {"name": "Water Bottle", "weight": 1, "value": 2},
    {"name": "Snacks", "weight": 0.5, "value": 1},
    {"name": "Camera", "weight": 2, "value": 4},
    {"name": "First Aid Kit", "weight": 1, "value": 9},
    {"name": "Jacket", "weight": 1, "value": 8},
    {"name": "Flashlight", "weight": 0.5, "value": 6},
    {"name": "Portable Charger", "weight": 0.5, "value": 5},
    {"name": "Smartphone", "weight": 0.3, "value": 9},
    {"name": "Tablet", "weight": 1, "value": 7},
    {"name": "Sunglasses", "weight": 0.2, "value": 2},
    {"name": "Umbrella", "weight": 1, "value": 3},
    {"name": "Hiking Boots", "weight": 2, "value": 6},
    {"name": "Tent", "weight": 4, "value": 8},
    {"name": "Sleeping Bag", "weight": 2, "value": 7},
    {"name": "Camping Stove", "weight": 1.5, "value": 5},
    {"name": "Map and Compass", "weight": 0.2, "value": 4},
    {"name": "Binoculars", "weight": 1, "value": 3},
    {"name": "Insect Repellent", "weight": 0.2, "value": 2},
    {"name": "Sunscreen", "weight": 0.3, "value": 2},
    {"name": "Multi-tool", "weight": 0.3, "value": 5},
    {"name": "Rope", "weight": 1, "value": 3},
    {"name": "Water Filter", "weight": 0.5, "value": 6},
    {"name": "Fire Starter", "weight": 0.1, "value": 4},
    {"name": "Emergency Whistle", "weight": 0.1, "value": 2},
    {"name": "Hammock", "weight": 1, "value": 4},
    {"name": "Solar Charger", "weight": 0.5, "value": 5},
    {"name": "Hand Sanitizer", "weight": 0.2, "value": 1}
]

MAX_WEIGHT = 7

# Define the fitness function
def fitness_function(individual):
    total_weight = sum(items[i]["weight"] for i, gene in enumerate(individual.item) if gene)

    if total_weight > MAX_WEIGHT:
        return -1  # Penalty for exceeding weight limit
    total_value = sum(items[i]["value"] for i, gene in enumerate(individual.item) if gene)
    return total_value

# Create the gene pool
pool = ArrayPool(gene_array=np.array([0, 1]), fitness_function=fitness_function, length=len(items))

# Define the layers
layers = [
    Populate(population=100, gene_pool=pool),
    # Keep only positive mutations.
    SwapMutation(selection_function=RandomSelection(percent_to_select=0.05).select, overpowered=True),
    ParentNPoint(selection_function=RankBasedSelection(amount_to_select=2, factor=10).select, families=8, children=2,
                 n_points=5),
    SortByFitness(),
    CapPopulation(max_population=90)
]

# Create and compile the environment
env = Environment(layers=layers, verbose_every=10)
env.compile()

# Evolve the population
env.evolve(generations=1000)

# Get and print the best solution
best = env.best_ever
print("Best solution:")
total_weight = 0
total_value = 0
for i, gene in enumerate(best.item):
    if gene:
        print(f"- {items[i]['name']}")
        total_weight += items[i]['weight']
        total_value += items[i]['value']
print(f"Total weight: {total_weight}/{MAX_WEIGHT}")
print(f"Total value: {total_value}")

# Plot the fitness history
env.plot()