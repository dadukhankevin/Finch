from Finch.tools import individualselectors as selectors
from Finch.environments import Sequential
from Finch.genepools import ArrayPool
from Finch.layers import *
import numpy as np

# Config section
max_weight = 5
max_items = 5
population_size = 50
mutate_amount = 4
gene_selection = 1
parent_count = 8
selection_rank_factor = 100
children_count = 4
max_population = 100
evolution_steps = 2000
penalty_mult = 40

selector = selectors.RankBasedSelection(selection_rank_factor, amount_to_select=parent_count)


class Item:
    def __init__(self, name, weight, value):
        self.name = name
        self.weight = weight
        self.value = value


items = np.array([
    Item('Water Bottle', 2, 3),
    Item('Trail Mix', 0.5, 2),
    Item('Flashlight', 1.5, 5),
    Item('Map', 0.5, 8),
    Item('Multi-tool', 0.8, 4),
    Item('First Aid Kit', 1.2, 6),
    Item('Rain Jacket', 1.5, 4),
    Item('Compass', 0.3, 7),
    Item('Tent', 4, 9),
    Item('Sleeping Bag', 3, 8),
    Item('Cooking Stove', 1.8, 6),
    Item('Dried Food Pack', 1.2, 5),
    Item('Headlamp', 0.6, 5),
    Item('Hiking Boots', 2.5, 7),
    Item('Camera', 1, 6),
    Item('Notebook', 0.4, 3),
    Item('Sunscreen', 0.3, 4),
    Item('Insect Repellent', 0.4, 4),
    Item('Sunglasses', 0.2, 3),
])


def print_backpack(backpack):
    value = backpack.fitness
    backpack = backpack.genes
    print("Backpack Info:")
    print("===================")
    print(f"Total weight: {sum(item.weight for item in backpack)}")
    print(f"Total value: {value}")
    print("Backpack Contents:")
    print("===================")

    for item in backpack:
        print(f"{item.name}: Weight {item.weight}, Value {item.value}")
    print("===================")


def fitness(individual):
    backpack = individual.genes
    total_weight = 0
    total_value = 0
    item_names = set()

    for item in backpack:
        total_weight += item.weight
        total_value += item.value
        item_names.add(item.name)

    duplicate_penalty = (len(backpack) - len(item_names)) * penalty_mult
    if total_weight > max_weight:
        return 0
    score = total_value - duplicate_penalty

    return score


gene_pool = ArrayPool(gene_array=items, length=max_items, unique=True)

environment = Sequential(layers=[
    Populate(gene_pool, population=population_size),
    Mutate(individual_selection=mutate_amount, gene_selection=gene_selection),
    InsertionDeletionMutation(individual_selection=mutate_amount, gene_selection=gene_selection),
    ParentSimple(selector.select, children=children_count, refit=True),
    SortByFitness(),
    CapPopulation(max_population=max_population),
])

if __name__ == "__main__":
    environment.compile(fitness_function=fitness, verbose_every=100)
    environment.evolve(evolution_steps)

    print(environment.best_ever.fitness)

    print_backpack(environment.best_ever)
    environment.plot()
