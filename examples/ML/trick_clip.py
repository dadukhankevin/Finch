from Finch.fitness.ml import image
from Finch.fitness.fitness_tools import MixFitness
from Finch.environments import Sequential
from Finch.genepools import BinaryPool, FloatPool
from Finch.layers import BinaryMutate, ParentNPoint, Populate, CapPopulation, SortByFitness



size = (320, 240)
length = size[0] * size[1] * 3
population = 100
new_individuals = 1

parents = 3
children = 2
parent_points = 4

individual_selection = 5
gene_selection = 10

generations = 1000

shape = (size[1], size[0], 3)

target = ["dog"]
other_labels = ["random noise", "black", "other"]
clip1 = image.ZeroShotImage(target_labels=target, other_labels=other_labels, shape=shape, denormalize=1)

clip2 = image.ZeroShotImage(target_labels=target, other_labels=other_labels,
                            model='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', shape=shape, denormalize=1)

clip3 = image.ZeroShotImage(target_labels=target, other_labels=other_labels,
                            model='patrickjohncyh/fashion-clip', shape=shape, denormalize=1)


fitness_function = MixFitness([
    clip1.enhance_fit,
    clip2.enhance_fit,
    clip3.enhance_fit,
], weights=[.35, .3, .35])

pool = BinaryPool(length=length)


environment = Sequential([
    Populate(pool, population+new_individuals),
    ParentNPoint(families=parents, points=parent_points, children=children),
    BinaryMutate(individual_selection=individual_selection, gene_selection=gene_selection),
    SortByFitness(),
    CapPopulation(max_population=population)
])

environment.compile(fitness_function=fitness_function)
environment.evolve(generations=generations)
environment.plot()


