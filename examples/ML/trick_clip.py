from Finch.fitness.ml import image
from Finch.fitness.fitness_tools import MixFitness
from Finch.environments import Sequential
from Finch.genepools import BinaryPool, FloatPool
from Finch.layers import Mutate, ParentNPoint, Populate, CapPopulation, SortByFitness, BinaryMutate, BatchFitness, \
    ParentSimple, FloatMutateRange
from Finch.tools.individualselectors import RankBasedSelection
import requests
from PIL import Image
from io import BytesIO

def resize_image(url, new_size=(64, 64)):
    # Download the image from the URL
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    # Resize the image
    resized_image = image.resize(new_size)

    return resized_image


base_image = resize_image("https://imagez.tmz.com/image/39/1by1/2023/07/15/39558dccad4e47ae90482f7c6a7c34d5_xl.jpg")
base_image

size = (300, 300)
length = size[0] * size[1] * 3
shape = (size[1], size[0], 3)
print(length)

batch_size = 100  # How many concerrent images to test on the GPU at a time
target = ["dog"]
other_labels = ["random noise", "white", "black"]  # Our goal is for 'dog' to outscore any of these true labels
clip1 = image.ZeroShotImage(target_labels=target, other_labels=other_labels,
                            shape=shape, denormalize=True, batch_size=batch_size)  # Default Clip model

# Optionally you can define a different model!
# clip2 = image.ZeroShotImage(target_labels=target, other_labels=other_labels,
#                             model='patrickjohncyh/fashion-clip', shape=shape, denormalize=True, batch_size=batch_size)

fit = clip1.batch_enhance

# CONFIG
max_population = 40
start_pop = 2

parents = 6
factor = 10
parents = RankBasedSelection(factor=factor, amount_to_select=parents).select

children = 2
parent_points = 1

individual_selection = 20
gene_selection = 10

generations = 600

# ENVIRONMENT

if __name__ == "__main__":
    pool = BinaryPool(length=length, device='cpu', default=1)

    environment = Sequential([
        Populate(pool, start_pop),
        BinaryMutate(individual_selection=individual_selection,
                     gene_selection=gene_selection, refit=True),
        ParentNPoint(families=parents, points=parent_points, children=children,
                     refit=True),
        BatchFitness(batch_fitness_function=fit),
        # Calls the fitness function on every individual that has been modified
        SortByFitness(),
        CapPopulation(max_population=max_population)
    ])
    # evolve
    environment.compile(fitness_function='batch')
    _ = environment.evolve(generations=generations)
    # show the image
    clip1.show(environment.best_ever)
    # plot our progress
    environment.plot()
