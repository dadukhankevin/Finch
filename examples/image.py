from Finch.environmental import environments
from Finch.environmental import layers
from Finch.genetics import genepools
from Finch.functions import selection
#import matplotlib.pyplot as plt
# 1 min per 50
import numpy as np
from PIL import Image

# Load the reference image
reference_image = Image.open("C:/Users/danie/PycharmProjects/finch2/Finch/resources/img.png")  # Replace with the actual path to your reference image
image_shape = np.asarray(reference_image).shape
print(image_shape)
# Convert the reference image to a one-dimensional array
reference_array = np.array(reference_image).flatten()/255
gene_len = reference_array.shape
print(gene_len)
def fitness_function(image):
    similarity_score = np.sum(np.abs(reference_array - image))
    fitness = 1 / (1 + similarity_score)
    return fitness
n = 0
# def callback(individuals, pool):
#   global n
#   if n % 20 == 0:
#     image = individuals[0].genes
#     image = image.reshape(image_shape)
#     plt.imshow(image)
#     plt.axis('off')  # Hide axes
#     plt.show()
#     image = image.flatten()
#   n += 1
#   return individuals
gene_pool = genepools.FloatPool(0, 1, length=gene_len, fitness_function=fitness_function)
input()
environment = environments.Sequential(layers=[
    layers.Populate(gene_pool=gene_pool, population=4),
    layers.FloatOverPoweredMutation(amount_individuals=2, amount_genes=1000, tries=30, gene_pool=gene_pool, max_negative_mutation=-.1, max_positive_mutation=.1),
    layers.Controller(layers.Parent(num_children=3, num_families=4, selection_function=selection.rank_based_selection), execute_every=5),
    layers.FloatMomentumMutation(divider=3, amount_individuals=3, amount_genes=3000, execute_every=5, selection_arg=10, reset_baseline=1),
    layers.SortByFitness(),
    layers.Kill(percent=.1),
])

individuals, history = environment.evolve(500, verbose_every=1, track_float_diff_every=1)
print("The evolved result:")
print(individuals[0].genes)
