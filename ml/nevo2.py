import Finch.genetics.population as pop
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers as klayers
from Finch.environmental import environments
from Finch.environmental.layers import standard_layers as layers
from Finch.environmental.layers import mutation_layers
from Finch.functions import selection
from Finch.ml import neuro_pools
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(len(train_images))
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model
evo = keras.Sequential([
    klayers.Flatten(input_shape=(28, 28)),  # Input layer for MNIST images (28x28 pixels)
    klayers.Dense(8, activation='relu', use_bias=False),
    klayers.Dense(10, activation='softmax', use_bias=False)  # Output layer for 10-digit classification
])

# Compile the model
evo.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy for multi-class classification
            metrics=['accuracy'])
evo.summary()

# Define the fitness function
def fitness_function(model):
    num_images_to_predict = 40  # Number of images to predict
    indices = np.random.choice(len(train_images), num_images_to_predict, replace=False)  # Choose random indices
    selected_images = train_images[indices]
    selected_labels = train_labels[indices]

    _, accuracy = model.evaluate(selected_images, selected_labels, verbose=0)
    return accuracy

# The rest of your code remains the same
s = selection.RankBasedSelection(2, amount_to_select=2)
rank = selection.RankBasedSelection(1, 1)
num_images_to_predict = 600  # Number of images to predict
indices = np.random.choice(len(train_images), num_images_to_predict, replace=False)  # Choose random indices
selected_images = train_images[indices]
selected_labels = train_labels[indices]

gene_pool = neuro_pools.KerasPool(evo, fitness_function)
environment = environments.Sequential(layers=[
    layers.Populate(gene_pool, 10),
    #mutation_layers.FloatOverPoweredMutation(4500, gene_pool, tries=10, selection_function=s),
    layers.KerasTrain(selected_images, selected_labels, batch_size=1, gene_pool=gene_pool, epochs=1, selection_function=rank),
    layers.ParentMeanCrossover(num_families=2, num_children=4, selection_function=s),
    layers.SortByFitness(),
    layers.CapPopulation(9)
])

environment.evolve(50)
evo = neuro_pools.set_model_weights_from_array(evo, environment.individuals[-1].genes)[0]
