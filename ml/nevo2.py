import Finch.genetics.population as pop
import numpy as np
pop.NPCP = np
from tensorflow import keras
from tensorflow.keras import layers as klayers
import numpy as np
from Finch.environmental import environments
from Finch.environmental.layers import standard_layers as layers
from Finch.environmental.layers import mutation_layers
from Finch.functions import selection
from Finch.ml import neuro_pools
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model
evo = keras.Sequential([
    klayers.Flatten(input_shape=(28, 28)),  # Input layer for MNIST images (28x28 pixels)
    klayers.Dense(128, activation='relu'),
    klayers.Dropout(0.2),  # Dropout layer to reduce overfitting
    klayers.Dense(10, activation='softmax')  # Output layer for 10-digit classification
])

# Compile the model
evo.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy for multi-class classification
            metrics=['accuracy'])
evo.summary()

# Define the fitness function
def fitness_function(model):
    num_images_to_predict = 4  # Number of images to predict
    indices = np.random.choice(len(train_images), num_images_to_predict, replace=False)  # Choose random indices
    selected_images = train_images[indices]
    selected_labels = train_labels[indices]

    _, accuracy = model.evaluate(selected_images, selected_labels, verbose=0)
    return accuracy

# The rest of your code remains the same
s = selection.RankBasedSelection(5).select
gene_pool = neuro_pools.KerasPool(evo, fitness_function)
environment = environments.Sequential(layers=[
    layers.Populate(gene_pool, 4),
    mutation_layers.FloatOverPoweredMutation(10, 100, gene_pool, tries=5, selection_function=s),
    layers.ParentNPointCrossover(num_families=2, num_children=2, n = 10, selection_function=s),
    layers.SortByFitness(),
    layers.CapPopulation(5)
])

environment.evolve(5)
evo = neuro_pools.set_model_weights_from_array(evo, environment.individuals[-1].genes)[0]
