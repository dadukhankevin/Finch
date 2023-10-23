import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as klayers
import numpy as np
from Finch.environmental import environments
from Finch.environmental.layers import standard_layers as layers
from Finch.environmental.layers import mutation_layers
from Finch.functions import selection
from Finch.ml import neuro_pools
import matplotlib.pyplot as plt

# Generate some synthetic data for binary classification
data = np.random.random((1000, 2))  # 1000 samples with 2 features
labels = (data[:, 0] + data[:, 1] > 1).astype(int)  # Binary labels based on a simple rule

# Define the model
evo = keras.Sequential([
    klayers.Dense(8, input_dim=2, activation='relu'),  # Input layer with 2 features
    klayers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
evo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
evo.summary()

def fitness_function(model):
    # Generate two random numbers between 0 and 1
    feature1 = np.random.random()
    feature2 = np.random.random()

    # Create input data with the generated numbers
    input_data = np.array([[feature1, feature2]])

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Calculate Mean Absolute Error (MAE) between prediction and ground truth label
    # We assume the correct label based on the same rule used earlier.
    correct_label = int(feature1 + feature2 > 1)
    mae = abs(correct_label - prediction[0][0])

    return 1 - mae

gene_pool = neuro_pools.KerasPool(evo, fitness_function)
environment = environments.Sequential(layers=[
    layers.Populate(gene_pool, 5),
    mutation_layers.FloatMutateAmount(amount_individuals = 3,amount_genes=4, gene_pool = gene_pool),
    layers.ParentNPointCrossover(2, 2, n = 2),
    layers.SortByFitness(),
    layers.CapPopulation(20)
])

environment.evolve(10)
evo = neuro_pools.set_model_weights_from_array(evo, environment.individuals[-1].genes)
