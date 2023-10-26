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
s = selection.RankBasedSelection(5)
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
    feature1 = np.random.random(5)
    feature2 = np.random.random(5)
    input_data = np.column_stack((feature1, feature2))

    predictions = model.predict(input_data)
    correct_labels = (feature1 + feature2 > 1).astype(int)
    mae = np.abs(correct_labels - predictions[:, 0])

    scores = 1 - mae
    min_score = min(scores)

    return min_score
gene_pool = neuro_pools.KerasPool(evo, fitness_function)
environment = environments.Sequential(layers=[
    layers.Populate(gene_pool, 20),
    mutation_layers.FloatOverPoweredMutation(10, 3, gene_pool, tries=3),
    layers.ParentNPointCrossover(2, 2, n = 2, selection_function=s),
    layers.SortByFitness(),
    layers.CapPopulation(20)
])
environment
environment.evolve(20)
evo = neuro_pools.set_model_weights_from_array(evo, environment.individuals[-1].genes)[0]
for i in range(10):
    try:
        feature1 = float(input("Enter the first feature: "))
        feature2 = float(input("Enter the second feature: "))
        input_data = np.array([[feature1, feature2]])
        prediction = evo.predict(input_data)
        print(f"Model prediction: {prediction[0][0]:.4f}")
    except ValueError:
        print("Please enter valid numerical values.")