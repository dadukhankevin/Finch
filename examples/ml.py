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
    klayers.Dense(9, input_dim=2, activation='relu'),  # Input layer with 2 features
    klayers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
evo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
evo.summary()

ws = neuro_pools.get_model_weights_as_array(evo)
print(len(ws))
evo = neuro_pools.set_model_weights_from_array(evo, ws)[0]
print(evo)
input()
def fitness_function(m):
    # Generate two random numbers between 0 and 1
    feature1 = np.random.random()
    feature2 = np.random.random()

    # Create input data with the generated numbers
    input_data = np.array([[feature1, feature2]])

    # Make a prediction using the model
    prediction = m.predict(input_data)

    # Calculate Mean Absolute Error (MAE) between prediction and ground truth label
    # We assume the correct label based on the same rule used earlier.
    correct_label = int(feature1 + feature2 > 1)
    mae = abs(correct_label - prediction[0][0])

    return 1 - mae
gene_pool = neuro_pools.KerasPool(evo, fitness_function)
environment = environments.Sequential(layers=[
    layers.Populate(gene_pool, 5),
    mutation_layers.FloatMutateAmount(3, amount_genes=3, gene_pool=gene_pool),
    layers.ParentNPointCrossover(2, 2, n=2),
    layers.SortByFitness(),
    layers.CapPopulation(10)
])

environment.evolve(10)
plt.plot(environment.history)
plt.show()