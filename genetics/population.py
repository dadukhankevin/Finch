import numpy as np

# Try importing CuPy
try:
    import cupy as cp

    # Check if GPU is available
    if cp.cuda.get_device_id() == 0:
        print("GPU detected. Using CuPy.")


        array_module = cp
    else:
        print("GPU not detected. Using NumPy.")
        array_module = np

except ImportError:
    print("CuPy not found. Using NumPy.")
    array_module = np
np = array_module


class Individual:
    def __init__(self, genes, fitness_function):
        self.genes = genes  # a numpy array
        self.fitness = 0
        self.fitness_function = fitness_function

    def fit(self):
        self.fitness = self.fitness_function(self.genes)

    def copy(self): # this wont work with mutable fitness functions!!!
        copied_genes = np.copy(self.genes)
        copied_fitness_function = self.fitness_function
        copied_individual = Individual(copied_genes, copied_fitness_function)
        copied_individual.fitness = self.fitness
        return copied_individual
