import numpy as np

class Individual:
    def __init__(self, pool, ar, fitness_func):
        self.pool = pool
        self.genes = ar
        self.fitness_func = fitness_func
        self.fitness = 0
        self.age = 0
    def raw_fit(self):
        self.genes = self.genes.reshape(self.pool.shape)
        f = self.fitness_func(self.genes)
        self.genes = self.genes.flatten()
        return f
    def fit(self, factor=1):
        """
        :param factor: values closer to 0 favor the earlier fitness while values closer to 1 favors the new fitness
        """
        self.genes = self.genes.reshape(self.pool.shape)

        self.fitness = ((1 - factor) * self.fitness) + (
                factor * self.fitness_func(self.genes))  # to prevent division by zero
        self.genes = self.genes.flatten()
        return self.fitness


