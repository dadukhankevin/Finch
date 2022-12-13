import numpy as np

class Individual:
    def __init__(self, ar, fitness_func):
        self.genes = ar
        self.fitness_func = fitness_func
        self.fitness = 0
        self.age = 0
    def fit(self, factor=1):
        """
        :param factor: values closer to 0 favor the earlier fitness while values closer to 1 favors the new fitness
        """
        self.fitness = ((1 - factor) * self.fitness) + (
                factor * self.fitness_func(self.genes))  # to prevent division by zero
        return self.fitness

