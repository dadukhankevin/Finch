import numpy as np


class Sequential:
    def __init__(self, layers, individuals=None):
        if individuals is None:
            individuals = []
        self.individuals = individuals
        self.layers = layers
        self.stop = False

    def evolve(self, generations: int, callback, verbose=True):
        history = []
        fitness = 0
        for i in range(generations):
            if verbose:
                print(f"Generation {i + 1}/{generations}. Max fitness: {fitness}")
            for layer in self.layers:
                self.individuals = layer.run(self.individuals)
            callback(self.individuals, self)
            fitness = self.individuals[-1].fitness
            history.append(fitness)
            if self.stop:
                return self.individuals, history
        return self.individuals, history

    def stop(self):
        self.stop = True
