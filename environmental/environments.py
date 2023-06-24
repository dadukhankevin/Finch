import numpy as np


class Sequential:
    def __init__(self, layers, individuals=None):
        if individuals is None:
            individuals = []
        self.individuals = np.asarray(individuals)
        self.layers = layers
        self.stop = False

    def evolve(self, generations: int, callback=None, verbose_every=False):
        history = []
        fitness = 0
        for i in range(generations):
            if verbose_every and i%verbose_every == 1:
                print(f"Generation {i + 1}/{generations}. Max fitness: {fitness}. Population: {len(self.individuals)}")
            for layer in self.layers:
                self.individuals = layer.run(self.individuals)
            if callback:
                callback(self.individuals, self)
            fitness = self.individuals[-1].fitness
            history.append(fitness)
            if self.stop:
                return self.individuals, history
        return self.individuals, history

    def stop(self):
        self.stop = True
