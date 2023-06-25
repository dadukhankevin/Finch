import numpy as np

# Try importing CuPy

from Finch.exceptions.environment_exceptions import NoIndividualsAtEndOfRun


class Sequential:
    def __init__(self, layers, individuals=None):
        if individuals is None:
            individuals = []
        self.individuals = np.asarray(individuals)
        self.layers = layers
        self.stop = False
        self.original = None
        self.diff = None
        self.iteration = 0

    def evolve(self, generations: int, callback=None, verbose_every=False, track_float_diff_every=False):
        history = []
        fitness = 0
        for i in range(generations):
            self.iteration = i
            if verbose_every and i % verbose_every == 0:
                print(f"Generation {i + 1}/{generations}. Max fitness: {fitness}. Population: {len(self.individuals)}")
            for layer in self.layers:
                self.individuals = layer.run(self.individuals, self)
            if callback:
                callback(self.individuals, self)
            if self.individuals.size == 0:
                raise NoIndividualsAtEndOfRun("Your environment has a population of 0 after running.")
            if self.original is None:
                self.original = self.individuals[0]
            if track_float_diff_every and i % track_float_diff_every == 0:
                self.diff = -(self.original.genes - self.individuals[0].genes)
            fitness = self.individuals[-1].fitness
            history.append(fitness)
            if self.stop:
                return self.individuals, history
        return self.individuals, history

    def stop(self):
        self.stop = True
