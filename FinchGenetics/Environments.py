import numpy as np


class Environment:
    def __init__(self, layers):
        """
        :param layers:
        """
        self.layers = layers
        self.iterations = 0

    def evolve(self, generations, data=[], verbose=1, callback=None):
        """
        :param generations: The number of generations/epochs
        :param data: If any pre-existing data
        :param verbose: Print every n
        :return:
        """
        history = []
        for i in range(generations):
            self.iterations += 1
            for layer in self.layers:
                data = layer.run(data)
            if self.iterations % verbose == 0:
                print(f"{self.iterations}: {data[-1].fitness}")
            if callback:
                callback(data)
            history.append(data[-1].fitness)

        return data, history


class Adversarial:  # TODO: test this
    def __init__(self, environments):
        """
        Competes environments against each other
        :param environments:
        """
        self.environments = environments
        self.data = None

    def compete(self, epochs):  # TODO: compute which environment performs best!
        """
        :param epochs: Number of competitions
        """
        best = 0
        fullhist = []
        new_data = self.data
        for i in range(epochs):
            self.data = new_data
            for env in self.environments:  # simulate each environment
                data, history = env.evolve(1, self.data)
                fullhist.append(history)

                if history[-1] > best:
                    best = history[-1]
                    new_data = data  # Sets the best new data
        return self.data, fullhist
