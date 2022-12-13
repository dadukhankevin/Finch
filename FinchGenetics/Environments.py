import numpy as np


class Environment:
    def __init__(self, layers):
        """
        :param layers:
        """
        self.layers = layers
        self.iterations = 0

    def evolve(self, generations, data=[], verbose=1):
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
                print(f"{self.iterations}: {data[-1].fitness}, {data[-1].genes}")
            history.append(data[-1].fitness)

        return data, history