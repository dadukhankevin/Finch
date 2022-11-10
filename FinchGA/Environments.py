import math
import Finch.FinchGA.generic as generic
import logging
import sys
import matplotlib.pyplot as plt
handler = logging.StreamHandler(sys.stdout)


class SequentialEnvironment:
    def __init__(self, layers=[]):
        """
        :param every: Defines the environment
        :param layers: The layers of the environment
        """
        self.layers = layers
        # To be defined in compile
        self.data = None
        self.epochs = None
        self.fitness = None
        self.every = None
        self.stop = None
        self.keep_going = None

    def compile(self, epochs, fitness, every=1, data=None, stop_threshold=math.inf, keep_going=False):
        """
        Prepares environment to be simulated
        :param epochs: Number of loops
        :param fitness: The fitness function (you specify)
        :param every: Print info every n epochs
        :param data: If you want to use data from another environment or saved file
        :param stop_threshold: Stop early if fitness achieves a certain level
        :param keep_going: If true will ignore epochs and go until stop_threshold is met
        :return:
        """
        if keep_going:
            logging.warning("You have set keep_going to 'true'. This can result in your environment running "
                            "indefinitely")
        self.epochs = epochs
        self.fitness = fitness
        self.every = every
        self.data = data
        self.history = []
        self.stop = stop_threshold
        self.best = 0
        self.best_ind = []
        self.keep_going = keep_going
        if self.data is None:
            self.data = generic.Generation([])  # ensures data is of correct type

    def simulate_env(self, data=None):
        """
        :param data: If you want to use data from previous environment
        """

        last = 0
        if data is not None:
            self.data = data
        history = []
        for i in range(self.epochs):
            for d in self.layers: # call the run function of each layer
                d.run(self.data, self.fitness)
            ind = self.data.individuals[-1]
            self.best_ind = ind
            self.best = self.data.individuals[-1].fitness  # when you run out of variable names
            history.append(ind.fitness)
            if self.best > last:
                last = self.best  # the most fit
            if self.best >= self.stop:
                print("\033[92m Stopping: ", self.best, ind.genes)
                self.history = history
                return self.data, history
            if i % self.every == 0:
                print("\033[92m", self.best, ind.genes)
            if self.keep_going:
                self.epochs += 1  # So that it continues until self.stop threshold is met.
        self.history = history
        return self.data, history
    def plot(self):
        plt.plot(self.history)
        plt.show()
    def display_history(self):
        return "environment took {} epochs".format(len(self.history))

class Adversarial: # TODO: re name this
    def __init__(self, environments):
        """
        Competes environments against each other
        :param environments:
        """
        self.environments = environments
        self.data = None

    def compete(self, epochs): # TODO: compute which environment performs best!
        """
        :param epochs: Number of competitions
        """
        best = 0
        fullhist = []
        new_data = self.data
        for i in range(epochs):
            self.data = new_data
            for env in self.environments: # simulate each environment
                data, history = env.simulate_env(self.data)
                fullhist.append(history)

                if history[-1] > best:
                    best = history[-1]
                    new_data = data # Sets the best new data
        return self.data, fullhist
