from Finch.FinchGA.generic import img2genes
import numpy as np
import imageio


class ValueWeightFunction:
    def __init__(self, maxweight):
        self.maxweight = maxweight

    def func(self, individual):

        weight = 0
        value = 0
        for i in individual:
            weight += float(i[2])
            value += float(i[1])
        if weight > self.maxweight:
            return 0
        else:
            return value


class EquationFitness:
    def __init__(self, desired_result, equation):
        """
        :param desired_result: What you want it to equal
        :param equation: The Equation object
        """
        self.desired = desired_result
        self.equation = equation

    def func(self, individual):
        result = self.equation.evaluate(individual)
        if result > self.desired:
            try:
                return self.desired / result
            except ZeroDivisionError or OverflowError:
                return 0
        else:
            try:
                return result / self.desired
            except ZeroDivisionError or OverflowError:
                return 0


class ImageSimilarity:
    def __init__(self, path):
        self.path = path
        target_im = imageio.imread(path)
        self.target_im = np.asarray(target_im, dtype=np.float)
        self.target_genes = img2genes(self.target_im) / 255
        self.max_fitness = self.fitness_raw(self.target_genes)

    def fitness_raw(self, solution):
        fitness = np.sum(np.abs(self.target_genes - solution))
        fitness = np.sum(self.target_genes) - fitness
        return fitness

    def fitness_fun(self, solution):
        fitness = np.sum(np.abs(self.target_genes - solution))
        fitness = np.sum(self.target_genes) - fitness
        return fitness / self.max_fitness
