import numpy as np
import imageio


def has_duplicates(lst):
    s = set(map(tuple, lst))

    # Check if there are duplicates
    hd = len(s) != len(lst)
    return hd


class ValueWeightFunction:
    def __init__(self, maxweight, force_unique_items=False):
        self.maxweight = maxweight
        self.force_unique_items = force_unique_items

    def func(self, individual):

        if self.force_unique_items:
            if has_duplicates(individual):
                return 0

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

        try:
            result = self.equation.evaluate(individual)
        except ZeroDivisionError or OverflowError:
            return 0
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

    def fitness(self, solution):
        f = sum(np.abs(self.target_im - solution).flatten())
        return f

