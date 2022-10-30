class Fittness:
    def __init__(self, funcs, thresh=[]):
        self.funcs = iter(funcs)
        self.thresholds = iter(thresh)
        self.funct = next(self.funcs)
        self.thresh = next(self.thresholds)

    def func(self, individual):
        points, individual = self.funct(individual)
        if points >= self.thresh:
            try:
                self.funct = next(self.funcs)
                self.thresh = next(self.thresholds)
            except StopIteration:
                pass
        return points, individual
class Equation:
    def __init__(self, equation, *args):
        self.equation = equation

class Genes:
    def __init__(self, data, weights):