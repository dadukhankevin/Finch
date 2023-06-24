class Fittness:
    def __init__(self, funcs, thresh=[]):
        """
        :param funcs: The fitness functions
        :param thresh: The threshold of fitness before you want to move on to the next function
        """
        self.funcs = iter(funcs)
        self.thresholds = iter(thresh)
        self.funct = next(self.funcs)
        self.thresh = next(self.thresholds)

    def func(self, individual):
        points, individual = self.funct(individual)
        if points >= self.thresh:  # gets then next function
            try:
                self.funct = next(self.funcs)
                self.thresh = next(self.thresholds)
            except StopIteration:
                pass
        return points, individual
