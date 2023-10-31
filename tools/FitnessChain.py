class Chain:
    def __init__(self, funcs, thresholds=[]):
        """
        :param funcs: The fitness functions
        :param thresholds: The threshold of fitness before you want to move on to the next function
        """
        self.fitness_functions = iter(funcs)
        self.thresholds = iter(thresholds)
        self.funct = next(self.fitness_functions)
        self.thresh = next(self.thresholds)

    def func(self, individual):
        points, individual = self.funct(individual)
        if points >= self.thresh:  # gets then next function
            try:
                self.funct = next(self.fitness_functions)
                self.thresh = next(self.thresholds)
            except StopIteration:
                pass
        return points, individual
