class ValueWeightFunction:
    def __init__(self, max_weight):
        self.max_weight = max_weight

    def func(self, individual):

        weight = 0
        value = 0
        for i in individual:
            weight += float(i[2])
            value += float(i[1])
        if weight > self.max_weight:
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
