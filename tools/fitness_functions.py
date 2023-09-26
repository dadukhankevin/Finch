from Finch.genetics.population import NPCP as np


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


class BinaryCrossEntropyLoss:
    def __init__(self, epsilon=1e-12):
        """
        :param epsilon: A small value to avoid division by zero or log of zero
        """
        self.epsilon = epsilon

    def func(self, y_true, y_pred):
        """
        :param y_true: The true labels, either 0 or 1
        :param y_pred: The predicted probabilities, between 0 and 1
        :return: The binary cross entropy loss
        """
        # Clip the predictions to avoid log of zero or one
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        # Calculate the negative log likelihood
        nll = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return nll


class MeanSquaredErrorLoss:
    def __init__(self):
        pass

    def func(self, y_true, y_pred):
        """
        :param y_true: The true values
        :param y_pred: The predicted values
        :return: The mean squared error loss
        """
        # Calculate the squared difference
        diff = (y_true - y_pred) ** 2
        # Calculate the mean over all samples
        mse = np.mean(diff)
        return -mse
