import random
import matplotlib.pyplot as plt


class Rate:
    def __init__(self, start, end, epochs, return_int=False):
        """
        :param start: The starting value (int or float)
        :param end: The target value after epochs
        :param epochs: The number of epochs to reach the target value
        :param return_int: Whether to return only integers
        """
        self.value = start  # use a more descriptive name than start
        self.end = end
        self.epochs = epochs
        self.change_rate = (end - start) / epochs  # use underscores for variable names
        self.return_int = return_int  # use the same name as the parameter

    def next(self):
        # return the current value and update it by the change rate
        result = self.value
        self.value += self.change_rate
        if self.return_int:
            return int(result)
        return result

    def get(self):
        # return the current value without updating it
        if self.return_int:
            return int(self.value)
        else:
            return self.value

    def graph(self):
        # plot the history of the value over epochs
        history = []
        temp = self.value  # store the current value temporarily
        for i in range(self.epochs):
            history.append(temp)
            temp += self.change_rate
        print("min", min(history))
        print("max", max(history))
        plt.plot(history)
        plt.show()
        self.value = temp  # restore the current value


def make_switcher(x):
    x = make_callable(x)
    n = x()

    def r():
        # return a random choice of -n or n
        return random.choice([-n, n])

    return r


def make_callable(x):
    if not callable(x):
        # return a function that always returns x
        def constant():
            return x

        return constant
    else:
        # return x as it is
        return x
