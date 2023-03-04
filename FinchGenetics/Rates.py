import random

import matplotlib.pyplot as plt


class Rate:
    def __init__(self, start, end, epochs, return_int=False):
        """
        :param start: The start ind or float
        :param end: What you want it to return after x epochs
        :param epochs: How many epochs until it reaches this rate
        :param return_int: Do you want it to return only ints?
        """
        self.start = start
        self.end = end
        self.epochs = epochs
        self.chagerate = (end - start) / epochs
        self.ri = return_int

    def next(self):
        r = self.start
        self.start += self.chagerate
        if self.ri:
            return int(r)
        return r

    def get(self):
        if self.ri:
            return int(self.start)
        else:
            return self.start

    def graph(self):
        s = self.start
        history = []
        for i in range(self.epochs):
            history.append(s)
            s += self.chagerate
        print("min", min(history))
        print("max", max(history))
        plt.plot(history)
        plt.show()

def make_switcher(x):
    x = make_callable(x)
    n = x()
    def r():
        return random.choice([-n, n])
    return r
def make_callable(x):
    if not callable(x):
        def constant():
            return x
        return constant
    else:
        return x
