import logging

import matplotlib.pyplot as plt
import math


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


class Rates:
    def __init__(self, rate, arg):  # , epochs, max_population):
        self.rate = rate
        self.arg = arg
        self.first = rate
        logging.warning("The Rates class is deprecated, please use the Rate class instead. You can also define your "
                        "own rate function.")

    def slow(self):
        """Decreases rate by percent"""
        self.rate = self.rate * (1 - self.arg)
        return round(self.rate, 5)

    def speed(self):
        self.rate = self.rate * (1 + self.arg)
        return round(self.rate, 5)

    def multiply(self):
        self.rate = self.rate * self.arg
        return self.rate

    def constant(self):
        return self.rate

    def divide(self):
        self.rate = self.rate / self.arg
        return self.rate

    def exponential(self):
        self.rate = self.rate ** self.arg
        return self.rate

    def add(self):
        self.rate += self.arg
        return self.rate

    def sigmoid(self):  # TODO: make this actually work haha
        exp = math.pow(math.e, self.rate)
        self.rate = 1 / (1 + exp)  # the sigmoid function?
        return self.rate


def make_constant_rate(n):
    if not callable(n):
        return Rates(n, 0).constant
    else:
        return n


def graph(func, itrations):
    hist = []
    for i in range(itrations):
        hist.append(func())
    plt.plot(hist)
    plt.show()
