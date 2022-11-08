import matplotlib.pyplot as plt
import math


class Rates:
    def __init__(self, rate, arg):  # , epochs, max_population):
        self.rate = rate
        self.arg = arg
        self.first = rate  # the first rate

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

    def sigmoid(self): #TODO: make this actually work haha
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