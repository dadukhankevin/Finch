import random
import random as r
import string

import numpy as np
from difflib import SequenceMatcher
import time
from matplotlib import pyplot as plt
from textblob import TextBlob

allowed = list(range(0,99))

class String:
    def __init__(self, arrlen, length, fittnessfunc, genfunc=None, letters=allowed):
        self.name = "String"
        self.arrlen = arrlen
        self.length = length
        self.letters = letters
        if genfunc != None:
            self.genfunc = genfunc
        else:
            self.genfunc = self.gen

    def gen(self, length, num):
        # choose from all lowercase letter
        ret = []
        for i in range(num):
            result_str = [r.choice(self.letters) for i in range(length)]
            ret.append([result_str])
        return ret

    def run(self, data, func):
        if len(data) - 1 < self.arrlen:
            data = data + self.genfunc(self.length, self.arrlen - (len(data) - 1))
        return data


class StringMutate:
    def __init__(self, mutation_function="change", percentage=5, letters=allowed, small_percent=30):
        self.name = "StringMutate"
        self.mutation_function = mutation_function
        self.percentage = percentage
        self.letters = letters
        self.small_percent = small_percent

    def mix(self, data):
        ret = []
        for i in data:
            if r.randint(1, 100) <= self.percentage:
                l = i
                r.shuffle(l)
                ret.append(l)
            else:
                ret.append(i)
        return ret

    def mutate_one(self, element):  # TODO: make this lots better
        for i in range(len(element)):
            if r.randint(1, 100) <= self.small_percent:
                ind = r.randint(0, len(element) - 1)
                thing = random.choice(self.letters)
                element.pop(ind)
                element.insert(ind, thing)
        return [element]

    def change(self, data):
        return [self.mutate_one(i[0]) if r.randint(1, 100) <= self.percentage else i for i in data]

    def run(self, data, func):
        if self.mutation_function == "mix":
            return self.mix(data)
        if self.mutation_function == "change":
            return self.change(data)


class Parent:
    def __init__(self, num, typ="", gene_size=3, family_size=2):
        self.name = "Parent"
        self.num = num
        self.gene_size = gene_size
        self.fs = family_size

    def parent(self, data, mom, dad):

        l = len(mom)

        mom = np.array_split(mom[0], self.gene_size)
        dad = np.array_split(dad[0], self.gene_size)
        children = []
        dad = dad + mom
        both = dad
        for i in range(self.fs):
            r.shuffle(both)
            children.append([both[0: l]])
            data.pop(0)
        data = data + children + mom + dad

        return [i[0] for i in data]

    def run(self, data, func):
        return [self.parent(data, data[-1], data[-1])]
def keyy(data):
    return data[0]

class Environment:
    def __init__(self, *args):
        self.classes = list(args)

    def compile(self, epochs, func):
        data = []
        history = []
        for n in range(epochs):
            for i in self.classes:
                data = i.run(data=data, func=func)
                data = [func(i) for i in data]
                data = sorted(data)
                try:
                    top = data[-1]
                    history.append(data[-1][0])
                except:
                    top = []
                data = [i[1] for i in data]
                print(int((n / epochs) * 100), top)


        return data, history

    def add(self, layer):
        self.classes.append(layer)


class Kill:
    def __init__(self, percent):
        self.name = "kill"
        self.percent = percent

    def run(self, data, func):
        return data[int(len(data) * self.percent):-1]


class RemoveTwins:
    def __init__(self):
        pass

    def run(self, data, func):
        return np.unique(data).tolist()


class NarrowPool:
    def __init__(self):
        pass

    def run(self):
        pass


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def ifin(ls, str1):
    for i in ls:
        if i in str1:
            return True
        else:
            return False


def fit(data):
    points = 0
    if data != []:
        for i in data[0]:
            if i % 2 == 0:
                points += 1

    return (points, data)


environment = Environment()
environment.add(
    Kill(.2)
)
environment.add(
    String(100, 10, fit)
)
environment.add(
    StringMutate(percentage=50, small_percent=5)
)

environment.add(
    Parent(5, gene_size=1)
)
data, history = environment.compile(1000, func=fit)
plt.plot(history)
plt.show()
