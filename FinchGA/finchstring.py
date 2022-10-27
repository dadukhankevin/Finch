import random
import random as r
import string

import numpy as np
from difflib import SequenceMatcher
import time
from matplotlib import pyplot as plt
from textblob import TextBlob

allowed = "1234567890qwertyuiopasd=fghjklzxcvbnm()\t '"
allowed = string.printable

class String:
    def __init__(self, arrlen, length, fittnessfunc, genfunc=None, letters=allowed):
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
            result_str = ''.join(r.choice(self.letters) for i in range(length))
            ret.append(result_str)
        return ret

    def run(self, data, func):
        if len(data) - 1 < self.arrlen:
            data = data + self.genfunc(self.length, self.arrlen - (len(data) - 1))
        return data


class StringMutate:
    def __init__(self, mutation_function="change", percentage=5, letters=allowed, small_percent=30):
        """Percentage is the percent of 'children' that will be mutated. small_percent is the percent of letters
        within the child that will be changed. Not this is all chance."""
        self.mutation_function = mutation_function
        self.percentage = percentage
        self.letters = letters
        self.small_percent = small_percent

    def mix(self, data):
        ret = []
        for i in data:
            if r.randint(1, 100) <= self.percentage:
                l = list(i)
                r.shuffle(l)
                ret.append("".join(l))
            else:
                ret.append(i)
        return ret

    def mutate_one(self, element):  # TODO: make this lots better
        for i in range(len(element)):
            if r.randint(1, 100) <= self.small_percent:
                ind = r.randint(0, len(element) - 1)
                thing = random.choice(self.letters)
                element = list(element)
                element.pop(ind)
                element.insert(ind, thing)
                element = "".join(element)
        return element

    def change(self, data):
        return [self.mutate_one(i) if r.randint(1, 100) <= self.percentage else i for i in data]

    def run(self, data, func):
        if self.mutation_function == "mix":
            return self.mix(data)
        if self.mutation_function == "change":
            return self.change(data)


class Parent:
    def __init__(self, num, typ="", gene_size=3, family_size=2):
        self.num = num
        self.gene_size = gene_size
        self.fs = family_size

    def parent(self, data, mom, dad):
        mom = np.asarray(list(mom))
        dad = np.asarray(list(dad))
        l = len(mom)
        mom = np.array_split(mom, self.gene_size)
        dad = np.array_split(dad, self.gene_size)
        mom = ["".join(l.tolist()) for l in mom]
        dad = ["".join(l.tolist()) for l in dad]
        children = []
        dad = dad + mom
        both = dad
        for i in range(self.fs):
            r.shuffle(both)
            children.append("".join(both)[0: l])
            data.pop(0)
        data = data + children + mom + dad
        return data

    def run(self, data, func):
        return self.parent(data, data[-1], data[-2])


class Environment:
    def __init__(self, *args):
        self.classes = list(args)

    def compile(self, epochs, func, verbose=True):
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
            if verbose:
                if n % 10 == 0:
                    print(int((n / epochs) * 100), top)
        return data, history

    def add(self, layer):
        self.classes.append(layer)


class Kill:
    def __init__(self, percent):
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


