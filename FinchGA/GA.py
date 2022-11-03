import os
import random
import random as r
import string
from Finch.FinchGA.EvolveRates import *
from Finch.FinchGA.generic import *
import numpy as np
from difflib import SequenceMatcher
import json

# TODO: Where i left off
"""
Add delay arg to each init and then call super().__init__(delay=delay)
"""
# TODO: switch to numpy
allowed = list(string.printable)


class Layer:
    def __init__(self, delay=0):
        self.num = .0
        self.delay = delay


class Data(Layer):
    def __init__(self, arrlen, length, genfunc=None):
        self.arrlen = arrlen
        self.length = length
        super().__init__()
        if genfunc is not None:
            self.genfunc = genfunc
        else:
            self.genfunc = self.gen

    @staticmethod
    def gen(length, num, func):
        """

        :param length:
        :param num:
        :param func: the fitness function to be used
        :return:
        """
        global allowed
        # choose from all lowercase letter
        ret = []
        for i in range(num):
            result_str = [r.choice(allowed) for _ in range(length)]
            ret += Individual(result_str, func)  # it will return a list of Individuals
        return ret

    def run(self, data, func):
        if self.num <= self.delay:
            if len(data) - 1 < self.arrlen:
                data = data + self.genfunc(self.length, self.arrlen - (len(data) - 1))
        else:
            self.num += 1
        return data


class DataMutate(Layer):
    def __init__(self, mutation_function="change", select_percent=None,
                 percent_mutate=None, delay=0):
        global allowed
        """Percentage is the percent of 'children' that will be mutated. small_percent is the percent of letters
        within the child that will be changed. Not this is all chance."""
        super().__init__(delay)
        if not callable(percent_mutate):
            percent_mutate = Rates(percent_mutate, 0).constant
        if not callable(select_percent):
            select_percent = Rates(select_percent, 0).constant

        self.mutation_function = mutation_function
        self.small_function = percent_mutate
        self.big_function = select_percent

    def mix(self, data):
        """Shuffles each individual, rarely helpful but who knows"""
        ret = []
        for individual in data:
            if r.randint(1, 100) <= self.small_function():
                r.shuffle(individual)
                ret.append(individual)
            else:
                ret.append(individual)  # Don't shuffle
        return ret

    def mutate_one(self, element):  # TODO: make this lots better
        global allowed
        for i in range(len(element)):
            m = self.small_function()
            if r.randint(1, 100) <= m:
                ind = r.randint(0, len(element) - 1)
                thing = random.choice(allowed)
                element.pop(ind)
                element.insert(ind, thing)
                element = element
        return element

    def change(self, data):
        return [self.mutate_one(i) if r.randint(1, 100) <= self.big_function() else i for i in data]

    def run(self, data, func):
        if self.num <= self.delay:
            self.num += 1
            return data
        if not data:
            return []
        if self.mutation_function == "mix":
            return self.mix(data)
        if self.mutation_function == "change":
            return self.change(data)


class Parent(Layer):
    def __init__(self, num, gene_size=3, family_size=2, delay=0):
        self.num = num
        self.gene_size = gene_size
        self.fs = family_size
        super().__init__(delay=delay)

    def parent(self, X, Y):
        ret = []
        X = np.asarray(X)
        Y = np.asarray(Y)
        for i in range(self.fs):
            choice = np.random.randint(self.gene_size, size=X.size).reshape(X.shape).astype(bool)
            ret.append(np.where(choice, X, Y).tolist()[0:X.size])
        return ret

    def deprecated_parent(self, mom, dad):
        ret = []
        length = len(mom)

        mom = [list(i) for i in np.array_split(mom, self.gene_size)]
        dad = [list(i) for i in np.array_split(dad, self.gene_size)]
        both = mom + dad
        both = [item for sublist in both for item in sublist]
        r.shuffle(both)
        for i in range(self.fs):
            r.shuffle(both)

            ret.append(both[0:length])
        return ret

    def run(self, data, func):
        if self.num <= self.delay:
            self.num += 1
            return data
        return data + self.parent(data, data[-1], data[-2])

    def prun(self, data, func):
        self.run(data, func)


class Environment:
    def __init__(self, classes=[]):
        self.classes = classes
        self.data = []

    def save(self, env_name):
        os.mkdir("model")
        input([vars(i) for i in self.classes])
        with open("model/" + env_name + ".json", "w+") as f:
            json.dump({"classes": [str(i) for i in self.classes], "data": self.data}, f)

    def compile(self, epochs, func, verbose=True, every=10, lettrs=None, data=[]):
        global allowed
        top = []
        if lettrs is None:
            lettrs = allowed
        allowed = lettrs
        self.data = data
        """Epochs = number of repititions, func is the fitness function, verbose is wheather or not to pring
        every is 'print data every x epochs' """
        history = []
        for n in range(epochs):
            for i in self.classes:

                self.data = i.run(data=self.data,
                                  func=func)

                # transform data in such a way as defined in the run func of the class
                self.data = [func(i) for i in self.data]  # apply the fitness function to each

                self.data = sorted(self.data)  # sort the data based on fitness
                try:
                    history.append(self.data[-1][0])

                    top = self.data[-1]
                except IndexError:
                    top = []
                self.data = [i[1] for i in
                             self.data]  # get rid of fitness data #TODO: make this better performance or get rid of it.
            if verbose:
                if n % every == 0:
                    print(int((n / epochs) * 100), top)
                    pass

        return self.data, history

    def summary(self):
        sum = ""
        for i in self.classes:
            sum += str(type(i)) + str(vars(i)) + "\n\n"
        return sum

    def add(self, layer):
        self.classes.append(layer)


class Narrow(Layer):
    def __init__(self, delay=0):
        super().__init__(delay=delay)

    def run(self, data, func):
        global allowed
        if self.num >= self.delay:
            allowed = np.unique(data[-1])
        else:
            self.num += 1
        return data


class Kill(Layer):
    def __init__(self, percent, delay=0):
        super().__init__(delay=delay)
        self.name = "kill"
        if type(percent) == int or type(percent) == float:
            percent = Rates(percent, 0).constant()
        self.percent = percent
        self.now = 0

    def run(self, data, func):
        if self.now >= self.delay:
            ret = data[int(len(data) * self.percent()):-1]
            return ret
        else:
            self.now += 1
            return data


class Parents(Parent):
    def __init__(self, delay, num_children, gene_size=3, family_size=2, percent=50, kill_parents=False):
        self.percent = percent
        self.kill_parents = kill_parents
        super().__init__(delay=delay, num=num_children, gene_size=gene_size, family_size=family_size)

    def run(self, data, func):
        ret = []
        for i in data:
            if random.randint(0, 100) <= self.percent:
                mom = random.choice(data)
                dad = i
                if self.kill_parents:
                    ret = ret + self.parent(dad, mom)  # .parent() is from the super class Parent.
                else:
                    ret = ret + self.parent(dad, mom) + [mom] + [dad]  # keeps the parents
            else:
                ret = ret + [i]
        return ret


class Duplicate(Layer):
    def __init__(self, amt, delay=0):
        super().__init__(delay=delay)
        self.amt = amt

    def run(self, data, func):
        if self.num >= self.delay:
            del data[0:self.amt]
            data += [data[-1]] * self.amt
            return data
        else:
            self.num += 1
            return data


class TopPercent(Layer):
    """Kill all but top percent"""

    def __init__(self, percent, delay=0):
        super().__init__(delay=delay)
        self.percent = percent

    def run(self, data, func):
        if self.num >= self.delay:
            return data[int(len(data) * self.percent):]
        else:
            self.num += 1
            return data


class TopN(Layer):
    """Kill all but top n"""

    def __init__(self, number, delay=0):
        super().__init__(delay=delay)
        self.number = number

    def run(self, data, func):
        if self.num >= self.delay:

            return data[-self.number:]
        else:

            self.num += 1
            return data


class RemoveTwins(Layer):
    def __init__(self, delay=0):
        super().__init__(delay)

    def run(self, data, func):
        if self.num <= self.delay:
            return np.unique(data).tolist()
        else:
            self.num += 1
            return data


class ParentOpposites(Parent):
    def __init__(self, amount, num_children, gene_size=3, family_size=2, percent=50, kill_parents=False, delay=0):
        self.percent = percent
        self.amount = amount
        self.kill_parents = kill_parents
        super().__init__(delay=delay, num=num_children, gene_size=gene_size, family_size=family_size)

    def run(self, data, func):
        ret = []
        lnd = len(data) - self.amount

        for i in range(self.amount):
            if random.randint(0, 100) <= self.percent:
                mom = data[lnd + i]
                dad = data[-lnd - i]
                if self.kill_parents:
                    ret = ret + self.parent(dad, mom)  # .parent() is from the super class Parent.
                else:
                    ret = ret + self.parent(dad, mom) + [mom] + [dad]  # keeps the parents
            else:
                ret = ret + [i]
        return ret


class KeepLength(Layer):
    def __init__(self, amt, delay=0):
        super().__init__(delay=delay)
        self.amt = amt

    def run(self, data, func):
        if self.num >= self.delay:
            return data[-self.amt:]
        else:
            self.num += 1
            return data


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def ifin(ls, str1):
    for i in ls:
        if i in str1:
            return True
        else:
            return False
