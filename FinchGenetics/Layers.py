# TODO: add Gene Regulatory Networks
import copy
import math
import random
import numpy as np
import Finch.FinchGenetics.GenePools as GenePools
from Finch.FinchGenetics.Genetics import Individual
from Finch.FinchGenetics.Rates import make_callable


def mul_tup(tup):
    base = 1
    for i in tup:
        base = base * i
    return base


class Layer:
    def __init__(self, every=1, delay=0, iterations=1, native_run=None, end=math.inf):
        """
        :param every: Do this every x
        :param delay: Don't do this until x
        :param iterations: When you do this, do it x times
        :param native_run: The run function in the layer
        :param end: Stop doing this after x
        """
        self.delay = delay
        self.native_run = native_run
        self.every = make_callable(every)
        self.end = end
        self.iterations = make_callable(iterations)
        self.n = 0

    def run(self, data):
        """

        :param data: The Generation type to be modified
        :param func: Any relevant function that is needed in the native_run
        :return: Whatever native_function returns
        """
        if self.n >= self.end:
            return data
        if self.n % self.every() == 0:
            if self.delay <= 0:
                for i in range(self.iterations()):
                    new = self.native_run(data)
                    data = new
                    return data
            else:
                self.delay -= 1
                return data
        else:
            self.n += 1
            return data


class Generate(Layer):
    def __init__(self, pool: GenePools.Pool, amount, delay=0, end=math.inf, every=1, iterations=1):
        """
        :param pool:
        :param amount: The gene pool you are using
        :param shape: The shape of each individual
        :param delay: Don't do this until x iters
        :param end: Stop after x iters
        :param every: Do this every x iters
        :param iterations: Do this x times every time you do it.
        """
        super().__init__(delay=delay, end=end, every=every, iterations=iterations, native_run=self.run)
        self.pool = pool
        self.amount = amount

    def run(self, data):
        data = self.pool.gen_data(data, self.amount)
        return data


class SortFitness(Layer):
    def __init__(self, delay=0, end=math.inf, every=1, iterations=1, native_run=None):
        if native_run == None:
            native_run = self.run
        super().__init__(delay=delay, end=end, every=every, iterations=iterations, native_run=self.run)

    def run(self, data):
        data = np.asarray(data)
        sort = np.argsort([i.fitness for i in data]).astype(int)  # O(2n) ish
        data = data[sort]
        return data
class SortAllFitness(SortFitness):
    def __init__(self, delay=0, end=math.inf, every=1):
        super().__init__(delay=delay, end=end, every=every, iterations=1, native_run=self.run)
    def run(self, data):
        for individual in data:
            individual.fit(1)
        return super().native_run(data)
class Duplicate(Layer):
    def __init__(self, every=1, clones=1, delay=0, end=math.inf):
        """
        :param every: See Layer
        :param clones: Amount of clones
        :param delay: See Layer
        :param end: same
        """
        super().__init__(end=end, every=make_callable(every), delay=delay, iterations=clones,
                         native_run=self.native_run)

    def native_run(self, data):
        data = np.append(data, data[-1])
        return data


class Kill(Layer):
    def __init__(self, percent, every=1, delay=0, end=math.inf):
        """
        :param percent: The percent to kill (picks from the worst)
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        """
        super().__init__(end=end, every=make_callable(every), delay=delay, native_run=self.native_run, iterations=1)
        self.name = "kill"
        self.percent = make_callable(percent)

        self.now = 0

    def native_run(self, data, func):
        data = data[int(len(data) * self.percent()):-1]
        return data


class KillByAge(Layer):
    def __init__(self, age, every=1, delay=0, end=math.inf):
        """
        :param percent: The percent to kill (picks from the worst)
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        """
        super().__init__(end=end, every=make_callable(every), delay=delay, native_run=self.native_run, iterations=1)
        self.age = make_callable(age)

    def native_run(self, data, func):
        ret = []
        for i in data:
            if i.age <= self.age():
                ret.append(i)
        data = np.array(ret)
        return data


class Function(Layer):
    def __init__(self, fun, every=1, delay=0, iterations=1, end=math.inf):
        """
        :param fun: The function you want to apply
        :param every: See Layer
        :param delay: See Layer
        :param end: See Layer
        :param iterations: do this x times
        """
        super().__init__(end=end, every=make_callable(every), delay=delay, native_run=self.native_run,
                         iterations=iterations)
        self.func = fun

    def native_run(self, data):
        return self.func(data=data)


class Parent(Layer):
    def __init__(self, pool, every=1, gene_size=3, family_size=2, delay=0, iterations=1, native_run=None, end=math.inf):
        """
        :param pool: The gene pool to use
        :param every: Do this every n epochs
        :param gene_size: The gene size will determine how to mux parents
        :param family_size: The amount of children to generate
        :param delay: The delay in epochs until this takes effect
        :param native_run: Ignore this
        :param end: When to stop in epochs
        :param iterations: Do this x times
        """
        self.func = self.native_run
        self.pool = pool
        if native_run is not None:
            self.func = native_run
        self.gene_size = make_callable(gene_size)
        self.fs = make_callable(family_size)

        super().__init__(end=end, every=make_callable(every), delay=delay, native_run=self.func, iterations=iterations)

    def parent(self, X, Y):

        shape = X.genes.shape
        ret = np.array([])
        # get their raw data
        x = np.asarray(X.genes).reshape((-1, self.gene_size()))
        y = np.asarray(Y.genes).reshape((-1, self.gene_size()))
        if self.pool.treat_sublists_as_genes:
            x = x.reshape(self.pool.shape)
            y = y.reshape(self.pool.shape)
        for i in range(
                self.fs()):  # create fs() amount of children #TODO: I am unsure if anything in this loop is correct
            choice = np.array(random.choices([np.ones((1,) + x.shape[1:]), np.zeros((1,) + x.shape[1:])],
                                             k=mul_tup(shape)))  # Basically creates a mask
            choice.resize(x.shape)
            # print(choice)

            both = np.where(choice, x, y)
            both = np.concatenate(both).ravel()
            new = Individual(X.pool, copy.deepcopy(X.genes), X.fitness_func)
            if self.pool.replacement == False:
                # both = np.unique(both,axis=-1)
                pass
            new.genes = both
            new.fit(1)
            ret = np.append(ret, new)

        return ret

    def native_run(self, data):
        data = np.append(self.parent(data[-1], data[-2]), data)
        return data


class Parents(Parent):
    def __init__(self, pool, delay=0, every=1, gene_size=3, family_size=2, percent=1, method="random", amount=10,
                 end=math.inf):
        """
        :param delay: The delay in epochs until this will come into affect
        :param every: Do this ever n epochs
        :param gene_size: The gene size will determine how to mux parents
        :param family_size: The amount of children to generate
        :param percent: The percent to select when method=random
        :param method: Right now only "random" TODO: add more methods
        :param amount: if method is "best" then parents only the best ones
        """
        super().__init__(pool=pool, end=end, every=every, delay=delay, gene_size=gene_size,
                         family_size=family_size,
                         native_run=self.native_run)
        self.percent = make_callable(percent)
        self.method = method
        self.amount = make_callable(amount)

    def random(self, data):  # completely random method
        these_ones = random.choices(data, k=int(data.size * self.percent()))
        for i in these_ones:
            parent1 = i
            parent2 = random.choice(these_ones)
            data = np.append(data, self.parent(parent1, parent2))
        return data

    def best(self, data):
        these_ones = data[-int(self.amount() * self.percent()):]
        for i in these_ones:
            parent1 = i
            parent2 = random.choice(these_ones)
            data = np.append(data, self.parent(parent1, parent2))
        return data

    def native_run(self, data):
        if self.method == "random":
            return self.random(data)
        if self.method == "best":
            return self.best(data)
        return data


class KeepLength(Layer):
    def __init__(self, amount, delay=0, every=1, end=math.inf):
        """
        :param delay: Don't do until x iters
        :param every: Do this every other x iters
        """
        super().__init__(end=end, every=every, delay=delay, native_run=self.native_run, iterations=1)
        self.amount = make_callable(amount)

    def native_run(self, data):
        data = data[-int(self.amount()):]
        return data


class RemoveDuplicatesFromTop(Layer):
    def __init__(self, delay=0, every=1, end=math.inf, amount=2):
        """
        :param delay: same
        :param every: same
        :param end: same
        :param amount: The amount of individuals to check (at the top).
        """
        self.amount = amount
        super().__init__(delay=delay, every=every, end=end, native_run=self.native_run, iterations=1)

    def native_run(self, data):
        for i in range(self.amount):
            try:
                if np.all(data[-i].genes == data[-(i + 1)].genes):
                    data = data[0:-2]  # deletes last element
            except ValueError:
                pass
        return data


class FastMutateTop(Layer):
    def __init__(self, pool, delay=0, every=1, end=math.inf, amount=3, individual_mutation_amount=1,
                 fitness_mix_factor=1, adaptive=False):
        self.pool = pool
        self.individual_select = make_callable(individual_mutation_amount)
        self.amount = make_callable(amount)
        self.fitness_mix_factor = fitness_mix_factor
        super().__init__(delay=delay, every=every, end=end, native_run=self.native_run)

    def native_run(self, data):
        these_ones = data[-self.amount():]
        l = len(these_ones)
        for i in range(l - 1):
            this = these_ones[i]
            k = int(self.individual_select())
            choices = random.choices(list(range(0, len(this.genes) - 1)), k=k)
            these_ones[i].genes[choices] = self.pool.rand_many(amount=k)
            these_ones[i].fit(self.fitness_mix_factor)
        return data
