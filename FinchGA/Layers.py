import copy
import math

import numpy as np
import Finch.FinchGA.EvolveRates as er
from Finch.FinchGA.generic import Chromosome, Individual


class Layer:
    def __init__(self, every=1, delay=0, native_run=None, end=math.inf):
        """
        :param every: do this every x times
        :param delay: Delay until used, until then it will simply return what it is given
        :param native_run: The run function to be used from other classes
        :param end: stop after n epochs
        """
        self.delay = delay
        self.native_run = native_run
        self.every = every
        self.end = end
        if not callable(every):
            self.every = er.Rates(every, 0).constant
        self.iterations = 1

    def run(self, data, func):
        """

        :param data: The Generation type to be modified
        :param func: Any relevant function that is needed in the native_run
        :return: Whatever native_function returns
        """
        if self.iterations >= self.end:
            return data
        if self.iterations % self.every() == 0:
            if self.delay <= 0:
                new = self.native_run(data, func)
                data = new
                return data
            else:
                self.delay -= 1
                return data
        else:
            self.iterations += 1
            return data


class GenerateData(Layer):
    def __init__(self, gene_pool, population, array_length, delay=0, every=1, end=math.inf):
        """
        :param gene_pool: The gene pool to be used
        :param population: The population to be generated
        :param array_length:
        """
        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay,
                         native_run=self.native_run)  # the Layer class
        self.gene_pool = gene_pool
        self.population = population
        self.array_len = array_length

    def native_run(self, data, func):
        self.gene_pool.gen_data(data, self.population, self.array_len)
        return data.individuals


class NarrowGRN(Layer):  # Narrow Gene Regulatory Network. Promotes good chromosome (not individuals).
    def __init__(self, gene_pool, every=1, method="outer", amount=10, delay=0, reward=0.01, penalty=0.01, mx=1, mn=1,
                 end=math.inf):
        """
        :param gene_pool: The gene_pool to modify
        :param method: Can also be "all" defines how to calculate new weights. "all" recalculate
        all of them, "outer" will penalize the lowest fitness ones and reward the highest fitness. "best" will reward
        the best. "worst" will penalize the worst chromosome.
        :param amount: The amount of individuals to look at. Only relevant when the method is not "all".
        :param delay: The delay
        :param reward: The percentage to increase the weight of a gene
        :param penalty: Like reward
        """
        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.amount = er.make_constant_rate(amount)
        self.reward = er.make_constant_rate(reward)
        self.penalty = er.make_constant_rate(penalty)
        self.gene_pool = gene_pool
        self.method = method
        self.mx = mx
        self.mn = mn

    def best(self, data):
        """
        :param data: The data
        :return: data
        """

        best = data.individuals[-int(self.amount()):]  # Most
        fitest = best[-1].fitness
        if fitest == 0:
            return data
        for individual in best:
            for gene in individual.chromosome.genes:
                gene.weight *= 1 + self.reward()
                gene.weight = min(gene.weight, self.mx)

        return data

    def worst(self, data):  # Not working yet

        worst = data.individuals[0: int(self.amount())]  # Least fit
        best = data.individuals[-1].fitness
        if best == 0:
            return data
        for individual in worst:
            for gene in individual.chromosome.genes:
                gene.weight -= self.penalty()
                gene.weight = max(gene.weight, self.mn)
        return data

    def outer(self, data):  # not working yet
        """
        :param data: The data
        :return: data
        """

        data = self.best(data)
        data = self.worst(data)
        return data

    def alld(self, data):  # not working yet
        n = 0
        best = data.individuals[-1].fitness
        if best == 0:
            return None
        for individual in data.individuals:
            for gene in individual.chromosome.genes:
                gene.weight += (individual.fitness / best)

    def native_run(self, data, func):

        """
        :param data: T
        :param func: this does nothing
        :return:
        """
        if self.method == "all":
            self.alld(data)
        if self.method == "worst":
            self.worst(data)
        if self.method == "best":  # only working one right now
            self.best(data)
        if self.method == "outer":
            self.outer(data)

        return data


class SortFitness(Layer):
    def __init__(self, every=1, delay=0, end=math.inf):
        """
        :param every: same in Layer
        :param delay: same
        :param end: same
        """
        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)

    @staticmethod
    def native_run(data, func):
        """
        :param data: The generation to be sorted
        :param func:
        :return:
        """
        data.individuals = np.array(data.sort())
        return data


class Duplicate(Layer):
    def __init__(self, every=1, clones=1, delay=0, end=math.inf):
        """
        :param every: See Layer
        :param clones: Amount of clones
        :param delay: See Layer
        :param end: same
        """
        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.clones = clones

    def native_run(self, data, func):
        for i in range(self.clones):
            data.add(data.individuals[-1])
        return data


class Mutate(Layer):
    def __init__(self, pool, every=1, delay=0, select_percent=100, likelihood=10, end=math.inf):
        """
        :param pool: The gene pool to look through
        :param every: Do this every n
        :param delay: Do this every n after delay in epochs
        :param select_percent: The percent of individuals to select for mutation
        :param likelihood: The likelihood of any individual gene mutating from within the individual (if selected)
        """
        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.likelihood = likelihood
        self.pool = pool
        self.percent = er.make_constant_rate(select_percent)
        self.likelihood = er.make_constant_rate(likelihood)

    def native_run(self, data, func):
        for i in data.individuals:
            i.mutate(self.pool, self.likelihood, self.percent)
        return data


class Kill(Layer):
    def __init__(self, percent, every=1, delay=0, end=math.inf):
        """
        :param percent: The percent to kill (picks from the worst)
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        """
        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.name = "kill"
        self.percent = percent

        if not callable(percent):
            self.percent = er.Rates(percent, 0).constant
        self.now = 0

    def native_run(self, data, func):
        data.individuals = data.individuals[int(len(data.individuals) * self.percent()):-1]
        return data


class KillByAge(Layer):
    def __init__(self, age, every=1, delay=0, end=math.inf):
        """
        :param percent: The percent to kill (picks from the worst)
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        """
        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.age = er.make_constant_rate(age)

    def native_run(self, data, func):
        ret = []
        for i in data.individuals:
            if i.age <= self.age():
                ret.append(i)
        data.individuals = np.array(ret)
        return data


class UpdateWeights(Layer):
    def __init__(self, pool, every=1, delay=0, end=math.inf):
        """
        :param pool: The gene pool to update
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        """
        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.pool = pool

    def native_run(self, data, func):
        self.pool.update()
        return data


class Function(Layer):
    def __init__(self, fun, every=1, delay=0, end=math.inf):
        """
        :param fun: The function you want to apply
        :param every: See Layer
        :param delay: See Layer
        :param end: See Layer
        """
        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.func = fun

    def native_run(self, data, func):
        self.func(data=data)


class Parent(Layer):
    def __init__(self, pool, every=1, gene_size=3, family_size=2, delay=0, native_run=None, end=math.inf):
        """
        :param pool: The gene pool to use
        :param every: Do this every n epochs
        :param gene_size: The gene size will determine how to mux parents
        :param family_size: The amount of children to generate
        :param delay: The delay in epochs until this takes effect
        :param native_run: Ignore this
        :param end: When to stop in epochs
        """
        self.gene_size = gene_size
        self.fs = family_size
        self.func = self.native_run
        self.pool = pool
        if native_run is not None:
            self.func = native_run

        if not callable(gene_size):
            self.gene_size = er.Rates(gene_size, 0).constant
        if not callable(family_size):
            self.fs = er.Rates(family_size, 0).constant

        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay, native_run=self.func)

    def parent(self, X, Y):
        ret = np.array([])
        # get their raw data
        x = np.asarray(X.chromosome.get_raw())
        y = np.asarray(Y.chromosome.get_raw())

        for i in range(self.fs()): # create fs() amount of children
            choice = np.random.choice([1, 0], size=min(x.shape[0], y.shape[0])) # Basically creates a mask
            both = []
            combined = list(zip(x, y))

            n = 0
            for i in choice:
                if np.all(both != combined[n][i]) or self.pool.replacement:
                    both.append(combined[n][i]) # Choose which ones

                n += 1
            new = copy.deepcopy(X)
            new.chromosome.set_raw(both)
            new.fit(1)
            ret = np.append(ret, new)

        return ret

    def native_run(self, data, func):
        data.add(self.parent(data.individuals[-1], data.individuals[-2]))
        return data


class Parents(Parent):
    def __init__(self, pool, delay=0, every=1, gene_size=3, family_size=2, percent=100, method="random",amount=10, end=math.inf):
        """
        :param delay: The delay in epochs until this will come into affect
        :param every: Do this ever n epochs
        :param gene_size: The gene size will determine how to mux parents
        :param family_size: The amount of children to generate
        :param percent: The percent to select when method=random
        :param method: Right now only "random" TODO: add more methods
        :param amount: if method is "best" then parents only the best ones
        """
        super().__init__(pool=pool, end=end, every=er.make_constant_rate(every), delay=delay, gene_size=gene_size,
                         family_size=family_size,
                         native_run=self.native_run)
        self.percent = percent/100
        self.method = method
        self.amount = er.make_constant_rate(amount)
        if not callable(percent):
            self.percent = er.Rates(percent/100, 0).constant

    def random(self, data, func): # completely random method
        these_ones = np.random.choice(data.individuals, int(data.individuals.size * self.percent()))
        for i in these_ones:
            parent1 = i
            parent2 = np.random.choice(these_ones, 1)[0]
            data.add(self.parent(parent1, parent2))
        return data
    def best(self, data, func):
        these_ones = data.individuals[-int(self.amount()*self.percent()):]
        for i in these_ones:
            parent1 = i
            parent2 = np.random.choice(these_ones, 1)[0]
            data.add(self.parent(parent1, parent2))
        return data
    def native_run(self, data, func):
        if self.method == "random":
            return self.random(data, func)
        if self.method == "best":
            return self.best(data, func)
        return data


class Age(Layer):
    def __init__(self, delay=0, every=1, years=1, end=math.inf):
        """
        :param delay:
        :param every:
        :param years: The amount of years to age an individual
        """
        super().__init__(end=end, every=every, delay=delay, native_run=self.native_run)
        self.years = years

    def native_run(self, data, func):
        for i in data.individuals:
            i.age += self.years
        return data


class KeepLength(Layer):
    def __init__(self, amount, delay=0, every=1, end=math.inf):
        """
        :param delay:
        :param every:
        :param years: The amount of years to age an individual
        """
        super().__init__(end=end, every=every, delay=delay, native_run=self.native_run)
        self.amount = er.make_constant_rate(amount)

    def native_run(self, data, func):
        data.individuals = data.individuals[-int(self.amount()):]
