import numpy as np
import Finch.FinchGA.EvolveRates as er


class Layer:
    def __init__(self, every=1,  delay=0, native_run=None):
        """
        :param delay: Delay until used, until then it will simply return what it is given
        :param native_run: The run function to be used from other classes
        """
        self.delay = delay
        self.native_run = native_run
        self.every = every
        if not callable(every):
            self.every = er.Rates(every, 0).constant
        self.iterations = 1

    def run(self, data, func):
        """

        :param data: The Generation type to be modified
        :param func: Any relevant function that is needed in the native_run
        :return: Whatever native_function returns
        """

        if self.iterations % self.every() == 0:
            self.iterations = 1
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
    def __init__(self,  gene_pool, population, array_length, delay=0, every=1):
        """
        :param gene_pool: The gene pool to be used
        :param population: The population to be generated
        :param array_length:
        """
        super().__init__(every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)  # the Layer class
        self.gene_pool = gene_pool
        self.population = population
        self.array_len = array_length

    def native_run(self, data, func):
        self.gene_pool.gen_data(data, self.population, self.array_len)
        return data.individuals


class NarrowGRN(Layer):  # Narrow Gene Regulatory Network. Promotes good chromosome (not individuals).
    def __init__(self, gene_pool, every=1, method="outer", amount=10, delay=0, reward=0.01, penalty=0.01):
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
        super().__init__(every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.amount = amount
        self.reward = reward
        self.penalty = penalty
        if not callable(reward):
            self.reward = er.Rates(reward, 0).constant  # the reward will remain the same
        if not callable(penalty):
            self.penalty = er.Rates(penalty, 0).constant  # the penalty will remain the same

        if not callable(amount):
            self.amount = er.Rates(amount, 0).constant  # the amount will remain the same
        self.gene_pool = gene_pool
        self.method = method

    def best(self, data):
        """
        :param data: The data
        :return: data
        """

        best = data.individuals[-int(self.amount()):]  # Most

        for individual in best:
            for gene in individual.chromosome.genes:
                gene.reward(self.reward())
        return data

    def worst(self, data):

        worst = data.individuals[0: int(self.amount())]  # Least fit
        for individual in worst:
            for gene in individual.chromosome.genes:

                gene.penalize(self.penalty())


        return data

    def outer(self, data):
        """
        :param data: The data
        :return: data
        """

        data = self.best(data)
        data = self.worst(data)
        return data

    def alld(self, data):
        n = 0
        for gene in data:
            pass  # TODO: implement this

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
        if self.method == "best":
            self.best(data)
        if self.method == "outer":
            self.outer(data)

        return data


class CalcFitness(Layer):
    def __init__(self, every=1,  delay=0):
        super().__init__(every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)

    @staticmethod
    def native_run(data, func):
        data.fit_all(1)
        data.individuals = np.array(data.sort())
        return data


class Duplicate(Layer):
    def __init__(self, every=1,  clones=1, delay=0):
        super().__init__(every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.clones = clones

    def native_run(self, data, func):
        for i in range(self.clones):
            data.add(data.individuals[-1])
        return data


class Mutate(Layer):
    def __init__(self, pool, every=1, delay=0, percent=100):
        super().__init__(every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.pool = pool
        self.percent = er.make_constant_rate(percent)
    def native_run(self, data, func):
        for i in data.individuals:
            i.mfunction(self.pool)
        return data


class Kill(Layer):
    def __init__(self, percent, every=1, delay=0):
        """
        :param percent: The percent to kill (picks from the worst)
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        """
        super().__init__(every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.name = "kill"
        self.percent = percent

        if not callable(percent):
            self.percent = er.Rates(percent, 0).constant
        self.now = 0

    def native_run(self, data, func):
        data.individuals = data.individuals[int(len(data.individuals) * self.percent()):-1]
        return data


class UpdateWeights(Layer):
    def __init__(self, pool, every=1, delay=0):
        """
        :param pool: The gene pool to update
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        """
        super().__init__(every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.pool = pool

    def native_run(self, data, func):
        self.pool.update()
        return data


class Function(Layer):
    def __init__(self, fun, every=1, delay=0):
        super().__init__(every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.func = fun

    def native_run(self, data, func):
        self.func(data=data)
class Parent(Layer):
    def __init__(self, every=1,  gene_size=3, family_size=2, delay=0, native_run=None):
        """
        :param every: Do this every n epochs
        :param gene_size: The gene size will determine how to mux parents
        :param family_size: The amount of children to generate
        :param delay: The delay in epochs until this takes effect
        :param native_run: Ignore this
        """
        self.gene_size = gene_size
        self.fs = family_size
        self.func = self.native_run
        if native_run is not None:
            self.func = native_run

        if not callable(gene_size):
            self.gene_size = er.Rates(gene_size, 0).constant
        if not callable(family_size):
            self.fs = er.Rates(family_size, 0).constant
        
        super().__init__(every=er.make_constant_rate(every), delay=delay, native_run=self.func)

    def parent(self, X, Y):
        ret = np.array([])
        X = np.asarray(X)
        Y = np.asarray(Y)
        for i in range(self.fs()):
            choice = np.random.randint(self.gene_size(), size=X.size).reshape(X.shape).astype(bool)
            ret = np.append(ret, np.where(choice, X, Y).tolist()[0:X.size])
        return ret
    def native_run(self, data, func):
        data.add(self.parent(data.individuals[-1], data.individuals[-2]))
        return data


class Parents(Parent):
    def __init__(self, delay=0, every=1, gene_size=3, family_size=2, percent=1, method="random"):
        """
        :param delay: The delay in epochs until this will come into affect
        :param every: Do this ever n epochs
        :param gene_size: The gene size will determine how to mux parents
        :param family_size: The amount of children to generate
        :param percent: The percent to select when method=random
        :param method: Right now only "random" TODO: add more methods
        """
        super().__init__(every=er.make_constant_rate(every), delay=delay, gene_size=gene_size, family_size=family_size, native_run=self.native_run)
        self.percent = percent
        self.method = method
        if not callable(percent):
            self.percent = er.Rates(percent, 0).constant
    def random(self, data, func):
        these_ones = np.random.choice(data.individuals, data.individuals.size*self.percent())
        for i in these_ones:
            parent1 = i
            parent2 = np.random.choice(these_ones, 1)
            data.add(self.parent(parent1, parent2))
        return data
    def native_run(self, data, func):
        if self.method == "random":
            return self.random(data, func)
        return data
