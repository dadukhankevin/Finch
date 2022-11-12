import copy
import logging
import math
import random
import numpy as np
import Finch.FinchGA.EvolveRates as er
from Finch.FinchGA.generic import Individual


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


class NarrowGRN(Layer):  # Narrow Gene Regulatory Network. Promotes good genes (not individuals).
    def __init__(self, gene_pool, every=1, method="fast", amount=10, delay=0, reward=0.01, penalty=0.01, end=math.inf):
        """
        :param gene_pool: The gene_pool to modify
        :param method: Can also be "all" defines how to calculate new weights. "all" recalculate
        all of them, "outer" will penalize the lowest fitness ones and reward the highest fitness. "best" will reward
        the best. "worst" will penalize the worst genes.
        :param amount: The amount of individuals to look at. Only relevant when the method is not "all".
        :param delay: The delay
        :param reward: The percentage to increase the weight of a gene
        :param penalty: Like reward
        """
        super().__init__(end=end, every=er.make_constant_rate(every), delay=delay, native_run=self.native_run)
        self.amount = er.make_constant_rate(amount)
        self.reward = er.make_constant_rate(reward)
        self.penalty = er.make_constant_rate(penalty)
        if method == "worst":
            gene_pool.set_all_weights(gene_pool.mx())
        if method == "outer":
            gene_pool.set_all_weights((gene_pool.mx() + gene_pool.mn()) / 2)
        self.gene_pool = gene_pool
        self.method = method

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
            for gene in individual.genes:
                Gene = self.gene_pool.get_weight(gene)[0]
                if individual.fitness > fitest:
                    Gene.weight += (self.reward())
                    Gene.weight = min(Gene.weight, self.gene_pool.mx())
        return data

    def worst(self, data):  # Not working yet

        worst = data.individuals[0: int(self.amount())]  # Least fit
        best = data.individuals[-1].fitness
        if worst == 0:
            return data
        for individual in worst:
            for gene in individual.genes:
                Gene = self.gene_pool.get_weight(gene)[0]
                if best > 0:
                    Gene.weight -= (self.penalty())
                    Gene.weight = max(Gene.weight, self.gene_pool.mn())

        return data

    def Fast(self, data):
        individuals = data.individuals[int(-self.amount()):]
        for i, individual in enumerate(individuals):
            ind, counts = np.unique(individual.genes, return_counts=True)
            l = ind.size
            for i, gene in enumerate(ind):
                Gene = self.gene_pool.get_weight(gene)[0]
                Gene.weight = (.3 * Gene.weight) + (.7 * max(self.gene_pool.mn(), counts[i] / l))

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
            for gene in individual.genes:
                gene.weight += (individual.fitness / best)

    def native_run(self, data, func):

        """
        :param data: T
        :param func: this does nothing
        :return:
        """
        if self.method == "fast":
            self.Fast(data)
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
        x = np.asarray(X.genes)
        y = np.asarray(Y.genes)

        for i in range(
                self.fs()):  # create fs() amount of children #TODO: I am unsure if anything in this loop is correct
            choice = np.array(random.choices([np.ones((1,) + x.shape[1:]), np.zeros((1,) + x.shape[1:])],
                                             k=x.size))  # Basically creates a mask
            choice.resize(x.shape)
            # print(choice)

            both = np.where(choice, x, y)
            new = copy.deepcopy(X)
            if self.pool.replacement == False:
                # both = np.unique(both,axis=-1)
                pass
            new.genes = both
            new.fit(1)
            ret = np.append(ret, new)

        return ret

    def native_run(self, data, func):
        data.add(self.parent(data.individuals[-1], data.individuals[-2]))
        return data


class Parents(Parent):
    def __init__(self, pool, delay=0, every=1, gene_size=3, family_size=2, percent=100, method="random", amount=10,
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
        super().__init__(pool=pool, end=end, every=er.make_constant_rate(every), delay=delay, gene_size=gene_size,
                         family_size=family_size,
                         native_run=self.native_run)
        self.percent = percent / 100
        self.method = method
        self.amount = er.make_constant_rate(amount)
        if not callable(percent):
            self.percent = er.Rates(percent / 100, 0).constant

    def random(self, data, func):  # completely random method
        these_ones = random.choices(data.individuals, k=int(data.individuals.size * self.percent()))
        for i in these_ones:
            parent1 = i
            parent2 = random.choice(these_ones)
            data.add(self.parent(parent1, parent2))
        return data

    def best(self, data, func):
        these_ones = data.individuals[-int(self.amount() * self.percent()):]
        for i in these_ones:
            parent1 = i
            parent2 = random.choice(these_ones)
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


class RemoveDuplicatesFromTop(Layer):
    def __init__(self, delay=0, every=1, end=math.inf, amount=2):
        self.amount = amount
        super().__init__(delay=delay, every=every, end=end, native_run=self.native_run)

    def native_run(self, data, func):
        for i in range(self.amount):
            try:
                if np.all(data.individuals[-i].genes == data.individuals[-(i + 1)].genes):
                    data.individuals = data.individuals[0:-2]  # deletes last element
            except ValueError:
                pass
        return data


class FastMutateTop(Layer):
    def __init__(self, pool, delay=0, every=1, end=math.inf, amount=3, individual_mutation_amount=.3,
                 fitness_mix_factor=1, adaptive=False):
        self.pool = pool
        self.individual_select = er.make_constant_rate(individual_mutation_amount)
        self.amount = er.make_constant_rate(amount)
        self.fitness_mix_factor = fitness_mix_factor
        super().__init__(delay=delay, every=every, end=end, native_run=self.native_run)

    def native_run(self, data, func):
        these_ones = data.individuals[-self.amount():]
        l = len(these_ones)
        for i in range(l - 1):
            this = these_ones[i]
            k = int(self.individual_select())
            choices = random.choices(list(range(0, len(this.genes) - 1)), k=k)
            these_ones[i].genes[choices] = self.pool.rand_many(index=1, amount=k)
            these_ones[i].fit(self.fitness_mix_factor)
        return data


class OverPoweredMutation(Layer):
    def __init__(self, pool, iterations, index, fitness_function, delay=0, every=1, end=math.inf):
        super().__init__(delay=delay, every=every, end=end, native_run=self.native_run)
        self.pool = pool
        self.iterations = iterations
        self.index = index
        self.fitness_function = fitness_function
        logging.warning("Using OverPoweredMutation will override any custom fitness factor you set for your specified "
                        "index.")

    def native_run(self, data, func):
        individual = data.individuals[self.index]
        fitness = individual.fit(1)
        l = len(individual.genes)
        for i in range(self.iterations):
            new = copy.deepcopy(individual)
            new.genes[random.randint(0, l - 1)] = self.pool.rand(index=1)
            newf = new.fit(1)
            if newf > fitness:
                data.individuals[self.index] = new
                fitness = newf
                individual = data.individuals[self.index]
        return data


class RecalculateDirectionalMutation(Layer):  # This might be a little overkill
    def __init__(self, pool, fitness_function, amount=1, change_constant=1, delay=0, every=1, end=math.inf):
        super().__init__(delay=delay, every=every, end=end, native_run=self.native_run)
        self.fitness_function = fitness_function
        self.amount = er.make_constant_rate(amount)
        self.pool = pool
        self.change_constant = er.make_constant_rate(change_constant)

        logging.warning("Use this only with float and int type individuals")

    def native_run(self, data, func):
        these_ones = data.individuals[-self.amount():]
        if self.pool.directional_weights is None:
            self.pool.directional_weights = np.zeros(len(these_ones[0]))
        for i, individual in enumerate(these_ones):
            for n, genes in enumerate(individual.genes):
                old = genes[n]
                fake_geneslow = copy.deepcopy(genes)
                fake_geneshigh = copy.deepcopy(genes)
                fake_geneshigh[n] += self.change_constant()
                fake_geneslow[n] -= self.change_constant()
                high = self.fitness_function(fake_geneshigh)
                low = self.fitness_function(fake_geneslow)
                if high > low:
                    self.pool.directional_weights[n] = 1
                    genes[n] = fake_geneshigh
                elif low > high:
                    self.pool.directional_weights[n] = -1
                    genes[n] = fake_geneslow
                else:
                    self.pool.directional_weights[n] = 0
                    # don't modify genes
        return data


class MutateByDirectionalWeights(Layer):
    def __init__(self, pool, fitness_function, amount=1, change_constant=1, delay=0, every=1, divisor=1.1,
                 end=math.inf):
        super().__init__(delay=delay, every=every, end=end, native_run=self.native_run)
        self.fitness_function = fitness_function
        self.amount = er.make_constant_rate(amount)
        self.pool = pool
        self.change_constant = er.make_constant_rate(change_constant)
        self.divisor = divisor
        logging.warning("Use this only with float and int type individuals")

    def native_run(self, data, func):
        these_ones = data.individuals[-self.amount():]
        if self.pool.directional_weights is None:
            self.pool.directional_weights = np.zeros(len(these_ones[0]))
        for i, individual in enumerate(these_ones):
            for n, genes in enumerate(individual.genes):
                genes += self.pool.directional_weights * self.change_constant()
                self.pool.directional_weights /= self.divisor
        return data
class DetermineFitnessAndSort(Layer):
    def __init__(self, factor =1 , delay=0, every=0, end=0):
        super().__init__(delay=delay, every=every, end=end, native_run=self.native_run)
        self.factor = factor
    def native_run(self, data, func):
        for individual in data.individuals:
            individual.fit(self.factor)
        data.sort()
        return data
