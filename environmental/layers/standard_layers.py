"""
Notes:
- Finish making fitness for Layers and Environments make sense, make a better fitness function that
can change quickly
- Apply this to the Adaptable Environment so that it can use the fitness data to switch environments
- Savable environments would be cool anyway
- Save the end sequence to a ChronologicalEnvironment so that you can use it for later ones
"""

import math
import numpy as np
from Finch.functions import parenting, selection
from Finch.genetics import Pool
from Finch.genetics.population import NPCP, Individual
from Finch.tools import rates
from Finch.functions.selection import Select

randomSelect = selection.RandomSelection()


class Layer(Individual):  # Layers are individuals, environments are layers,
    # thus environments and layers both have fitness
    def __init__(self):
        super().__init__(self, self.fit_func)
        self.fitness = 1
        self.before_m = 1
        self.after_m = 1
        self.total = 1
        self.runs = 1
        self.layer_history = []

    def Measure(run):
        def adjust(self, individuals, environment):
            try:
                self.before_m = environment.individuals[-1].fitness
                individuals = run(self, individuals, environment)
                self.after_m = environment.individuals[-1].fitness
            except IndexError:
                individuals = run(self, individuals, environment)
            super().fit()
            return individuals

        return adjust

    def fit_func(self, genes):  # don't do anything with genes but its gotta be there for now #TODO
        change = self.after_m - self.before_m
        self.total += change
        self.runs += 1
        self.layer_history.append(self.total)
        return self.total

    def run(self, individuals: list[Individual], environment: any):
        pass


class Populate(Layer):
    def __init__(self, gene_pool: Pool, population: int):
        super().__init__()
        print("a")
        self.gene_pool = gene_pool
        print("b")

        self.population = rates.make_callable(population)

    @Layer.Measure
    def run(self, individuals, environment):
        print("c")

        individuals = list(individuals)  # TODO fix the need for this... Is this fixed???? I have no idea
        while len(individuals) < self.population():
            print("d")

            new = self.gene_pool.generate()
            print("e")

            new.fit()
            print("f")

            individuals += [new]
            print("g")

        return individuals


class KillRandom(Layer):
    def __init__(self, num_kill: int, selection_function: selection.RandomSelection().select):
        super().__init__()
        self.num_kill = rates.make_callable(num_kill)
        self.selection_function = selection_function

    @Layer.Measure
    def run(self, individuals, environment):
        # select some individuals to kill using the selection function
        to_kill = self.selection_function(individuals, self.num_kill())
        # remove the selected individuals from the population
        individuals = [ind for ind in individuals if ind not in to_kill]
        return individuals


class Kill(Layer):
    def __init__(self, percent: float):
        """
        :param percent: The percent to kill (picks from the worst)
        """
        super().__init__()
        self.name = "kill"
        self.percent = rates.make_callable(percent)
        self.now = 0

    @Layer.Measure
    def run(self, individuals, environment):
        data = individuals[:int(len(individuals) * (1 - self.percent()))]
        return data


class DuplicateRandom(Layer):
    def __init__(self, num_duplicate: int, selection_function: callable(Select.select) = randomSelect.select):
        super().__init__()
        self.num_duplicate = rates.make_callable(num_duplicate)
        self.selection_function = selection_function

    @Layer.Measure
    def run(self, individuals, environment):
        # select some individuals to duplicate using the selection function
        duplicates = self.selection_function(individuals, self.num_duplicate())
        # append the duplicates to the population
        individuals = np.append(individuals, duplicates)
        return individuals


class RemoveDuplicatesFromTop(Layer):
    def __init__(self, top_n: int):
        super().__init__()
        self.top_n = rates.make_callable(top_n)

    @Layer.Measure
    def run(self, individuals, environment):
        # remove any duplicates from the top n individuals
        unique_individuals = individuals[:self.top_n()]
        for i in range(self.top_n, len(individuals)):
            if not any(NPCP.array_equal(ind.genes, individuals[i].genes) for ind in unique_individuals):
                unique_individuals.append(individuals[i])
        return unique_individuals


class SortByFitness(Layer):
    def __init__(self):
        super().__init__()

    @Layer.Measure
    def run(self, individuals, environment):
        sorted_individuals = sorted(individuals, key=lambda x: -x.fitness)

        return np.asarray(sorted_individuals)


class CapPopulation(Layer):
    def __init__(self, max_population: int):
        super().__init__()
        self.max_population = rates.make_callable(max_population)

    @Layer.Measure
    def run(self, individuals, environment):
        return individuals[0:self.max_population()]  # kills only the worst ones assuming they are sorted


class Function(Layer):
    def __init__(self, function):
        super().__init__()
        self.function = function

    @Layer.Measure
    def run(self, individuals, environment):
        return self.function(individuals)


class Controller(Layer):
    def __init__(self, layer: Layer, execute_every=1, repeat=1, delay=0, stop_at=math.inf):
        super().__init__()
        self.layer = layer
        self.every = rates.make_callable(execute_every)
        self.delay = rates.make_callable(delay)
        self.end = stop_at
        self.repeat = rates.make_callable(repeat)
        self.n = 0

    @Layer.Measure
    def run(self, individuals, environment):
        self.n += 1
        if self.delay() <= self.n <= self.end:
            if self.every() % self.n == 0:
                for i in range(self.repeat() + 1):
                    individuals = self.layer.run(individuals, environment)
        return individuals


class ParentBestChild(Layer):
    def __init__(self, num_families: int, selection_function: callable(Select.select) = randomSelect.select):
        super().__init__()
        self.parenting_object = parenting.BestChild(num_families, selection_function)

    @Layer.Measure
    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentBestChildBinary(Layer):
    def __init__(self, num_families: int, selection_function: callable(Select.select) = randomSelect.select):
        super().__init__()
        self.parenting_object = parenting.BestChildBinary(num_families, selection_function)

    @Layer.Measure
    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentSinglePointCrossover(Layer):
    def __init__(self, num_families: int, num_children: int, selection_function: callable(Select.select) = randomSelect.select):
        super().__init__()
        self.parenting_object = parenting.SinglePointCrossover(num_families, selection_function,
                                                               num_children)

    @Layer.Measure
    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentUniformCrossover(Layer):
    def __init__(self, num_families: int, num_children: int, selection_function: callable(Select.select) = randomSelect.select):
        super().__init__()
        self.parenting_object = parenting.UniformCrossover(num_families, selection_function, num_children)

    @Layer.Measure
    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentNPointCrossover(Layer):
    def __init__(self, num_families: int, num_children: int, selection_function: callable(Select.select) = randomSelect.select,
                 n=2):
        super().__init__()
        self.parenting_object = parenting.NPointCrossover(num_families, selection_function, num_children, n)

    @Layer.Measure
    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentUniformCrossoverMultiple(Layer):
    def __init__(self, num_families: int = 2, num_children: int = 2, selection_function: callable(Select.select) = randomSelect.select):
        super().__init__()
        self.parenting_object = parenting.UniformCrossoverMultiple(num_families, selection_function, num_children)

    @Layer.Measure
    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentByGeneSegmentation(Layer):
    def __init__(self, num_families: int, num_children: int, selection_function: callable(Select.select) = randomSelect.select,
                 gene_size: int = 2):
        super().__init__()
        self.parenting_object = parenting.ParentByGeneSegmentation(num_families, selection_function, gene_size,
                                                                   num_children)

    @Layer.Measure
    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class Parent(Layer):
    def __init__(self, num_families: int, num_children: int, selection_function: callable(Select.select) = randomSelect.select):
        super().__init__()
        self.parenting_object = parenting.SinglePointCrossover(num_families, selection_function, num_children)

    @Layer.Measure
    def run(self, individuals, environment):
        new = self.parenting_object.parent(individuals, environment, self)
        individuals += new
        return individuals


class RemoveAllButBest(Layer):
    def __init__(self):
        super().__init__()

    @Layer.Measure
    def run(self, individuals, environment):
        return [individuals[0]]


class FreezeRandom(Layer):
    def __init__(self, amount_select: int, amount_genes: int, selection_function: callable = randomSelect.select):
        super().__init__()
        self.amount_select = rates.make_callable(amount_select)
        self.amount_genes = rates.make_callable(amount_genes)
        self.selection_function = selection_function

    @Layer.Measure
    def run(self, individuals, environment):
        selected_individuals = self.selection_function(individuals, self.amount_select())
        for individual in selected_individuals:
            random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes(), replace=False)
            individual.freeze(random_indices)
        return individuals

class KillByFitnessPercentile(Layer):
    def __init__(self, percentile: float):
        super().__init__()
        self.percentile = rates.make_callable(percentile)

    @Layer.Measure
    def run(self, individuals, environment):
        num_to_kill = int(len(individuals) * self.percentile())
        sorted_individuals = sorted(individuals, key=lambda x: x.fitness)
        remaining_individuals = sorted_individuals[num_to_kill:]
        return remaining_individuals
