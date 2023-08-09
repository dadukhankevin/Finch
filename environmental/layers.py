import math

import numpy as np

# Try importing CuPy


from Finch.functions import parenting, selection

from Finch.genetics.population import NPCP
from Finch.tools import rates

randomSelect = selection.RandomSelection()


class Populate:
    def __init__(self, gene_pool, population):
        self.gene_pool = gene_pool
        self.population = rates.make_callable(population)

    def run(self, individuals, environment):
        individuals = list(individuals)  # TODO fix the need for this...
        while len(individuals) < self.population():
            individuals += [self.gene_pool.generate()]
        if environment.original is None:
            environment.original = individuals[0].genes
        return individuals


class MutateAmount:
    def __init__(self, amount_individuals, amount_genes, gene_pool, refit=True,
                 selection_function=randomSelect.select):
        self.amount_individuals = rates.make_callable(amount_individuals)
        self.amount_genes = rates.make_callable(amount_genes)
        self.gene_pool = gene_pool
        self.refit = refit
        self.selection_function = selection_function

    def run(self, individuals, environment):
        self.amount_individuals = rates.make_callable(min(len(individuals), self.amount_individuals()))
        selected_individuals = self.selection_function(individuals=individuals, amount=self.amount_individuals())
        for individual in selected_individuals:
            individual = self.mutate_one(individual)
            if self.refit:
                individual.fit()
        return individuals

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(len(individual.genes), self.amount_genes(), replace=False)
        individual.genes[random_indices] = self.gene_pool.generate_genes(self.amount_genes())
        return individual


class FloatMutateAmountUniform(MutateAmount):
    def __init__(self, amount_individuals, amount_genes, gene_pool, max_mutation=0.1
                 , refit=True, selection_function=randomSelect.select):
        super().__init__(amount_individuals, amount_genes, gene_pool, refit, selection_function)
        self.max_mutation = rates.make_callable(max_mutation)

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes(), replace=False)
        mutation = NPCP.random.uniform(-self.max_mutation(), self.max_mutation())
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual


class KillRandom:
    def __init__(self, num_kill, selection_function):
        self.num_kill = rates.make_callable(num_kill)
        self.selection_function = selection_function

    def run(self, individuals, environment):
        # select some individuals to kill using the selection function
        to_kill = self.selection_function(individuals, self.num_kill())
        # remove the selected individuals from the population
        individuals = [ind for ind in individuals if ind not in to_kill]
        return individuals


class Kill:
    def __init__(self, percent):
        """
        :param percent: The percent to kill (picks from the worst)
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        """
        self.name = "kill"
        self.percent = rates.make_callable(percent)
        self.now = 0

    def run(self, individuals, environment):
        data = individuals[:int(len(individuals) * (1 - self.percent()))]
        return data


class DuplicateRandom:
    def __init__(self, num_duplicate, selection_function=randomSelect.select):
        self.num_duplicate = rates.make_callable(num_duplicate)
        self.selection_function = selection_function

    def run(self, individuals, environment):
        # select some individuals to duplicate using the selection function
        duplicates = self.selection_function(individuals, self.num_duplicate())
        # append the duplicates to the population
        individuals = np.append(individuals, duplicates)
        return individuals


class RemoveDuplicatesFromTop:
    def __init__(self, top_n):
        self.top_n = rates.make_callable(top_n)

    def run(self, individuals, environment):
        # remove any duplicates from the top n individuals
        unique_individuals = individuals[:self.top_n()]
        for i in range(self.top_n, len(individuals)):
            if not any(NPCP.array_equal(ind.genes, individuals[i].genes) for ind in unique_individuals):
                unique_individuals.append(individuals[i])
        return unique_individuals


class SortByFitness:
    def __init__(self):
        pass

    def run(self, individuals, environment):
        sorted_individuals = sorted(individuals, key=lambda x: -x.fitness)

        return np.asarray(sorted_individuals)


class CapPopulation:
    def __init__(self, max_population):
        self.max_population = rates.make_callable(max_population)

    def run(self, individuals, environment):
        return individuals[0:self.max_population()]  # kills only the worst ones assuming they are sorted


class OverPoweredMutation(MutateAmount):
    def __init__(self, amount_individuals, amount_genes, gene_pool, tries=1,
                 selection_function=randomSelect.select):
        super().__init__(amount_individuals, amount_genes, gene_pool, False)
        self.tries = rates.make_callable(tries)
        self.selection_function = selection_function

    def run(self, individuals, environment):
        selected_individuals = self.selection_function(individuals, self.amount_individuals())
        for individual in selected_individuals:
            for i in range(self.tries()):
                copied = individual.copy()  # Use a custom copy method instead of deepcopy
                super().mutate_one(copied)
                copied.fit()
                if copied.fitness > individual.fitness:
                    individual.genes = copied.genes  # Assign the genes directly without deepcopy
        return individuals


class Function:
    def __init__(self, function):
        self.function = function

    def run(self, individuals, environment):
        return self.function(individuals)


class Controller:
    def __init__(self, layer, execute_every=1, repeat=1, delay=0, stop_at=math.inf):
        self.layer = layer
        self.every = rates.make_callable(execute_every)
        self.delay = rates.make_callable(delay)
        self.end = stop_at
        self.repeat = rates.make_callable(repeat)
        self.n = 0

    def run(self, individuals, environment):
        self.n += 1
        if self.delay() <= self.n <= self.end:
            if self.every() % self.n == 0:
                for i in range(self.repeat()):
                    individuals = self.layer.run(individuals, environment)
        return individuals


class FloatMutateAmount(MutateAmount):
    def __init__(self, amount_individuals, amount_genes, gene_pool, max_negative_mutation=-0.1,
                 max_positive_mutation=0.1
                 , refit=True, selection_function=randomSelect.select):
        super().__init__(amount_individuals, amount_genes, gene_pool, refit, selection_function)
        self.max_negative_mutation = rates.make_callable(max_negative_mutation)
        self.max_positive_mutation = rates.make_callable(max_positive_mutation)
        self.amount_individuals = rates.make_callable(amount_individuals)
    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes(), replace=False)
        mutation = NPCP.random.uniform(self.max_negative_mutation(), self.max_positive_mutation(),
                                       size=self.amount_genes())
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    def run(self, individuals, environment):
        self.amount_individuals = min(len(individuals), self.amount_individuals())
        selected_individuals = self.selection_function(individuals, self.amount_individuals())
        for individual in selected_individuals:
            individual = self.mutate_one(individual)
            if self.refit:
                individual.fit()
        return individuals


class IntMutateAmount(MutateAmount):
    def __init__(self, amount_individuals, amount_genes, gene_pool, min_mutation=-1, max_mutation=1
                 , refit=True, selection_function=randomSelect.select):
        super().__init__(rates.make_callable(amount_individuals), amount_genes, gene_pool, refit, selection_function)
        self.min_mutation = rates.make_callable(min_mutation)
        self.max_mutation = rates.make_callable(max_mutation)

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes(), replace=False)
        mutation = NPCP.random.randint(self.min_mutation(), self.max_mutation() + 1, size=self.amount_genes())
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    def run(self, individuals, environment):
        self.amount_individuals = min(len(individuals), self.amount_individuals())
        selected_individuals = self.selection_function(individuals, self.amount_individuals())
        for individual in selected_individuals:
            individual = self.mutate_one(individual)
            if self.refit:
                individual.fit()
        return individuals


class IntOverPoweredMutation(OverPoweredMutation):
    def __init__(self, amount_individuals, amount_genes, gene_pool, tries=1,
                 selection_function=randomSelect.select):
        super().__init__(amount_individuals, amount_genes, gene_pool, tries, selection_function)

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes, replace=False)
        mutation = NPCP.random.randint(self.min_mutation(), self.max_mutation() + 1, size=self.amount_genes())
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    def run(self, individuals, environment):
        selected_individuals = self.selection_function(individuals, self.amount_individuals())
        for individual in selected_individuals:
            for i in range(self.tries()):
                copied = individual.copy()
                copied = self.mutate_one(copied)
                copied.fit()
                if copied.fitness > individual.fitness:
                    individual.genes = copied.genes
                    individual.fit()
        return individuals


class FloatOverPoweredMutation(OverPoweredMutation):
    def __init__(self, amount_individuals, amount_genes, gene_pool, max_negative_mutation=-0.1,
                 max_positive_mutation=0.1
                 , tries=1, selection_function=randomSelect.select):
        super().__init__(amount_individuals, amount_genes, gene_pool, tries, selection_function)
        self.max_negative_mutation = rates.make_callable(max_negative_mutation)
        self.max_positive_mutation = rates.make_callable(max_positive_mutation)

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes(), replace=False)
        mutation = NPCP.random.uniform(self.max_negative_mutation(), self.max_positive_mutation(),
                                       size=self.amount_genes())
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    def run(self, individuals, environment):
        selected_individuals = self.selection_function(individuals, self.amount_individuals())
        for individual in selected_individuals:
            for i in range(self.tries()):
                copied = individual.copy()
                copied = self.mutate_one(copied)
                copied.fit()
                if copied.fitness > individual.fitness:
                    individual.genes = copied.genes
                    individual.fit() #TODO: I feel like this should be = copied.fitness
                    break
        return individuals


class FloatMomentumMutation:
    def __init__(self, divider, amount_individuals, amount_genes, execute_every=1,
                 selection_function=randomSelect.select, reset_baseline=False):
        self.divider = rates.make_callable(divider)
        self.amount_individuals = rates.make_callable(amount_individuals)
        self.amount_genes = rates.make_callable(amount_genes)
        self.execute_every = rates.make_callable(execute_every)
        self.selection_function = selection_function
        self.reset_baseline = reset_baseline

    def run(self, individuals, environment):
        if not individuals:
            print("n")
            return individuals
        selected_individuals = self.selection_function(individuals, self.amount_individuals())
        if environment.iteration > 0:  # diff will be none until after
            for individual in selected_individuals:
                random_indices = NPCP.random.permutation(len(individual.genes) - 1)
                random_indices = random_indices[:self.amount_genes()]
                individual.genes[random_indices] += (environment.diff[random_indices] / self.divider())
                if self.reset_baseline:
                    environment.original = environment.individuals[0].genes
                individual.fit()
        return individuals


class ParentBestChild:
    def __init__(self, num_families, selection_function=randomSelect.select):
        self.parenting_object = parenting.BestChild(num_families, selection_function)

    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentSinglePointCrossover:
    def __init__(self, num_families, num_childrem, selection_function=randomSelect.select):
        self.parenting_object = parenting.SinglePointCrossover(num_families, selection_function,
                                                               num_childrem)

    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentUniformCrossover:
    def __init__(self, num_families, num_children, selection_function=randomSelect.select):
        self.parenting_object = parenting.UniformCrossover(num_families, selection_function, num_children)

    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentNPointCrossover:
    def __init__(self, num_families, selection_function=randomSelect.select, n=2):
        self.parenting_object = parenting.NPointCrossover(num_families, selection_function, n)

    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentUniformCrossoverMultiple:
    def __init__(self, num_families, num_children, selection_function=randomSelect.select):
        self.parenting_object = parenting.UniformCrossoverMultiple(num_families, selection_function, num_children)

    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class ParentByGeneSegmentation:
    def __init__(self, num_families, num_children, selection_function=randomSelect.select, gene_size=2):
        self.parenting_object = parenting.ParentByGeneSegmentation(num_families, selection_function, gene_size,
                                                                   num_children)

    def run(self, individuals, environment):
        individuals += self.parenting_object.parent(individuals=individuals,
                                                    environment=environment, layer=self)
        return individuals


class Parent:
    def __init__(self, num_families, num_children, selection_function=randomSelect.select):
        self.parenting_object = parenting.SinglePointCrossover(num_families, selection_function, num_children)

    def run(self, individuals, environment):
        new = self.parenting_object.parent(individuals, environment, self)
        individuals += new
        return individuals
