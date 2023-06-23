import copy
import math

import numpy as np
from finch import selection
from finch import crossover


class Populate:
    def __init__(self, gene_pool, population):
        self.gene_pool = gene_pool
        self.population = population

    def run(self, individuals):
        while len(individuals) < self.population:
            individuals = np.append(individuals, self.gene_pool.generate())
        return individuals


class MutateAmount:
    def __init__(self, amount_individuals, amount_genes, gene_pool, refit=True,
                 selection_function=selection.random_selection):

        self.amount_individuals = amount_individuals
        self.amount_genes = amount_genes
        self.gene_pool = gene_pool
        self.refit = refit
        self.selection_function = selection_function

    def run(self, individuals):
        self.amount_individuals = min(len(individuals), self.amount_individuals)
        selected_individuals = self.selection_function(individuals, self.amount_individuals)
        for individual in selected_individuals:
            individual = self.mutate_one(individual)
            if self.refit:
                individual.fit()
        return individuals

    def mutate_one(self, individual):
        random_indices = np.random.choice(len(individual.genes), self.amount_genes, replace=False)
        individual.genes[random_indices] = self.gene_pool.generate_genes(self.amount_genes)
        return individual


class Parent:
    def __init__(self, num_children, num_families, selection_function=selection.random_selection,
                 crossover_function=crossover.parent_by_gene_segmentation, gene_size=None):
        self.num_children = num_children
        self.num_families = num_families
        self.gene_size = gene_size  # TODO: figure out a better way to do this

        self.selection_function = selection_function
        self.crossover_function = crossover_function

    def run(self, individuals):
        to_parent = self.selection_function(individuals, self.num_families * 2)
        new_individuals = []
        for parent1, parent2 in zip(to_parent[::2], to_parent[1::2]):
            for i in range(self.num_children):
                if self.gene_size:
                    new_ind = self.crossover_function(parent1, parent2, self.gene_size)
                    new_ind.fit()
                    new_individuals.append(new_ind)
                else:
                    new_ind = self.crossover_function(parent1, parent2)
                    new_ind.fit()
                    new_individuals.append(new_ind)
        individuals = np.append(individuals, new_individuals)
        return individuals


class KillRandom:
    def __init__(self, num_kill, selection_function):
        self.num_kill = num_kill
        self.selection_function = selection_function

    def run(self, individuals):
        # select some individuals to kill using the selection function
        to_kill = self.selection_function(individuals, self.num_kill)
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
        self.percent = percent
        self.now = 0

    def run(self, data):
        data = data[:int(len(data) * (1 - self.percent))]
        return data


class DuplicateRandom:
    def __init__(self, num_duplicate, selection_function=selection.random_selection):
        self.num_duplicate = num_duplicate
        self.selection_function = selection_function

    def run(self, individuals):
        # select some individuals to duplicate using the selection function
        duplicates = self.selection_function(individuals, self.num_duplicate)
        # append the duplicates to the population
        individuals = np.append(individuals, duplicates)
        return individuals


class RemoveDuplicatesFromTop:
    def __init__(self, top_n):
        self.top_n = top_n

    def run(self, individuals):
        # remove any duplicates from the top n individuals
        unique_individuals = individuals[:self.top_n]
        for i in range(self.top_n, len(individuals)):
            if not any(np.array_equal(ind.genes, individuals[i].genes) for ind in unique_individuals):
                unique_individuals.append(individuals[i])
        return unique_individuals


class SortByFitness:
    def __init__(self):
        pass

    def run(self, individuals):
        sorted_indices = np.argsort([-individual.fitness for individual in individuals])
        sorted_individuals = individuals[sorted_indices]
        return sorted_individuals


class CapPopulation:
    def __init__(self, max_population):
        self.max_population = max_population

    def run(self, individuals):
        return individuals[0:self.max_population]  # kills only the worst ones assuming they are sorted


class OverPoweredMutation(MutateAmount):
    def __init__(self, amount_individuals, amount_genes, gene_pool, tries, selection_function=selection.random_selection):
        super().__init__(amount_individuals, amount_genes, gene_pool, False)
        self.tries = tries
        self.selection_function = selection_function

    def run(self, individuals):
        selected_individuals = self.selection_function(individuals, self.amount_individuals)
        for individual in selected_individuals:
            copied = individual.copy()  # Use a custom copy method instead of deepcopy
            super().mutate_one(copied)
            copied.fit()
            if copied.fitness > individual.fitness:
                individual.genes = copied.genes  # Assign the genes directly without deepcopy
        return individuals

