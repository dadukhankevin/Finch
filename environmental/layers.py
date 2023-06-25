import math

import numpy as np

# Try importing CuPy


from Finch.functions import crossover, selection

from Finch.genetics.population import NPCP


class Populate:
    def __init__(self, gene_pool, population):
        self.gene_pool = gene_pool
        self.population = population

    def run(self, individuals, environment):
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

    def run(self, individuals, environment):
        self.amount_individuals = min(len(individuals), self.amount_individuals)
        selected_individuals = self.selection_function(individuals, self.amount_individuals)
        for individual in selected_individuals:
            individual = self.mutate_one(individual)
            if self.refit:
                individual.fit()
        return individuals

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(len(individual.genes), self.amount_genes, replace=False)
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

    def run(self, individuals, environment):
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

    def run(self, individuals, environment):
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

    def run(self, individuals, environment):
        data = individuals[:int(individuals.size * (1 - self.percent))]
        return data


class DuplicateRandom:
    def __init__(self, num_duplicate, selection_function=selection.random_selection):
        self.num_duplicate = num_duplicate
        self.selection_function = selection_function

    def run(self, individuals, environment):
        # select some individuals to duplicate using the selection function
        duplicates = self.selection_function(individuals, self.num_duplicate)
        # append the duplicates to the population
        individuals = np.append(individuals, duplicates)
        return individuals


class RemoveDuplicatesFromTop:
    def __init__(self, top_n):
        self.top_n = top_n

    def run(self, individuals, environment):
        # remove any duplicates from the top n individuals
        unique_individuals = individuals[:self.top_n]
        for i in range(self.top_n, len(individuals)):
            if not any(NPCP.array_equal(ind.genes, individuals[i].genes) for ind in unique_individuals):
                unique_individuals.append(individuals[i])
        return unique_individuals


class SortByFitness:
    def __init__(self):
        pass

    def run(self, individuals, environment):
        print(individuals.__class__)
        sorted_indices = np.argsort(np.asarray([-individual.fitness for individual in individuals]))
        sorted_individuals = individuals[sorted_indices]
        return sorted_individuals


class CapPopulation:
    def __init__(self, max_population):
        self.max_population = max_population

    def run(self, individuals, environment):
        return individuals[0:self.max_population]  # kills only the worst ones assuming they are sorted


class OverPoweredMutation(MutateAmount):
    def __init__(self, amount_individuals, amount_genes, gene_pool, tries=1,
                 selection_function=selection.random_selection):
        super().__init__(amount_individuals, amount_genes, gene_pool, False)
        self.tries = tries
        self.selection_function = selection_function

    def run(self, individuals, environment):
        selected_individuals = self.selection_function(individuals, self.amount_individuals)
        for individual in selected_individuals:
            for i in range(self.tries):
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
        self.every = execute_every
        self.delay = delay
        self.end = stop_at
        self.repeat = repeat
        self.n = 0

    def run(self, individuals, environment):
        self.n += 1
        if self.delay >= self.n and self.end >= self.n:
            if self.every % self.n == 0:
                for i in range(self.repeat):
                    individuals = self.layer.run(individuals)
        return individuals


class FloatMutateAmount(MutateAmount):
    def __init__(self, amount_individuals, amount_genes, gene_pool, max_negative_mutation=-0.1,
                 max_positive_mutation=0.1
                 , refit=True, selection_function=selection.random_selection):
        super().__init__(amount_individuals, amount_genes, gene_pool, refit, selection_function)
        self.max_negative_mutation = max_negative_mutation
        self.max_positive_mutation = max_positive_mutation

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes, replace=False)
        mutation = NPCP.random.uniform(self.max_negative_mutation, self.max_positive_mutation,
                                               size=self.amount_genes)
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    def run(self, individuals, environment):
        self.amount_individuals = min(individuals.size, self.amount_individuals)
        selected_individuals = self.selection_function(individuals, self.amount_individuals)
        for individual in selected_individuals:
            individual = self.mutate_one(individual)
            if self.refit:
                individual.fit()
        return individuals


class IntMutateAmount(MutateAmount):
    def __init__(self, amount_individuals, amount_genes, gene_pool, min_mutation=-1, max_mutation=1
                 , refit=True, selection_function=selection.random_selection):
        super().__init__(amount_individuals, amount_genes, gene_pool, refit, selection_function)
        self.min_mutation = min_mutation
        self.max_mutation = max_mutation

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes, replace=False)
        mutation = NPCP.random.randint(self.min_mutation, self.max_mutation + 1, size=self.amount_genes)
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    def run(self, individuals, environment):
        self.amount_individuals = min(individuals.size, self.amount_individuals)
        selected_individuals = self.selection_function(individuals, self.amount_individuals)
        for individual in selected_individuals:
            individual = self.mutate_one(individual)
            if self.refit:
                individual.fit()
        return individuals


class IntOverPoweredMutation(OverPoweredMutation):
    def __init__(self, amount_individuals, amount_genes, gene_pool, tries=1,
                 selection_function=selection.random_selection):
        super().__init__(amount_individuals, amount_genes, gene_pool, tries, selection_function)

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes, replace=False)
        mutation = NPCP.random.randint(self.min_mutation, self.max_mutation + 1, size=self.amount_genes)
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    def run(self, individuals, environment):
        selected_individuals = self.selection_function(individuals, self.amount_individuals)
        for individual in selected_individuals:
            for i in range(self.tries):
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
                 , tries=1, selection_function=selection.random_selection):
        super().__init__(amount_individuals, amount_genes, gene_pool, tries, selection_function)
        self.max_negative_mutation = max_negative_mutation
        self.max_positive_mutation = max_positive_mutation

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes, replace=False)
        mutation = NPCP.random.uniform(self.max_negative_mutation, self.max_positive_mutation, size=self.amount_genes)
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    def run(self, individuals, environment):
        selected_individuals = self.selection_function(individuals, self.amount_individuals)
        for individual in selected_individuals:
            for i in range(self.tries):
                copied = individual.copy()
                copied = self.mutate_one(copied)
                copied.fit()
                if copied.fitness > individual.fitness:
                    individual.genes = copied.genes
                    individual.fit()
        return individuals


class FloatMomentumMutation:
    def __init__(self, divider, amount_individuals, amount_genes, execute_every=1, based_on_probability=False,
                 selection_function=selection.rank_based_selection, selection_arg=1, calculate_custom_diff=False,
                 probability_power=1):
        self.divider = divider
        self.based_on_probability = based_on_probability
        self.amount_individuals = amount_individuals
        self.amount_genes = amount_genes
        self.execute_every = execute_every
        self.selection_function = selection_function
        self.calculate_custom_diff = calculate_custom_diff
        self.selection_arg = selection_arg
        self.probability_power = probability_power

    def run(self, individuals, environment):
        selected_individuals = self.selection_function(individuals, self.selection_arg)
        if environment.iteration > 0:  # diff will be none until after
            for individual in selected_individuals:
                if self.calculate_custom_diff:
                    if self.based_on_probability:
                        diff = (environment.original.genes - individual.genes)
                        p = NPCP.abs(diff) ** self.probability_power
                        p = p / NPCP.sum(p)
                        p = NPCP.nan_to_num(p)
                        # Normalize the probabilities by dividing them by their sum
                        # Use np.random.multinomial to get the random indices based on the probabilities
                        random_indices = NPCP.random.multinomial(len(individual.genes) - 1, p)
                        # Use np.unique to remove any duplicates from the random indices
                        random_indices = NPCP.unique(random_indices)
                        # Take the first amount_genes elements of the unique indices as the final indices
                        random_indices = random_indices[:self.amount_genes]
                        individual.genes[random_indices] += (diff[random_indices] / self.divider)
                    else:
                        diff = (environment.original.genes - individual.genes)
                        # Use np.random.permutation to get the random indices without probabilities
                        random_indices = NPCP.random.permutation(len(individual.genes) - 1)
                        # Take the first amount_genes elements of the random indices as the final indices
                        random_indices = random_indices[:self.amount_genes]
                        individual.genes[random_indices] += (diff[random_indices] / self.divider)
                else:
                    if self.based_on_probability:
                        p = NPCP.abs(environment.diff) ** self.probability_power
                        p = p / NPCP.sum(p)
                        p = NPCP.nan_to_num(p)
                        # Normalize the probabilities by dividing them by their sum
                        random_indices = NPCP.random.multinomial(len(individual.genes) - 1, p)
                        # Use np.unique to remove any duplicates from the random indices
                        random_indices = NPCP.unique(random_indices)
                        # Take the first amount_genes elements of the unique indices as the final indices
                        random_indices = random_indices[:self.amount_genes]
                        individual.genes[random_indices] += (environment.diff[random_indices] / self.divider)
                    else:
                        # Use np.random.permutation to get the random indices without probabilities
                        random_indices = NPCP.random.permutation(len(individual.genes) - 1)
                        # Take the first amount_genes elements of the random indices as the final indices
                        random_indices = random_indices[:self.amount_genes]
                        individual.genes[random_indices] += (environment.diff[random_indices] / self.divider)
            individual.fit()
        return individuals
