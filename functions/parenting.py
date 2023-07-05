from Finch.genetics.population import Individual
from Finch.genetics.population import NPCP as np
import random


class Parent:
    def __init__(self, num_children, num_families, selection_function, parent_function):
        self.num_children = num_children
        self.num_families = num_families
        self.selection_function = selection_function
        self.parent_function = parent_function

    def parent(self, individuals, environment, layer):
        kids = []
        for i in range(self.num_families):
            parents = self.selection_function(individuals, 2)
            for j in range(self.num_children):
                kids += self.parent_function(parents[0], parents[1], environment, layer)
        for kid in kids:
            kid.fit()
        return kids


class BestChild(Parent):
    def __init__(self, num_families, selection_function):
        super().__init__(num_children=1, num_families=num_families, selection_function=selection_function,
                         parent_function=self.crossover)

    def best_child(self, parent1, parent2, baseline):
        """
        Takes the most mutated (and therefore best) genes from each parent to create a new individual.
        Novel to Finch
        :param parent1:
        :param parent2:
        :param baseline: The individual without any mutations, like a first generation that was not randomly initialized
        :return:
        """
        parent1_genes = np.array(parent1.genes)
        parent2_genes = np.array(parent2.genes)
        new_genes = np.zeros_like(parent1_genes)  # create an array of zeros with the same shape as parent1_genes
        # find the 50% indices of parent1_genes that are the most different from the baseline
        diff1 = np.abs(parent1_genes - baseline)  # calculate the absolute difference between parent1_genes and baseline
        indices1 = np.argsort(diff1)[-len(diff1) // 2:]  # get the indices of the 50% largest differences
        # do the same thing for parent2_genes but make sure NONE of the indices are the same as the ones selected from parent1
        diff2 = np.abs(parent2_genes - baseline)  # calculate the absolute difference between parent2_genes and baseline
        mask = np.ones_like(diff2, dtype=bool)  # create a boolean mask of ones with the same shape as diff2
        mask[indices1] = False  # set the mask to False for the indices selected from parent1
        indices2 = np.argsort(diff2[mask])[
                   -len(diff2) // 4:]  # get the indices of the 50% largest differences among the remaining ones
        # Place the values of the genes into the new_genes using these indices
        new_genes[indices1] = parent1_genes[indices1]  # copy the genes from parent1 using indices1
        new_genes[indices2] = parent2_genes[indices2]  # copy the genes from parent2 using indices2
        return [Individual(new_genes, parent1.fitness_function)]

    def crossover(self, parent1, parent2, environment, layer):
        return self.best_child(parent1, parent2, environment.original)


class SinglePointCrossover(Parent):
    def __init__(self, num_families, selection_function, num_children):
        super().__init__(num_children=num_children, num_families=num_families, selection_function=selection_function,
                         parent_function=self.crossover)

    def crossover(self, parent1, parent2, environment, layer):
        point = random.randint(1, len(parent1.genes) - 1)
        offspring1 = Individual(np.append(parent1.genes[:point], parent2.genes[point:]), parent1.fitness_function)
        offspring2 = Individual(np.append(parent2.genes[:point], parent1.genes[point:]), parent2.fitness_function)

        return [offspring1, offspring2]


class UniformCrossover(Parent):
    def __init__(self, num_families, selection_function, num_children, probability=0.5):
        super().__init__(num_children=num_children, num_families=num_families, selection_function=selection_function,
                         parent_function=self.crossover)
        self.probability = probability

    def crossover(self, parent1, parent2, environment, layer):
        genes1 = []
        genes2 = []

        for gene1, gene2 in zip(parent1.genes, parent2.genes):
            if random.random() < self.probability:
                genes1.append(gene2)
                genes2.append(gene1)
            else:
                genes1.append(gene1)
                genes2.append(gene2)

        offspring1 = Individual(genes1, parent1.fitness_function)
        offspring2 = Individual(genes2, parent2.fitness_function)
        return offspring1, offspring2


class NPointCrossover(Parent):
    def __init__(self, num_families, selection_function, num_children, n):
        super().__init__(num_children=num_children, num_families=num_families, selection_function=selection_function,
                         parent_function=self.crossover)
        self.n = n

    def crossover(self, parent1, parent2, environment, layer):
        """
         Perform n-point crossover between two parents.
         Args:
             parent1 (Individual): The first parent.
             parent2 (Individual): The second parent.
             n (int): The number of crossover points.
         Returns: tuple: A tuple containing the offspring generated from the crossover.
         """
        points = sorted(random.sample(range(1, len(parent1.genes)), self.n))
        offspring1 = parent1
        offspring2 = parent2

        for i in range(0, len(points), 2):
            start = points[i]
            end = points[i + 1] if i + 1 < len(points) else len(parent1.genes)
            offspring1.genes[start:end], offspring2.genes[start:end] = offspring2.genes[start:end], offspring1.genes[
                                                                                                    start:end]

        return offspring1, offspring2


class UniformCrossoverMultiple(Parent):
    def __init__(self, num_families, selection_function, num_children):
        super().__init__(num_children=num_children, num_families=num_families, selection_function=selection_function,
                         parent_function=self.crossover)

    def crossover(self, parent1, parent2, environment, layer):
        parents = [parent1, parent2]
        offspring_genes = []
        num_genes = len(parents[0].genes)

        for gene_index in range(num_genes):
            selected_parent = random.choice(parents)
            offspring_genes.append(selected_parent.genes[gene_index])

        offspring = Individual(offspring_genes, parents[0].fitness_function)
        return offspring


class ParentByGeneSegmentation(Parent):
    def __init__(self, num_families, selection_function, num_children, gene_size=2):
        super().__init__(num_children=num_children, num_families=num_families, selection_function=selection_function,
                         parent_function=self.crossover)
        self.gene_size = gene_size

    def crossover(self, parent1, parent2, environment, layer):
        min_len = min(len(parent1.genes), len(parent2.genes))
        num_genes = min_len - (min_len % self.gene_size)

        segments = []

        # Select gene segments from parent1 and parent2
        for i in range(0, num_genes, self.gene_size):
            if random.random() < 0.5:
                segments.append(parent1.genes[i:i + self.gene_size])
            else:
                segments.append(parent2.genes[i:i + self.gene_size])

        # Concatenate gene segments to form offspring
        child = np.concatenate(segments)

        return Individual(child, parent1.fitness_function)
