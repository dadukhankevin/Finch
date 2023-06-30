from Finch.genetics.population import Individual
from Finch.genetics.population import NPCP as np

class Parent:
    def __init__(self, num_children, num_families, selection_function):
        self.num_children = num_children
        self.num_families = num_families
        self.selection_function = selection_function

    def parent(self, individuals, environment, layer):
        parents = self.selection_function.select(individuals)

class BestChild(Parent):
    def __init__(self, num_families, selection_function):
        super().__init__(num_children=1, num_families=num_families, selection_function=selection_function)

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
        return Individual(new_genes, parent1.fitness_function)
    def crossover(self, parent1, parent2, environment):