import random

from Finch.universal import ARRAY_MANAGER
from Finch.generic import GenePool, Individual, Layer, make_callable
from typing import Callable, List, Any, Union
import numpy as np


class ArrayPool(GenePool):
    def __init__(self, gene_array: ARRAY_MANAGER.ndarray, fitness_function: Callable, length: int, device="cpu", unique=False):
        """
        A GenePool meant for the creation of individuals by picking items from an array.
        :param gene_array: The array from which genes will be picked.
        :param length: Amount of genes in each individual
        :param device: Where genes in Individuals should be kept 'gpu' or 'cpu'
        :param unique: Whether the generated genes should be unique
        """
        super().__init__(generator_function=self.generate_array, fitness_function=fitness_function)
        self.length = length
        self.device = device
        self.gene_array = gene_array
        self.unique = unique

    def generate_array(self):
        """
        :param amount: Amount of genes to generate
        :return: numpy or cupy array containing genes picked from the array
        """
        if self.device == "cpu":
            if self.unique:
                genes = np.random.choice(self.gene_array, size=self.length, replace=False)
            else:
                genes = np.random.choice(self.gene_array, size=self.length)

        elif self.device == "gpu":
            if self.unique:
                genes = ARRAY_MANAGER.random.choice(self.gene_array, size=self.length, replace=False)
            else:
                genes = ARRAY_MANAGER.random.choice(self.gene_array, size=self.length)
        ind = Individual(item=genes, fitness_function=self.fitness_function)

        return ind


class ParentNPoint(Layer):
    def __init__(self, selection_function: Callable, families: int, children: int, n_points: int = 3, device='cpu'):
        super().__init__(application_function=self.parent, selection_function=selection_function, repeat=families)
        self.children = make_callable(children)
        self.device = device
        self.n_points = n_points

    def parent(self, individuals):
        parent1, parent2 = individuals
        children = []

        for _ in range(self.children()):
            # Generate n random crossover points
            if self.device == "cpu":
                crossover_points = sorted(np.random.choice(len(parent1.item), size=self.n_points, replace=False))
                child_genes = np.zeros_like(parent1.item)
            elif self.device == "gpu":
                crossover_points = sorted(
                    ARRAY_MANAGER.random.choice(len(parent1.item), size=self.n_points, replace=False))
                child_genes = ARRAY_MANAGER.zeros_like(parent1.item)

            # Perform n-point crossover
            current_parent = parent1
            start = 0
            for point in crossover_points:
                child_genes[start:point] = current_parent.item[start:point]
                current_parent = parent2 if current_parent is parent1 else parent1
                start = point

            # Fill in the last segment
            child_genes[start:] = current_parent.item[start:]

            # Create new individual
            child = Individual(item=child_genes, fitness_function=parent1.fitness_function)
            children.append(child)

        self.environment.add_individuals(children)


class ParentUniform(Layer):
    def __init__(self, selection_function: Callable, families: int, children: int, crossover_rate: float = 0.5,
                 device='cpu'):
        super().__init__(application_function=self.parent, selection_function=selection_function, repeat=families)
        self.children = make_callable(children)
        self.device = device
        self.crossover_rate = crossover_rate

    def parent(self, individuals):
        parent1, parent2 = individuals
        children = []

        for _ in range(self.children()):
            if self.device == "cpu":
                mask = np.random.random(len(parent1.item)) < self.crossover_rate
                child_genes = np.where(mask, parent1.item, parent2.item)
            elif self.device == "gpu":
                mask = ARRAY_MANAGER.random.random(len(parent1.item)) < self.crossover_rate
                child_genes = ARRAY_MANAGER.where(mask, parent1.item, parent2.item)

            child = Individual(item=child_genes, fitness_function=parent1.fitness_function)
            children.append(child)

        self.environment.add_individuals(children)


class SwapMutation(Layer):
    def __init__(self, selection_function, device: str = 'cpu', overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection_function)
        self.device = device
        self.overpowered = overpowered

    def mutate_all(self, individuals: List[Individual]):
        for individual in individuals:
            self.mutate(individual)

    def mutate(self, individual: Individual) -> Individual:
        original_fitness = individual.fitness if self.overpowered else None
        original_item = individual.item.copy() if self.overpowered else None

        if self.device == "cpu":
            idx = np.random.choice(len(individual.item), size=2, replace=False)
            individual.item[idx[0]], individual.item[idx[1]] = individual.item[idx[1]], individual.item[idx[0]]
        elif self.device == "gpu":
            idx = ARRAY_MANAGER.random.choice(len(individual.item), size=2, replace=False)
            temp = individual.item[idx[0]].copy()
            individual.item[idx[0]] = individual.item[idx[1]]
            individual.item[idx[1]] = temp

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < original_fitness:
                individual.item = original_item
                individual.fitness = original_fitness

        return individual


class InversionMutation(Layer):
    def __init__(self, selection_function, device: str = 'cpu', overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection_function)
        self.device = device
        self.overpowered = overpowered

    def mutate_all(self, individuals: List[Individual]):
        for individual in individuals:
            self.mutate(individual)

    def mutate(self, individual: Individual) -> Individual:
        original_fitness = individual.fitness if self.overpowered else None
        original_item = individual.item.copy() if self.overpowered else None

        if self.device == "cpu":
            start, end = sorted(np.random.choice(len(individual.item), size=2, replace=False))
            individual.item[start:end] = individual.item[start:end][::-1]
        elif self.device == "gpu":
            start, end = sorted(ARRAY_MANAGER.random.choice(len(individual.item), size=2, replace=False))
            individual.item[start:end] = ARRAY_MANAGER.flip(individual.item[start:end])

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < original_fitness:
                individual.item = original_item
                individual.fitness = original_fitness

        return individual


class ScrambleMutation(Layer):
    def __init__(self, selection_function, device: str = 'cpu', overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection_function)
        self.device = device
        self.overpowered = overpowered

    def mutate_all(self, individuals: List[Individual]):
        for individual in individuals:
            self.mutate(individual)

    def mutate(self, individual: Individual) -> Individual:
        original_fitness = individual.fitness if self.overpowered else None
        original_item = individual.item.copy() if self.overpowered else None

        if self.device == "cpu":
            start, end = sorted(np.random.choice(len(individual.item), size=2, replace=False))
            subset = individual.item[start:end]
            np.random.shuffle(subset)
            individual.item[start:end] = subset
        elif self.device == "gpu":
            start, end = sorted(ARRAY_MANAGER.random.choice(len(individual.item), size=2, replace=False))
            subset = individual.item[start:end]
            ARRAY_MANAGER.random.shuffle(subset)
            individual.item[start:end] = subset

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < original_fitness:
                individual.item = original_item
                individual.fitness = original_fitness

        return individual


class ReplaceMutation(Layer):
    def __init__(self, mutation_rate: float, selection_function, possible_values: Union[List[Any], np.ndarray],
                 device: str = 'cpu', overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection_function)
        self.mutation_rate = mutation_rate
        self.possible_values = np.array(possible_values)  # Convert to numpy array
        self.device = device
        self.overpowered = overpowered

    def mutate_all(self, individuals: List[Individual]):
        for individual in individuals:
            self.mutate(individual)

    def mutate(self, individual: Individual) -> Individual:
        original_fitness = individual.fitness if self.overpowered else None
        original_item = individual.item.copy() if self.overpowered else None

        if self.device == "cpu":
            mask = np.random.random(individual.item.shape) < self.mutation_rate
            replacements = np.random.choice(self.possible_values, size=int(np.sum(mask)))
            individual.item[mask] = replacements
        elif self.device == "gpu":
            mask = ARRAY_MANAGER.random.random(individual.item.shape) < self.mutation_rate
            replacements = ARRAY_MANAGER.random.choice(self.possible_values, size=int(ARRAY_MANAGER.sum(mask)))
            individual.item[mask] = replacements

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < original_fitness:
                individual.item = original_item
                individual.fitness = original_fitness

        return individual


class InsertionDeletionMutation(Layer):
    def __init__(self, selection_function, device: str = 'cpu', overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection_function)
        self.device = device
        self.overpowered = overpowered

    def mutate_all(self, individuals: List[Individual]):
        for individual in individuals:
            self.mutate(individual)

    def mutate(self, individual: Individual) -> Individual:
        # If there's only one gene, don't perform the mutation
        if len(individual.item) <= 1:
            return individual

        original_fitness = individual.fitness if self.overpowered else None
        original_item = individual.item.copy() if self.overpowered else None

        if self.device == "cpu":
            # Select a random gene to remove
            remove_idx = np.random.randint(0, len(individual.item))
            # Select a random position to insert (can be the same as remove_idx)
            insert_idx = np.random.randint(0, len(individual.item))

            # Remove the gene and insert it at the new position
            gene = individual.item[remove_idx]
            individual.item = np.delete(individual.item, remove_idx)
            individual.item = np.insert(individual.item, insert_idx, gene)

        elif self.device == "gpu":
            # Select a random gene to remove
            remove_idx = ARRAY_MANAGER.random.randint(0, len(individual.item))
            # Select a random position to insert (can be the same as remove_idx)
            insert_idx = ARRAY_MANAGER.random.randint(0, len(individual.item))

            # Remove the gene and insert it at the new position
            gene = individual.item[remove_idx].copy()
            individual.item = ARRAY_MANAGER.delete(individual.item, remove_idx)
            individual.item = ARRAY_MANAGER.insert(individual.item, insert_idx, gene)

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < original_fitness:
                individual.item = original_item
                individual.fitness = original_fitness

        return individual