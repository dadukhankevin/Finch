import numpy as np
from Finch.universal import ARRAY_MANAGER
from Finch.generic import GenePool, Individual, Layer, make_callable
from typing import Callable, List, Union


class FloatPool(GenePool):
    def __init__(self, ranges: List[List[float]], length: int, fitness_function: Callable, device="cpu"):
        """
        A GenePool for creating individuals with float genes within specified ranges.
        :param ranges: List of [min, max] ranges for each gene
        :param length: Number of genes in each individual
        :param device: Where genes in Individuals should be kept 'gpu' or 'cpu'
        """
        super().__init__(generator_function=self.generate_float_array, fitness_function=fitness_function)
        self.ranges = np.array(ranges)
        self.length = length
        self.device = device

    def generate_float_array(self):
        if self.device == "cpu":
            genes = np.random.uniform(
                low=self.ranges[:, 0],
                high=self.ranges[:, 1],
                size=self.length
            )
        elif self.device == "gpu":
            genes = ARRAY_MANAGER.random.uniform(
                low=self.ranges[:, 0],
                high=self.ranges[:, 1],
                size=self.length
            )
        return Individual(item=genes, fitness_function=self.fitness_function)


class ParentBlendFloat(Layer):
    def __init__(self, selection_function: Callable, families: int, children: int, alpha: float = 0.5, device='cpu'):
        super().__init__(application_function=self.parent, selection_function=selection_function, repeat=families)
        self.children = make_callable(children)
        self.device = device
        self.alpha = alpha

    def parent(self, individuals):
        parent1, parent2 = individuals
        children = []

        for _ in range(self.children()):
            if self.device == "cpu":
                gamma = np.random.uniform(-self.alpha, 1 + self.alpha, size=len(parent1.item))
                child_genes = parent1.item + gamma * (parent2.item - parent1.item)
            elif self.device == "gpu":
                gamma = ARRAY_MANAGER.random.uniform(-self.alpha, 1 + self.alpha, size=len(parent1.item))
                child_genes = parent1.item + gamma * (parent2.item - parent1.item)

            child = Individual(item=child_genes, fitness_function=parent1.fitness_function)
            children.append(child)

        self.environment.add_individuals(children)


class ParentSimulatedBinaryFloat(Layer):
    def __init__(self, selection_function: Callable, families: int, children: int, eta: float = 1.0, device='cpu'):
        super().__init__(application_function=self.parent, selection_function=selection_function, repeat=families)
        self.children = make_callable(children)
        self.device = device
        self.eta = eta

    def parent(self, individuals):
        parent1, parent2 = individuals
        children = []

        for _ in range(self.children()):
            if self.device == "cpu":
                u = np.random.random(len(parent1.item))
                beta = np.where(u <= 0.5,
                                (2 * u) ** (1 / (self.eta + 1)),
                                (1 / (2 * (1 - u))) ** (1 / (self.eta + 1)))
                child_genes = 0.5 * ((1 + beta) * parent1.item + (1 - beta) * parent2.item)
            elif self.device == "gpu":
                u = ARRAY_MANAGER.random.random(len(parent1.item))
                beta = ARRAY_MANAGER.where(u <= 0.5,
                                           (2 * u) ** (1 / (self.eta + 1)),
                                           (1 / (2 * (1 - u))) ** (1 / (self.eta + 1)))
                child_genes = 0.5 * ((1 + beta) * parent1.item + (1 - beta) * parent2.item)

            child = Individual(item=child_genes, fitness_function=parent1.fitness_function)
            children.append(child)

        self.environment.add_individuals(children)


class ParentArithmeticFloat(Layer):
    def __init__(self, selection_function: Callable, families: int, children: int, alpha: Union[float, str] = 'uniform',
                 device='cpu'):
        super().__init__(application_function=self.parent, selection_function=selection_function, repeat=families)
        self.children = make_callable(children)
        self.device = device
        self.alpha = alpha

    def parent(self, individuals):
        parent1, parent2 = individuals
        children = []

        for _ in range(self.children()):
            if self.alpha == 'uniform':
                if self.device == "cpu":
                    alpha = np.random.random()
                elif self.device == "gpu":
                    alpha = float(ARRAY_MANAGER.random.random())
            else:
                alpha = self.alpha

            child_genes = alpha * parent1.item + (1 - alpha) * parent2.item

            child = Individual(item=child_genes, fitness_function=parent1.fitness_function)
            children.append(child)

        self.environment.add_individuals(children)


class GaussianMutation(Layer):
    def __init__(self, mutation_rate: float, sigma: float, selection_function: Callable, device: str = 'cpu',
                 overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection_function)
        self.mutation_rate = mutation_rate
        self.sigma = sigma
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
            mutation = np.random.normal(0, self.sigma, individual.item.shape)
            individual.item = np.where(mask, individual.item + mutation, individual.item)
        elif self.device == "gpu":
            mask = ARRAY_MANAGER.random.random(individual.item.shape) < self.mutation_rate
            mutation = ARRAY_MANAGER.random.normal(0, self.sigma, individual.item.shape)
            individual.item = ARRAY_MANAGER.where(mask, individual.item + mutation, individual.item)

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < original_fitness:
                individual.item = original_item
                individual.fitness = original_fitness

        return individual


class UniformMutation(Layer):
    def __init__(self, mutation_rate: float, lower_bound: float, upper_bound: float, selection_function: Callable,
                 device: str = 'cpu', overpowered: bool = False, refit=True):
        super().__init__(application_function=self.mutate_all, selection_function=selection_function, refit=refit)
        self.mutation_rate = mutation_rate
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
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
            mutation = np.random.uniform(self.lower_bound, self.upper_bound, individual.item.shape)
            individual.item = np.where(mask, mutation, individual.item)
        elif self.device == "gpu":
            mask = ARRAY_MANAGER.random.random(individual.item.shape) < self.mutation_rate
            mutation = ARRAY_MANAGER.random.uniform(self.lower_bound, self.upper_bound, individual.item.shape)
            individual.item = ARRAY_MANAGER.where(mask, mutation, individual.item)

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < original_fitness:
                individual.item = original_item
                individual.fitness = original_fitness

        return individual


class PolynomialMutation(Layer):
    def __init__(self, mutation_rate: float, eta: float, bounds: List[List[float]], selection_function: Callable,
                 device: str = 'cpu', overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection_function)
        self.mutation_rate = mutation_rate
        self.eta = eta
        self.bounds = np.array(bounds)
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
            u = np.random.random(individual.item.shape)
            delta = np.where(
                u < 0.5,
                (2 * u) ** (1 / (self.eta + 1)) - 1,
                1 - (2 * (1 - u)) ** (1 / (self.eta + 1))
            )
            lower, upper = self.bounds[:, 0], self.bounds[:, 1]
            mutation = individual.item + delta * (upper - lower)
            individual.item = np.where(mask, np.clip(mutation, lower, upper), individual.item)
        elif self.device == "gpu":
            mask = ARRAY_MANAGER.random.random(individual.item.shape) < self.mutation_rate
            u = ARRAY_MANAGER.random.random(individual.item.shape)
            delta = ARRAY_MANAGER.where(
                u < 0.5,
                (2 * u) ** (1 / (self.eta + 1)) - 1,
                1 - (2 * (1 - u)) ** (1 / (self.eta + 1))
            )
            lower, upper = self.bounds[:, 0], self.bounds[:, 1]
            mutation = individual.item + delta * (upper - lower)
            individual.item = ARRAY_MANAGER.where(mask, ARRAY_MANAGER.clip(mutation, lower, upper), individual.item)

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < original_fitness:
                individual.item = original_item
                individual.fitness = original_fitness

        return individual


class InsertionDeletionMutationFloat(Layer):
    def __init__(self, selection_function: Callable, device: str = 'cpu', overpowered: bool = False):
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
            remove_idx = int(ARRAY_MANAGER.random.randint(0, len(individual.item)))
            # Select a random position to insert (can be the same as remove_idx)
            insert_idx = int(ARRAY_MANAGER.random.randint(0, len(individual.item)))

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

