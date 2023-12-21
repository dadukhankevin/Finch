import numpy as np
from Finch.genetics import Individual
from Finch.layers import layer
import random

cp = None
try:
    import cupy as cp
except ImportError:
    pass


def crossover(parent1: Individual, parent2: Individual) -> list:
    """
    Perform single-point crossover between two parents.

    Args:
        parent1 (Individual): The first parent.
        parent2 (Individual): The second parent.

    Returns:
        list of Individual: Two offspring resulting from the crossover.
    """
    average_fitness = (parent1.fitness + parent2.fitness) / 2
    point = random.randint(1, parent1.genes.size - 1)
    array_manager = cp if parent1.device == 'gpu' else np

    offspring1 = Individual(genes=array_manager.append(parent1.genes[:point], parent2.genes[point:]),
                            fitness=average_fitness,
                            device=parent1.device, gene_pool=parent1.gene_pool)

    offspring2 = Individual(genes=array_manager.append(parent2.genes[:point], parent1.genes[point:]),
                            fitness=average_fitness,
                            device=parent1.device, gene_pool=parent1.gene_pool)

    return [offspring1, offspring2]


def uniform_crossover(parent1: Individual, parent2: Individual, probability=0.5) -> list:
    """
    Perform uniform crossover between two parents.

    Args:
        parent1 (Individual): The first parent.
        parent2 (Individual): The second parent.
        probability (float): The probability of selecting a gene from the first parent.

    Returns:
        list of Individual: Two offspring resulting from the crossover.
    """
    average_fitness = (parent1.fitness + parent2.fitness) / 2
    array_manager = cp if parent1.device == 'gpu' else np

    mask = array_manager.random.uniform(0, 1, size=parent1.genes.size) < probability

    genes1 = array_manager.where(mask, parent1.genes, parent2.genes)
    genes2 = array_manager.where(mask, parent2.genes, parent1.genes)

    offspring1 = Individual(genes=genes1,
                            fitness=average_fitness,
                            device=parent1.device, gene_pool=parent1.gene_pool)

    offspring2 = Individual(genes=genes2,
                            fitness=average_fitness,
                            device=parent1.device, gene_pool=parent1.gene_pool)

    return [offspring1, offspring2]


def n_point_crossover(parent1: Individual, parent2: Individual, n=1) -> list:
    """
    Perform n-point crossover between two parents.

    Args:
        parent1 (Individual): The first parent.
        parent2 (Individual): The second parent.
        n (int): Number of crossover points.

    Returns:
        list of Individual: Offspring resulting from the crossover.
    """
    average_fitness = (parent1.fitness + parent2.fitness) / 2
    gene_length = parent1.genes.size
    array_manager = cp if parent1.device == 'gpu' else np

    points = sorted(random.sample(range(1, gene_length), n))
    points.append(gene_length)

    mask = array_manager.zeros(gene_length, dtype=bool)
    current_parent = 1  # Start with parent 1

    for point in points:
        mask[point:] = ~mask[point:]
        current_parent = 3 - current_parent  # Switch parent

    genes1 = array_manager.where(mask, parent1.genes, parent2.genes)

    offspring = Individual(genes=genes1,
                           fitness=average_fitness,
                           device=parent1.device, gene_pool=parent1.gene_pool)

    return [offspring]


class ParentSimple(layer.Parent):
    def __init__(self, families, children=2, refit=True):
        """
        Initialize a simple parent with crossover.

        Args:
            families: Selection method for individuals.
            children (int): Number of children to generate.
            refit (bool): If true will retest fitness in new children
        """
        if type(children) == int:
            assert children > 1, f"This type of crossover must return more than 1 child, you specified '{children}'."
        super().__init__(families=families,  children=children, refit=refit)

    def parent(self, parent1: Individual, parent2: Individual, environment) -> list:
        """
        Generate children through crossover.

        Args:
            parent1 (Individual): The first parent.
            parent2 (Individual): The second parent.
            environment: Additional environment information.

        Returns:
            list of Individual: Offspring resulting from the crossover.
        """
        return crossover(parent1=parent1, parent2=parent2)


class ParentNPoint(layer.Parent):
    def __init__(self, families, points=1, children=2, refit=True, track_genealogies = False):
        """
        Initialize an n-point parent with crossover.

        Args:
            families: Selection method for individuals.
            points (int): Number of crossover points.
            children (int): Number of children to generate.
            refit (bool): If true will retest fitness in new children
            track_genealogies (bool): If true will have individuals remember their parents
        """
        self.points = points
        super().__init__(families=families,  children=children, refit=refit, track_genealogies=track_genealogies)

    def parent(self, parent1: Individual, parent2: Individual, environment) -> list:
        """
        Generate children through n-point crossover.

        Args:
            parent1 (Individual): The first parent.
            parent2 (Individual): The second parent.
            environment: Additional environment information.

        Returns:
            list of Individual: Offspring resulting from the crossover.
        """

        children =  n_point_crossover(parent1=parent1, parent2=parent2, n=self.points)
        return children
