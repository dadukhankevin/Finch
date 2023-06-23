import random
from finch.genetics import *


def single_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1.genes) - 1)
    offspring1 = Individual(parent1.genes[:point] + parent2.genes[point:], parent1.fitness_function)
    offspring2 = Individual(parent2.genes[:point] + parent1.genes[point:], parent2.fitness_function)
    return offspring1, offspring2


def uniform_crossover(parent1, parent2, probability=0.5):
    genes1 = []
    genes2 = []

    for gene1, gene2 in zip(parent1.genes, parent2.genes):
        if random.random() < probability:
            genes1.append(gene2)
            genes2.append(gene1)
        else:
            genes1.append(gene1)
            genes2.append(gene2)

    offspring1 = Individual(genes1, parent1.fitness_function)
    offspring2 = Individual(genes2, parent2.fitness_function)
    return offspring1, offspring2


def n_point_crossover(parent1, parent2, n):
    """
         Perform n-point crossover between two parents.
         Args:
             parent1 (Individual): The first parent.
             parent2 (Individual): The second parent.
             n (int): The number of crossover points.
         Returns: tuple: A tuple containing the offspring generated from the crossover.
         """
    points = sorted(random.sample(range(1, len(parent1.genes)), n))
    offspring1 = parent1
    offspring2 = parent2

    for i in range(0, len(points), 2):
        start = points[i]
        end = points[i + 1] if i + 1 < len(points) else len(parent1.genes)
        offspring1.genes[start:end], offspring2.genes[start:end] = offspring2.genes[start:end], offspring1.genes[
                                                                                                start:end]

    return offspring1, offspring2


def uniform_crossover_multiple(parents):
    offspring_genes = []
    num_genes = len(parents[0].genes)

    for gene_index in range(num_genes):
        selected_parent = random.choice(parents)
        offspring_genes.append(selected_parent.genes[gene_index])

    offspring = Individual(offspring_genes, parents[0].fitness_function)
    return offspring


def parent_by_gene_segmentation(parent1, parent2, gene_size=2):
    min_len = min(len(parent1.genes), len(parent2.genes))
    num_genes = min_len - (min_len % gene_size)

    segments = []

    # Select gene segments from parent1 and parent2
    for i in range(0, num_genes, gene_size):
        if random.random() < 0.5:
            segments.append(parent1.genes[i:i + gene_size])
        else:
            segments.append(parent2.genes[i:i + gene_size])

    # Concatenate gene segments to form offspring
    child = np.concatenate(segments)

    return Individual(child, parent1.fitness_function)