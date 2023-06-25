from Finch.genetics.population import *


class DefaultPool:
    def __init__(self, valid_genes: np.ndarray, length: int, fitness_function):
        self.valid_genes = valid_genes
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        return Individual(NPCP.random.choice(self.valid_genes, size=self.length), self.fitness_function)

    def generate_genes(self, num_genes):
        return NPCP.random.choice(self.valid_genes, size=num_genes)


class FloatPool:
    def __init__(self, minimum_float, maximum_float, length: int, fitness_function):
        self.minimum_gene = minimum_float
        self.maximum_gene = maximum_float
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        genes = NPCP.random.uniform(self.minimum_gene, self.maximum_gene, size=self.length)
        return Individual(genes, self.fitness_function)

    def generate_genes(self, num_genes):
        return NPCP.random.uniform(self.minimum_gene, self.maximum_gene, size=num_genes)


class IntPool:
    def __init__(self, minimum_int, maximum_int, length: int, fitness_function):
        self.minimum_gene = minimum_int
        self.maximum_gene = maximum_int
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        genes = NPCP.random.randint(self.minimum_gene, self.maximum_gene + 1, size=self.length)
        return Individual(genes, self.fitness_function)

    def generate_genes(self, num_genes):
        return NPCP.random.randint(self.minimum_gene, self.maximum_gene + 1, size=num_genes)


class BinaryPool:
    def __init__(self, length: int, fitness_function):
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        genes = NPCP.random.randint(0, 2, size=self.length)
        return Individual(genes, self.fitness_function)

    def generate_genes(self, num_genes):
        return NPCP.random.randint(0, 2, size=(num_genes, self.length))


class StringPool:
    def __init__(self, valid_characters: str, length: int, fitness_function):
        self.valid_characters = valid_characters
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        genes = np.random.choice(list(self.valid_characters), size=self.length)
        return Individual(genes, self.fitness_function)

    def generate_genes(self, num_genes):
        genes = np.random.choice(list(self.valid_characters), size=(num_genes, self.length))
        return ["".join(gene) for gene in genes]


class PermutationPool:
    def __init__(self, valid_genes: np.ndarray, length: int, fitness_function):
        self.valid_genes = valid_genes
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        genes = NPCP.random.permutation(self.valid_genes)[:self.length]
        return Individual(genes, self.fitness_function)

    def generate_genes(self, num_genes):
        genes = NPCP.random.permutation(self.valid_genes)[:num_genes * self.length]
        return NPCP.split(genes, num_genes)


# Below are ML related gene pools, which have not been properly tested yet
