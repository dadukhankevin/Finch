import numpy as np
import genetics
import torch
import tensorflow as tf


class DefaultPool:
    def __init__(self, valid_genes: np.ndarray, length: int, fitness_function):
        self.valid_genes = valid_genes
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        return genetics.Individual(np.random.choice(self.valid_genes, size=self.length), self.fitness_function)

    def generate_genes(self, num_genes):
        return np.random.choice(self.valid_genes, size=num_genes)


class FloatPool:
    def __init__(self, minimum_float, maximum_float, length: int, fitness_function):
        self.minimum_gene = minimum_float
        self.maximum_gene = maximum_float
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        genes = np.random.uniform(self.minimum_gene, self.maximum_gene, size=self.length)
        return genetics.Individual(genes, self.fitness_function)

    def generate_genes(self, num_genes):
        return np.random.uniform(self.minimum_gene, self.maximum_gene, size=num_genes)


class IntPool:
    def __init__(self, minimum_int, maximum_int, length: int, fitness_function):
        self.minimum_gene = minimum_int
        self.maximum_gene = maximum_int
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        genes = np.random.randint(self.minimum_gene, self.maximum_gene + 1, size=self.length)
        return genetics.Individual(genes, self.fitness_function)

    def generate_genes(self, num_genes):
        return np.random.randint(self.minimum_gene, self.maximum_gene + 1, size=num_genes)


class BinaryPool:
    def __init__(self, length: int, fitness_function):
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        genes = np.random.randint(0, 2, size=self.length)
        return genetics.Individual(genes, self.fitness_function)

    def generate_genes(self, num_genes):
        return np.random.randint(0, 2, size=(num_genes, self.length))


class StringPool:
    def __init__(self, valid_characters: str, length: int, fitness_function):
        self.valid_characters = valid_characters
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        genes = np.random.choice(list(self.valid_characters), size=self.length)
        return genetics.Individual(genes, self.fitness_function)

    def generate_genes(self, num_genes):
        genes = np.random.choice(list(self.valid_characters), size=(num_genes, self.length))
        return ["".join(gene) for gene in genes]


class PermutationPool:
    def __init__(self, valid_genes: np.ndarray, length: int, fitness_function):
        self.valid_genes = valid_genes
        self.length = length
        self.fitness_function = fitness_function

    def generate(self):
        genes = np.random.permutation(self.valid_genes)[:self.length]
        return genetics.Individual(genes, self.fitness_function)

    def generate_genes(self, num_genes):
        genes = np.random.permutation(self.valid_genes)[:num_genes * self.length]
        return np.split(genes, num_genes)


# Below are ML related gene pools, which have not been properly tested yet

class KerasPool:
    def __init__(self, model, fitness_function):
        self.model = model
        self.fitness_function = fitness_function
        self.length = self.flatten_model(model).size

    def generate(self):
        genes = self.generate_genes(1)
        return genetics.Individual(genes, self.fitness_function_wrapped)

    def fitness_function_wrapped(self, individual):
        model = self.unflatten_model(self.model, individual.genes)
        return self.fitness_function(model)

    def generate_genes(self, num_genes):
        flat_weights = np.random.uniform(-1, 1, size=(num_genes, self.length))
        return flat_weights

    def flatten_model(self, model):
        weights = model.get_weights()
        flat_weights = np.concatenate([w.flatten() for w in weights])
        return flat_weights

    def unflatten_model(self, model, flat_weights):
        shapes = [w.shape for w in model.get_weights()]
        sizes = [np.prod(s) for s in shapes]
        split_weights = np.split(flat_weights, np.cumsum(sizes)[:-1])
        reshaped_weights = [w.reshape(s) for w, s in zip(split_weights, shapes)]
        model.set_weights(reshaped_weights)
        return model


class PyTorchPool:
    def __init__(self, model, fitness_function):
        self.model = model
        self.fitness_function = fitness_function
        self.length = self.flatten_model(model).size

    def generate(self):
        genes = self.generate_genes(1)
        return genetics.Individual(genes, self.fitness_function_wrapped)

    def fitness_function_wrapped(self, individual):
        model = self.unflatten_model(self.model, individual.genes)
        return self.fitness_function(model)

    def generate_genes(self, num_genes):
        flat_weights = np.random.uniform(-1, 1, size=(num_genes, self.length))
        return flat_weights

    def flatten_model(self, model):
        weights = [param.data.flatten() for param in model.parameters()]
        flat_weights = torch.cat(weights)
        return flat_weights

    def unflatten_model(self, model, flat_weights):
        shapes = [param.data.shape for param in model.parameters()]
        sizes = [torch.tensor(s).prod().item() for s in shapes]
        split_weights = torch.split(flat_weights, sizes)
        reshaped_weights = [w.view(s) for w, s in zip(split_weights, shapes)]
        with torch.no_grad():
            for param, weight in zip(model.parameters(), reshaped_weights):
                param.data.copy_(weight)
        return model


class TensorFlowPool:
    def __init__(self, model, fitness_function):
        self.model = model
        self.fitness_function = fitness_function
        self.length = self.flatten_model(model).size

    def generate(self):
        genes = self.generate_genes(1)
        return genetics.Individual(genes, self.fitness_function_wrapped)

    def fitness_function_wrapped(self, individual):
        model = self.unflatten_model(self.model, individual.genes)
        return self.fitness_function(model)

    def generate_genes(self, num_genes):
        flat_weights = np.random.uniform(-1, 1, size=(num_genes, self.length))
        return flat_weights

    def flatten_model(self, model):
        weights = [tf.reshape(param, (-1,)) for param in model.trainable_variables]
        flat_weights = tf.concat(weights, axis=0)
        return flat_weights.numpy()

    def unflatten_model(self, model, flat_weights):
        shapes = [param.shape for param in model.trainable_variables]
        sizes = [tf.reduce_prod(s).numpy() for s in shapes]
        split_weights = tf.split(flat_weights, sizes)
        reshaped_weights = [tf.reshape(w, s) for w, s in zip(split_weights, shapes)]
        model.set_weights(reshaped_weights)
        return model
