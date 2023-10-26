import torch
import tensorflow as tf
from Finch.genetics.population import Individual
from Finch.genetics.population import NPCP as np
import numpy
from Finch.ml.llm import LLM
from keras.layers import Layer

from keras.models import Model


def get_model_weights_as_array(model):
    weights = np.array([])
    for layer in model.layers:
        if isinstance(layer, Model):  # Handle nested models
            weights += np.append(weights, np.asarray(get_model_weights_as_array(layer)))
        else:
            weights = np.append(weights, np.asarray(layer.get_weights()))
    flattened_weights = np.concatenate([w.flatten() for w in weights], axis=0)
    print('past this part')
    return flattened_weights


def set_model_weights_from_array(model, weights_array, index=0):
    for layer in model.layers:
        if isinstance(layer, Model):  # Handle nested models
            index = set_model_weights_from_array(layer, weights_array, index)[1]
        else:
            layer_weights = layer.get_weights()
            new_weights = []

            for w in layer_weights:
                shape = w.shape
                size = np.prod(shape)
                new_weight = weights_array[index:index + size].reshape(shape)
                new_weights.append(new_weight)
                index += size

            layer.set_weights(new_weights)

    return model, index


class KerasPool:
    def __init__(self, model, fitness_function):
        self.model = model
        self.fitness_function = fitness_function
        self.length = len(get_model_weights_as_array(model))

    def generate(self):
        genes = self.generate_genes(1)
        r = Individual(genes, self.fitness_function_wrapped)
        return r

    def fitness_function_wrapped(self, individual):
        self.model = set_model_weights_from_array(self.model, individual)[0]
        return self.fitness_function(self.model)

    def randomize_model_weights(self, model):
        flat_weights = []
        total_weights = 0

        for layer in model.layers:
            if isinstance(layer, Model):
                new_flat_weights, num_weights = self.randomize_model_weights(layer)  # Handle nested models
                flat_weights.extend(new_flat_weights)
                total_weights += num_weights
            elif isinstance(layer, Layer):
                layer_weights = layer.get_weights()
                new_weights = [np.random.randn(*w.shape) for w in layer_weights]
                layer.set_weights(new_weights)
                flat_weights.extend([w.flatten() for w in new_weights])
                if flat_weights:
                    total_weights += len(flat_weights[-1])

        return np.concatenate(flat_weights)
    def generate_genes(self, num_genes):
        return self.randomize_model_weights(self.model)


class PyTorchPool:
    def __init__(self, model, fitness_function):
        self.model = model
        self.fitness_function = fitness_function
        self.length = self.flatten_model(model).size

    def generate(self):
        genes = self.generate_genes(1)
        return Individual(genes, self.fitness_function_wrapped)

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
        return Individual(genes, self.fitness_function_wrapped)

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


class PromptPool:
    def __init__(self, llm: LLM, fitness_funtion, temperature=.8, instructions="You are an expert in generating "
                                                                               "prompts for AI models.",
                 user_message="generate a prompt"):
        self.llm = llm
        self.message = user_message
        self.instructions = instructions
        self.llm.temperature = temperature
        self.temperature = temperature
        self.fitness_function = fitness_funtion

    def generate(self):
        self.llm.system_prompt = self.instructions  # ensure it always stays the same in this context
        self.llm.temperature = self.temperature
        return Individual(self.llm.run(self.message), fitness_function=self.fitness_function, as_array=False)
