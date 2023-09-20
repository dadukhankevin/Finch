import torch
import tensorflow as tf
from Finch.genetics.population import Individual
from Finch.genetics.population import NPCP as np
from Finch.ml.llm import LLM


class KerasPool:
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
        self.llm.system_prompt = instructions
        self.llm.temperature = temperature
        self.fitness_function = fitness_funtion

    def generate(self):
        return Individual(self.llm.run(self.message), fitness_function=self.fitness_function, as_array=False)
