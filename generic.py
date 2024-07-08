from Finch.universal import ARRAY_MANAGER
import copy
from typing import Callable, Union, List, Dict
from matplotlib import pyplot as plt
import math

make_callable = lambda x: x if callable(x) else lambda: x


class Individual:
    def __init__(self, item, fitness_function):
        self.fitness = -math.inf
        self.item = item
        self.fitness_function = fitness_function
    def fit(self):
        self.fitness = self.fitness_function(self)
        return self.fitness
    def copy(self):
        return copy.deepcopy(self)


class Layer:
    def __init__(self, application_function: Callable, selection_function: Union[Callable, int], repeat: int = 1, refit=True):
        self.application_function = application_function
        self.selection_function = selection_function
        self.repeat = repeat
        self.environment = None
        self.refit = refit

    def set_environment(self, environment):
        self.environment = environment

    def execute(self, individuals: List[Individual]):
        assert self.environment, "Environment is not set, please compile the environment or call set_environment(...)"
        for i in range(self.repeat):
            selected = self.selection_function(individuals)
            self.application_function(selected)
            if self.refit:
                for individual in selected:
                    individual.fit()


class Environment:
    def __init__(self, layers: List[Layer], individuals, verbose_every=1, early_stopping=0):
        self.layers = layers
        self.individuals = individuals
        self.best_ever = None
        self.early_stopping = early_stopping

        self.history = {
            'fitness': [],
            'population': [],
        }
        self.verbose_every = verbose_every
    def add_layer(self, layer: Layer):
        self.layers.append(layer)
    def evolve(self, generations: int):
        for i in range(generations):
            for layer in self.layers:
                layer.execute(self.individuals)

            fitness = self.individuals[0].fitness

            if self.best_ever:
                if fitness > self.best_ever.fitness:
                    self.best_ever = self.individuals[0].copy()
            else:
                self.best_ever = self.individuals[0].copy()
            if fitness > self.best_ever.fitness:
                return
            self.history['fitness'].append(fitness)
            self.history['population'].append(len(self.individuals))
            if i % self.verbose_every == 0:
                print(f"Generation: {i} Fitness: {fitness} Population: {len(self.individuals)}")

    def add_individuals(self, individuals: List[Individual]):
        for individual in individuals:
            individual.environment = self
        self.individuals.extend(individuals)

    def compile(self):
        for layer in self.layers:
            layer.set_environment(self)

    def plot(self):
        plt.plot(self.history['fitness'])
        plt.legend(['fitness', 'population'])
        plt.show()

class GenePool:
    def __init__(self, generator_function, fitness_function: Callable):
        self.generator_function = generator_function
        self.fitness_function = fitness_function

    def generate_individuals(self, amount: int):
        return [self.generator_function() for _ in range(amount)]


class Competition:
    def __init__(self, environments: Dict[Environment, str], adaptive_mode: str = 'neither', verbose_every: int = 10):
        self.environments = environments
        self.adaptive_mode = adaptive_mode
        self.verbose_every = verbose_every
        self.history = {name: {'fitness': [], 'population': []} for name in environments.values()}

        if adaptive_mode not in ['best', 'worst', 'neither']:
            raise ValueError("adaptive_mode must be 'best', 'worst', or 'neither'")

    def evolve(self, total_generations: int):
        env_names = list(self.environments.values())
        env_count = len(env_names)

        for gen in range(total_generations):
            gen_allocation = self._allocate_generations(env_count) if self.adaptive_mode != 'neither' else {name: 1 for
                                                                                                            name in
                                                                                                            env_names}

            for env, name in self.environments.items():
                env.evolve(gen_allocation[name])
                fitness = env.individuals[0].fitness
                population = len(env.individuals)

                self.history[name]['fitness'].append(fitness)
                self.history[name]['population'].append(population)

            if gen % self.verbose_every == 0:
                best_fitness = max(env.individuals[0].fitness for env in self.environments)
                print(f"Generation {gen}: Best fitness = {best_fitness}")

    def _allocate_generations(self, env_count):
        performances = [(name, env.individuals[0].fitness) for env, name in self.environments.items()]
        performances.sort(key=lambda x: x[1], reverse=(self.adaptive_mode == 'best'))

        total_weight = sum(range(1, env_count + 1))
        return {
            name: max(1, int(((env_count - i if self.adaptive_mode == 'best' else i + 1) / total_weight) * env_count))
            for i, (name, _) in enumerate(performances)}

    def plot(self):
        plt.figure(figsize=(12, 6))
        for name, data in self.history.items():
            plt.plot(data['fitness'], label=f'{name} (Fitness)')
        plt.title('Fitness History Across Environments')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        for name, data in self.history.items():
            plt.plot(data['population'], label=f'{name} (Population)')
        plt.title('Population History Across Environments')
        plt.xlabel('Generation')
        plt.ylabel('Population Size')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_best_environment(self):
        return max(((env, name, env.best_ever.fitness) for env, name in self.environments.items()), key=lambda x: x[2])

    def get_worst_environment(self):
        return min(((env, name, env.best_ever.fitness) for env, name in self.environments.items()), key=lambda x: x[2])