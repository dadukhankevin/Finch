import warnings

import matplotlib.pyplot as plt
from Finch.exceptions.environment_exceptions import NoIndividualsAtEndOfRun
import time
from tabulate import tabulate
from Finch.environmental.layers import Layer

from Finch.genetics.population import Individual


class Environment(Layer):
    def __init__(self, layers: list[Layer] = None, name="Environment", verbose_every=1):
        super().__init__()
        self.verbose_every = verbose_every
        self.callback = None
        self.name = name
        self.layers = layers
        self.individuals = []
        self.iteration = 0
        self.history = []
        self.compiled = False

    def compile(self, individuals: list[Individual] = None, callback: callable = None, verbose_every: int = 1):
        self.individuals = individuals
        self.compiled = True
        if self.individuals is None:
            self.individuals = []
        self.callback = callback
        self.verbose_every = verbose_every

    def evolve(self, generations):
        fitness = 0
        for i in range(generations):
            self.iteration = i
            if self.verbose_every and i % self.verbose_every == 0:
                print(
                    f"{self.name}: generation {i + 1}/{generations}. Max fitness: {fitness}. Population: "
                    f"{len(self.individuals)}")
            for layer in self.layers:
                self.individuals = layer.run(self.individuals, self)
            if self.callback:
                self.callback(self.individuals, self)
            if len(self.individuals) == 0:
                raise NoIndividualsAtEndOfRun("Your environment has a population of 0 after running.")
            fitness = self.individuals[-1].fitness
            self.history.append(fitness)
        return self.individuals, self.history

    def run(self, individuals: list[Individual], environment):
        warnings.warn("You are using this environment as a layer")
        individuals = []
        for layer in self.layers:
            individuals = layer.run(individuals, environment)
        return individuals


class Sequential(Environment):
    def __init__(self, layers: list[Layer], name="default"):
        super().__init__(layers, name)
        self.layers = layers
        self.original = None
        self.diff = None
        self.track_float_diff_every = False

    def compile(self, individuals: list[Individual] = None, callback: callable = None, verbose_every: int = 1):
        super().compile(individuals, callback, verbose_every)

    def evolve(self, generations: int):
        return super().evolve(generations)

    def reset(self):
        self.individuals = []
        self.history = []
        self.original = []
        self.diff = []
        self.iteration = 0


class Adversarial(Environment):
    def __init__(self, environments, name="Adversarial Environment"):
        super().__init__(environments, name)
        self.environments = environments

    def compile(self, individuals: list[Individual] = None, callback: callable = None, verbose_every: int = 1):
        super().compile(individuals, callback, verbose_every)
        for environment in self.environments:
            if not environment.compiled:
                environment.compile(verbose_every=verbose_every)

    def evolve(self, generations: int = 1):
        results = []
        results_history = []
        for env in self.environments:
            start_time = time.time()
            individuals, history = env.evolve(generations)
            end_time = time.time()
            evolution_time = end_time - start_time
            results.append((env.name, history[-1], evolution_time))
            results_history.append((env.name, history[-1], evolution_time, history))
            if self.callback:
                self.callback(individuals)
        table_headers = ["Environment", "Max Fitness", "Evolution Time (s)"]

        print(tabulate(results, headers=table_headers, tablefmt="pretty"))
        best_environment = max(results, key=lambda x: x[1])  # Choose the environment with the highest fitness
        print(f"Best environment: {best_environment[0]}, Max fitness: {best_environment[1]}")

        self.plot_fitness_histories(results_history)  # Call the plot function with results

        return best_environment

    def plot_fitness_histories(self, results):
        plt.figure(figsize=(10, 6))
        for env_name, max_fitness, _, history in results:
            plt.plot(history, label=env_name)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(self.name + ", Fitness History Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()


class ChronologicalEnvironment(Environment):
    def __init__(self, environments_and_generations, name="mixed_environment"):
        super().__init__(None, name)
        self.environments_and_generations = environments_and_generations
        self.combined_history = []
        self.best_environment = None
        self.individuals = []

    def evolve(self, generations=1):
        for env, generations_here in self.environments_and_generations:
            if generations_here is None:
                warnings.warn("Generally avoid using more than 1 generation in a ChronologicalEnvironment, instead "
                              "pass them in the init")
                generations_here = generations
            print(f"Running {env.name} for {generations_here} generations...")
            env.individuals = self.individuals
            individuals, history = env.evolve(generations_here)
            self.individuals.extend(individuals)
            self.combined_history.extend(history)
            if not self.best_environment or history[-1] > self.best_environment[1]:
                self.best_environment = (env.name, history[-1])
        return self.individuals, self.combined_history

    def compile(self, individuals: list[Individual] = None, callback: callable = None, verbose_every: int = 1):
        super().compile(individuals, callback, verbose_every)
        for environment, generations_here in self.environments_and_generations:
            if not environment.compiled:
                environment.compile(verbose_every=verbose_every)

    def plot(self):
        plt.figure(figsize=(10, 6))

        # Define a custom color palette with distinct colors
        custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        start_idx = 0

        for idx, (env, generations) in enumerate(self.environments_and_generations):
            end_idx = start_idx + generations
            history = self.combined_history[start_idx:end_idx]

            # Use modulo to cycle through custom colors
            color_idx = idx % len(custom_colors)

            # Create a line segment for this environment with a changing color
            plt.plot(range(start_idx + 1, end_idx + 1), history, label=f"{env.name}", color=custom_colors[color_idx])

            start_idx = end_idx

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Combined Fitness History")
        plt.legend()
        plt.grid(True)
        plt.show()

class AdaptiveEnvironment(Environment):
    def __init__(self, environments, switch_every=10, name="AdaptiveEnvironment"):
        super().__init__(name=name)
        self.environments = environments
        self.switch_every = switch_every
        self.current_environment = None
        self.current_generation = 0

    def compile(self, individuals: list[Individual] = None, callback: callable = None, verbose_every: int = 1):
        super().compile(individuals, callback, verbose_every)
        for environment in self.environments:
            if not environment.compiled:
                environment.compile(verbose_every=verbose_every)

    def evolve(self, generations: int):
        for i in range(generations):
            self.current_generation += 1
            if self.current_generation % self.switch_every == 0:
                self.switch_environment()
            if self.current_environment is not None:
                self.individuals, history = self.current_environment.evolve(1)
                self.history.extend(history)
                if self.callback:
                    self.callback(self.individuals)
        return self.individuals, self.history

    def switch_environment(self):
        max_fitness_increase = -float('inf')
        best_environment = None

        for environment in self.environments:
            if self.current_environment is None:
                fitness_increase = environment.individuals[-1].fitness
            else:
                current_fitness = self.current_environment.individuals[-1].fitness
                new_fitness = environment.individuals[-1].fitness
                fitness_increase = new_fitness - current_fitness

            if fitness_increase > max_fitness_increase:
                max_fitness_increase = fitness_increase
                best_environment = environment

        if best_environment:
            self.current_environment = best_environment
            print(f"Switching to environment: {best_environment.name} (Generation {self.current_generation})")
