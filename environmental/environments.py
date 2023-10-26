import random
import warnings

import matplotlib.pyplot as plt
from Finch.exceptions.environment_exceptions import NoIndividualsAtEndOfRun
import time
from tabulate import tabulate
from Finch.environmental.layers.standard_layers import Layer

from Finch.genetics.population import Individual


class Environment(Layer):
    def __init__(self, layers: list[Layer] = None, name="Environment", verbose_every=1):
        super().__init__()
        self.generations = 0
        self.verbose_every = verbose_every
        self.callback = None
        self.name = name
        self.layers = layers
        self.individuals = []
        self.iteration = 0
        self.history = []
        self.compiled = False
        self.deactivated = False
        self.best_ever = None


    def deactivate(self):
        self.deactivated = True

    def compile(self, individuals: list[Individual] = None, callback: callable = None, verbose_every: int = 1):
        self.individuals = individuals
        self.compiled = True
        if self.individuals is None:
            self.individuals = []
        self.callback = callback
        self.verbose_every = verbose_every

    def evolve(self, generations):
        if self.deactivated:
            return self.individuals
        fitness = 0
        self.generations = generations
        for i in range(self.generations):
            self.iteration = i
            self.run(self.individuals, self)
        return self.individuals, self.history

    @Layer.Measure
    def run(self, individuals: list[Individual], environment):
        for layer in self.layers:
            self.individuals = layer.run(self.individuals, self)
        if self.callback:
            self.callback(self.individuals, self)
        if len(self.individuals) == 0:
            raise NoIndividualsAtEndOfRun("Your environment has a population of 0 after running.")
        fitness = self.individuals[-1].fitness
        if self.best_ever:
            if fitness > self.best_ever.fitness:
                self.best_ever = self.individuals[-1].copy()
        else:
            self.best_ever = self.individuals[-1].copy()
        if self.verbose_every and self.iteration % self.verbose_every == 0 and self.iteration > 1:            print(
                f"{self.name}: generation {self.iteration + 1}/{self.generations}. Max fitness: {fitness}. Population: "
                f"{len(self.individuals)}")
        self.history.append(fitness)


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
            env.history = []
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
    def __init__(self, environments, switch_every=10, go_for=0, name="AdaptiveEnvironment"):
        super().__init__(name=name)
        self.environments = environments
        self.switch_every = switch_every
        self.current_environment = environments[0]
        self.current_generation = 0
        self.go_for = go_for

    def compile(self, individuals: list[Individual] = None, callback: callable = None, verbose_every: int = 1):
        super().compile(individuals, callback, verbose_every)
        for environment in self.environments:
            if not environment.compiled:
                environment.compile(verbose_every=verbose_every)

    def evolve(self, generations: int):
        for i in range(generations):
            self.iteration += 1
            self.current_generation += 1
            new = self.smart_evolve()
            new.individuals = self.current_environment.individuals
            new.history = self.current_environment.history
            self.current_environment = new
            self.current_environment.evolve(1)

        return self.individuals, self.history

    def smart_evolve(self):
        for i in self.environments:
            if i.fitness < 1:
                print("Deactivating ", i.name)
                i.deactivate()
        if self.iteration % self.switch_every == 0:
            weights = [x.fitness for x in self.environments]
            choices = [x for x in self.environments]
            return random.choices(choices, weights, k=1)[0]
        else:
            return self.current_environment

    def switch_environment(self):
        max_fitness_increase = -float('inf')
        best_environment = None
        first = self.current_environment.individuals[-1].fitness
        for environment in self.environments:
            environment.individuals = self.current_environment.individuals

            environment.evolve(self.go_for)
            new_fitness = environment.individuals[-1].fitness
            fitness_increase = new_fitness - first

            if fitness_increase > max_fitness_increase:
                max_fitness_increase = fitness_increase
                best_environment = environment

        if best_environment:
            best_environment.individuals = self.current_environment.individuals
            self.current_environment = best_environment
            print(f"Switching to environment: {best_environment.name} (Generation {self.current_generation})")

    def plot(self):
        plt.plot(self.current_environment.history)
        plt.show()
