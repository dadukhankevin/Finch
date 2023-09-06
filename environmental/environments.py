import numpy as np

# Try importing CuPy
import matplotlib.pyplot as plt
from Finch.exceptions.environment_exceptions import NoIndividualsAtEndOfRun
import time
from tabulate import tabulate


class Sequential:
    def __init__(self, layers, individuals=None, name="default"):
        if individuals is None:
            individuals = []
        self.name = name
        self.individuals = individuals
        self.layers = layers
        self.stop = False
        self.original = None
        self.diff = None
        self.iteration = 0
        self.history = []

    def evolve(self, generations: int, callback=None, verbose_every=False, track_float_diff_every=False):
        fitness = 0
        for i in range(generations):
            self.iteration = i
            if verbose_every and i % verbose_every == 0:
                print(
                    f"{self.name}: generation {i + 1}/{generations}. Max fitness: {fitness}. Population: {len(self.individuals)}")
            for layer in self.layers:
                self.individuals = layer.run(self.individuals, self)
            if callback:
                callback(self.individuals, self)
            if len(self.individuals) == 0:
                raise NoIndividualsAtEndOfRun("Your environment has a population of 0 after running.")
            self.original = self.individuals[0].copy().genes
            if track_float_diff_every and i % track_float_diff_every == 0:
                self.diff = (-(self.original - self.individuals[0].genes))
            fitness = self.individuals[0].fitness
            self.history.append(fitness)
            if self.stop:
                return self.individuals, self.history
        return self.individuals, self.history

    def stop(self):
        self.stop = True

    def reset(self):
        self.individuals = []
        self.history = []
        self.original = []
        self.diff = []
        self.iteration = 0
        self.stop = True


class Adversarial:
    def __init__(self, environments, generations, verbose_every=1):
        self.environments = environments
        self.generations = generations
        self.verbose_every = verbose_every

    def evolve(self):
        results = []
        results_history = []
        for env in self.environments:
            start_time = time.time()
            individuals, history = env.evolve(self.generations, verbose_every=self.verbose_every)
            end_time = time.time()
            evolution_time = end_time - start_time
            results.append((env.name, history[-1], evolution_time))
            results_history.append((env.name, history[-1], evolution_time, history))

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
        plt.title("Fitness History Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()


class MixedEnvironment:
    def __init__(self, environments_and_generations, verbose_every=1, name="mixed_environment"):
        self.environments_and_generations = environments_and_generations
        self.combined_history = []
        self.verbose_every = verbose_every
        self.best_environment = None
        self.individuals = []
        self.name = name

    def evolve(self, generations=1, verbose_every = 1):
        for env, generations_here in self.environments_and_generations:
            print(f"Running {env.name} for {generations_here} generations...")
            env.individuals = self.individuals
            individuals, history = env.evolve(generations_here, verbose_every=verbose_every)
            self.individuals.extend(individuals)
            self.combined_history.extend(history)
            if not self.best_environment or history[-1] > self.best_environment[1]:
                self.best_environment = (env.name, history[-1])
        return self.individuals, self.combined_history

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
