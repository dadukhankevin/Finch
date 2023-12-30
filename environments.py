import random
import warnings
import matplotlib.pyplot as plt
from Finch.exceptions.environment_exceptions import NoIndividualsAtEndOfRun
import time
from tabulate import tabulate
from Finch.layers import layer
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from Finch.genetics import Individual


class Environment(layer.Layer):
    """
        Represents an evolutionary environment that manages the evolution of a population of individuals over generations.

        Args:
        - layers (list[layer.Layer]): List of layers in the environment.
        - name (str): Name of the environment.
        - verbose_every (int): Frequency of verbose output during evolution (aka print every n times)
        - device (str): Device to run the environment on. 'gpu' or 'cpu'

        Attributes:
        - fitness_function: The fitness function used for evaluating individuals.
        - generations: Number of generations to evolve for.
        - callback: Callback function to be executed after each generation. Set this in .compile()
        - layers: List of layers in the environment.
        - individuals: List of individuals in the population.
        - iteration: Current iteration during evolution.
        - history: List to store best fitness values over generations.
        - compiled: Indicates whether the environment has been compiled.
        - deactivated: Indicates whether the environment is deactivated.
        - best_ever: The best individual observed during evolution.
        """

    def __init__(self, layers: list[layer.Layer] = None, name="Environment", verbose_every=1, device='cpu'):
        super().__init__(device=device, refit=False, individual_selection=None, fitness=1)
        self.fitness_function = None

        self.generations = 0
        self.verbose_every = verbose_every
        self.callback = None
        self.name = name
        self.layers = layers
        for lay in layers:
            lay.set_environment(self)
        self.individuals = []
        self.iteration = 0
        self.history = []
        self.compiled = False
        self.deactivated = False
        self.best_ever = None
        self.dead_individuals = []

    def deactivate(self):
        """
        Deactivates the environment, preventing further evolution.
        """
        self.deactivated = True

    def get_fitness_metric(self):
        """
        Returns the fitness metric of the best individual observed so far.
        """

        if self.best_ever:
            return self.best_ever.fitness
        else:
            return 0

    def batch(self, individual):
        individual.check_fitness = True
        return individual.fitness

    def compile(self, fitness_function, individuals: list[Individual] = None, callback: callable = None,
                verbose_every: int = 1):
        """
        Compiles the environment with necessary parameters for evolution.

        Args:
        - fitness_function: The fitness function for evaluating individuals.
        - individuals (list[Individual]): List of individuals to start evolution with.
        - callback (callable): Callback function to be executed after each generation.
        - verbose_every (int): Frequency of verbose output during evolution.
        """
        self.fitness_function = fitness_function

        if self.fitness_function == 'batch':
            self.fitness_function = self.batch

        self.individuals = individuals
        self.compiled = True
        if self.individuals is None:
            self.individuals = []
        self.callback = callback
        self.verbose_every = verbose_every

    def evolve(self, generations):
        """
        Evolves the population over a specified number of generations.

        Args:
        - generations (int): Number of generations for evolution.

        Returns:
        Tuple containing the final population and the fitness history.
        """

        if self.deactivated:
            return self.individuals
        self.generations = generations
        for i in range(self.generations):
            self.iteration = i
            self.execute(self.individuals)
        return self.individuals, self.history

    def execute(self, individuals: list[Individual]):
        """
        Executes one generation of evolution.

        Args:
        - individuals (list[Individual]): List of individuals in the population.

        Raises:
        - NoIndividualsAtEndOfRun: If the environment has a population of 0 after running.
        """
        for a_layer in self.layers:
            a_layer.run(self.individuals, self)
        if self.callback:
            self.callback(self.individuals, self)
        if len(self.individuals) == 0:
            raise NoIndividualsAtEndOfRun("Your environment has a population of 0 after running.")
        fitness = self.individuals[0].fitness
        self.fitness = 1 + fitness
        if self.best_ever:
            if fitness > self.best_ever.fitness:
                self.best_ever = self.individuals[0].copy()
        else:
            self.best_ever = self.individuals[0].copy()
        if self.verbose_every and self.iteration % self.verbose_every == 0 and self.iteration > 1:
            print(
            f"{self.name}: generation {self.iteration + 1}/{self.generations}. Max fitness: {fitness}. Population: "
            f"{len(self.individuals)}")
        self.history.append(fitness)

    def plot(self):
        plt.title("Fitness graph")
        plt.plot(self.history)
        plt.show()


class Sequential(Environment):
    """
        Represents a sequential evolutionary environment.

        Args:
        - layers: List of layers in the environment.
        - name: Name of the environment.
        """

    def __init__(self, layers, name="default"):
        super().__init__(layers, name)
        self.layers = layers

    def compile(self, fitness_function, individuals: list[Individual] = None, callback: callable = None,
                verbose_every: int = 1):
        super().compile(individuals=individuals, callback=callback, verbose_every=verbose_every,
                        fitness_function=fitness_function)

    def evolve(self, generations: int):
        return super().evolve(generations)

    def reset(self):
        self.individuals = []
        self.history = []
        self.iteration = 0


class Adversarial(Environment):
    """
    Represents an adversarial evolutionary environment managing multiple sub-environments.

    Args:
    - environments: List of sub-environments.
    - name: Name of the adversarial environment.
    """

    def __init__(self, environments, name="Adversarial Environment"):
        super().__init__(environments, name)
        self.environments = environments

    def compile(self, fitness_function, individuals: list[Individual] = None, callback: callable = None,
                verbose_every: int = 1):
        """
         Compiles the adversarial environment with necessary parameters for evolution.

         Args:
         - fitness_function: The fitness function for evaluating individuals.
         - individuals (list[Individual]): List of individuals to start evolution with.
         - callback (callable): Callback function to be executed after each generation.
         - verbose_every (int): Frequency of verbose output during evolution.
        """
        super().compile(individuals=individuals, callback=callback, verbose_every=verbose_every,
                        fitness_function=fitness_function)

        for environment in self.environments:
            if not environment.compiled:
                environment.compile(verbose_every=verbose_every, fitness_function=fitness_function)

    def evolve(self, generations: int = 1):
        """
        Evolves the population over a specified number of generations for each sub-environment.

        Args:
        - generations (int): Number of generations for evolution.

        Returns:
        Tuple containing information about the best-performing sub-environment.
        """
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
        """
        Plots the fitness histories of multiple sub-environments.

        Args:
        - results: List containing information about each sub-environment's evolution.
        """
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
    """
    Represents a chronological evolutionary environment with varying sub-environments.

    Args:
    - environments_and_generations: List of tuples containing sub-environments and their corresponding generations.
    - name: Name of the chronological environment.
    """

    def __init__(self, environments_and_generations, name="mixed_environment"):
        super().__init__(None, name)
        self.environments_and_generations = environments_and_generations
        self.combined_history = []
        self.best_environment = None
        self.individuals = []

    def evolve(self, generations=1):
        """
        Evolves the population over a specified number of generations, using different sub-environments for each period.

        Args:
        - generations (int): Number of generations for evolution.

        Returns:
        Tuple containing the final population and the combined fitness history.
        """

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

    def compile(self, fitness_function, individuals: list[Individual] = None, callback: callable = None,
                verbose_every: int = 1):
        """
        Compiles the chronological environment with necessary parameters for evolution.

        Args:
        - fitness_function: The fitness function for evaluating individuals.
        - individuals (list[Individual]): List of individuals to start evolution with.
        - callback (callable): Callback function to be executed after each generation.
        - verbose_every (int): Frequency of verbose output during evolution.
        """
        super().compile(individuals=individuals, callback=callback, verbose_every=verbose_every,
                        fitness_function=fitness_function)

        for environment, generations_here in self.environments_and_generations:
            if not environment.compiled:
                environment.compile(verbose_every=verbose_every)

    def plot(self):
        """
        Plots the combined fitness history of sub-environments over time.
        """
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
    """
    Represents an adaptive evolutionary environment that switches between different sub-environments.

    Args:
    - environments: List of sub-environments.
    - switch_every: Frequency of environment switching.
    - go_for: Number of generations to run each sub-environment before switching.
    - name: Name of the adaptive environment.
    """

    def __init__(self, environments, reset_every=20, name="AdaptiveEnvironment"):
        super().__init__(environments, name=name)
        self.environments = environments
        self.reset_every = reset_every
        self.current_environment = environments[0]
        self.current_generation = 0
        self.weights = [[1] for _ in range(len(environments))]
        self.total_history = []

    def compile(self, fitness_function, individuals: list[Individual] = None, callback: callable = None,
                verbose_every: int = 1):
        """
        Compiles the adaptive environment with necessary parameters for evolution.

        Args:
        - fitness_function: The fitness function for evaluating individuals.
        - individuals (list[Individual]): List of individuals to start evolution with.
        - callback (callable): Callback function to be executed after each generation.
        - verbose_every (int): Frequency of verbose output during evolution.
        """

        super().compile(individuals=individuals, callback=callback, verbose_every=verbose_every,
                        fitness_function=fitness_function)

        for environment in self.environments:
            if not environment.compiled:
                environment.compile(verbose_every=verbose_every, fitness_function=fitness_function)

    def evolve(self, generations: int):
        """
        Evolves the population over a specified number of generations with adaptive switching between sub-environments.

        Args:
        - generations (int): Number of generations for evolution.

        Returns:
        Tuple containing the final population and the fitness history.
        """
        old_fitness = 1
        for i in range(generations):
            weights = [max(sum(w)/len(w), 0.01) for w in self.weights]
            environment = random.choices(self.environments, weights=weights, k=1)[0]
            if i % self.reset_every == 0:
                self.weights = [[1] for _ in range(len(self.environments))]

            environment.individuals = self.individuals
            environment.evolve(1)
            self.individuals = environment.individuals
            new_fitness = environment.fitness
            self.total_history.append(self.environments.index(environment))

            dif = new_fitness - old_fitness
            old_fitness = new_fitness
            self.weights[self.environments.index(environment)].append(dif)
        weights = [sum(w) / len(w) for w in self.weights]
        print(weights)

    def plot(self):

        plt.plot(self.total_history)
        plt.show()
