Module Finch.environments
=========================

Classes
-------

`AdaptiveEnvironment(environments, switch_every=10, go_for=0, name='AdaptiveEnvironment')`
:   Represents an adaptive evolutionary environment that switches between different sub-environments.
    
    Args:
    - environments: List of sub-environments.
    - switch_every: Frequency of environment switching.
    - go_for: Number of generations to run each sub-environment before switching.
    - name: Name of the adaptive environment.
    
    :param execution_function: retrieved from a higher class
    :param gene_selection: gene selection method: float, int, Callable
    :param individual_selection: individual selection method: float, int, Callable
    :param fitness: set to 0
    :param device: 'cpu' or 'gpu'

    ### Ancestors (in MRO)

    * Finch.environments.Environment
    * Finch.layers.layer.Layer
    * Finch.genetics.Individual

    ### Methods

    `compile(self, fitness_function, individuals: list[Finch.genetics.Individual] = None, callback: <built-in function callable> = None, verbose_every: int = 1)`
    :   Compiles the adaptive environment with necessary parameters for evolution.
  
  Args:
  - fitness_function: The fitness function for evaluating individuals.
  - individuals (list[Individual]): List of individuals to start evolution with.
  - callback (callable): Callback function to be executed after each generation.
  - verbose_every (int): Frequency of verbose output during evolution.

    `evolve(self, generations: int)`
    :   Evolves the population over a specified number of generations with adaptive switching between sub-environments.
  
  Args:
  - generations (int): Number of generations for evolution.
  
  Returns:
  Tuple containing the final population and the fitness history.

    `plot(self)`
    :   Plots the fitness history of the current sub-environment.

    `smart_evolve(self)`
    :   Implements adaptive evolution by intelligently switching between sub-environments.
  
  Returns:
  The selected sub-environment for the next generation.

    `switch_environment(self)`
    :   Switches to the sub-environment with the highest fitness increase.

`Adversarial(environments, name='Adversarial Environment')`
:   Represents an adversarial evolutionary environment managing multiple sub-environments.
    
    Args:
    - environments: List of sub-environments.
    - name: Name of the adversarial environment.
    
    :param execution_function: retrieved from a higher class
    :param gene_selection: gene selection method: float, int, Callable
    :param individual_selection: individual selection method: float, int, Callable
    :param fitness: set to 0
    :param device: 'cpu' or 'gpu'

    ### Ancestors (in MRO)

    * Finch.environments.Environment
    * Finch.layers.layer.Layer
    * Finch.genetics.Individual

    ### Methods

    `compile(self, fitness_function, individuals: list[Finch.genetics.Individual] = None, callback: <built-in function callable> = None, verbose_every: int = 1)`
    :   Compiles the adversarial environment with necessary parameters for evolution.
  
  Args:
  - fitness_function: The fitness function for evaluating individuals.
  - individuals (list[Individual]): List of individuals to start evolution with.
  - callback (callable): Callback function to be executed after each generation.
  - verbose_every (int): Frequency of verbose output during evolution.

    `evolve(self, generations: int = 1)`
    :   Evolves the population over a specified number of generations for each sub-environment.
  
  Args:
  - generations (int): Number of generations for evolution.
  
  Returns:
  Tuple containing information about the best-performing sub-environment.

    `plot_fitness_histories(self, results)`
    :   Plots the fitness histories of multiple sub-environments.
  
  Args:
  - results: List containing information about each sub-environment's evolution.

`ChronologicalEnvironment(environments_and_generations, name='mixed_environment')`
:   Represents a chronological evolutionary environment with varying sub-environments.
    
    Args:
    - environments_and_generations: List of tuples containing sub-environments and their corresponding generations.
    - name: Name of the chronological environment.
    
    :param execution_function: retrieved from a higher class
    :param gene_selection: gene selection method: float, int, Callable
    :param individual_selection: individual selection method: float, int, Callable
    :param fitness: set to 0
    :param device: 'cpu' or 'gpu'

    ### Ancestors (in MRO)

    * Finch.environments.Environment
    * Finch.layers.layer.Layer
    * Finch.genetics.Individual

    ### Methods

    `compile(self, fitness_function, individuals: list[Finch.genetics.Individual] = None, callback: <built-in function callable> = None, verbose_every: int = 1)`
    :   Compiles the chronological environment with necessary parameters for evolution.
  
  Args:
  - fitness_function: The fitness function for evaluating individuals.
  - individuals (list[Individual]): List of individuals to start evolution with.
  - callback (callable): Callback function to be executed after each generation.
  - verbose_every (int): Frequency of verbose output during evolution.

    `evolve(self, generations=1)`
    :   Evolves the population over a specified number of generations, using different sub-environments for each period.
  
  Args:
  - generations (int): Number of generations for evolution.
  
  Returns:
  Tuple containing the final population and the combined fitness history.

    `plot(self)`
    :   Plots the combined fitness history of sub-environments over time.

`Environment(layers: list[Finch.layers.layer.Layer] = None, name='Environment', verbose_every=1, device='cpu')`
:   Represents an evolutionary environment that manages the evolution of a population of individuals over generations.
    
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
    
    :param execution_function: retrieved from a higher class
    :param gene_selection: gene selection method: float, int, Callable
    :param individual_selection: individual selection method: float, int, Callable
    :param fitness: set to 0
    :param device: 'cpu' or 'gpu'

    ### Ancestors (in MRO)

    * Finch.layers.layer.Layer
    * Finch.genetics.Individual

    ### Descendants

    * Finch.environments.AdaptiveEnvironment
    * Finch.environments.Adversarial
    * Finch.environments.ChronologicalEnvironment
    * Finch.environments.Sequential

    ### Methods

    `compile(self, fitness_function, individuals: list[Finch.genetics.Individual] = None, callback: <built-in function callable> = None, verbose_every: int = 1)`
    :   Compiles the environment with necessary parameters for evolution.
  
  Args:
  - fitness_function: The fitness function for evaluating individuals.
  - individuals (list[Individual]): List of individuals to start evolution with.
  - callback (callable): Callback function to be executed after each generation.
  - verbose_every (int): Frequency of verbose output during evolution.

    `deactivate(self)`
    :   Deactivates the environment, preventing further evolution.

    `evolve(self, generations)`
    :   Evolves the population over a specified number of generations.
  
  Args:
  - generations (int): Number of generations for evolution.
  
  Returns:
  Tuple containing the final population and the fitness history.

    `execute(self, individuals: list[Finch.genetics.Individual])`
    :   Executes one generation of evolution.
  
  Args:
  - individuals (list[Individual]): List of individuals in the population.
  
  Raises:
  - NoIndividualsAtEndOfRun: If the environment has a population of 0 after running.

    `get_fitness_metric(self)`
    :   Returns the fitness metric of the best individual observed so far.

    `plot(self)`
    :

`Sequential(layers, name='default')`
:   Represents a sequential evolutionary environment.
    
    Args:
    - layers: List of layers in the environment.
    - name: Name of the environment.
    
    :param execution_function: retrieved from a higher class
    :param gene_selection: gene selection method: float, int, Callable
    :param individual_selection: individual selection method: float, int, Callable
    :param fitness: set to 0
    :param device: 'cpu' or 'gpu'

    ### Ancestors (in MRO)

    * Finch.environments.Environment
    * Finch.layers.layer.Layer
    * Finch.genetics.Individual

    ### Methods

    `reset(self)`
    :