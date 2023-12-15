Module Finch3.layers.parenting
==============================

Functions
---------

    
`crossover(parent1: Finch3.genetics.Individual, parent2: Finch3.genetics.Individual) ‑> list`
:   Perform single-point crossover between two parents.
    
    Args:
        parent1 (Individual): The first parent.
        parent2 (Individual): The second parent.
    
    Returns:
        list of Individual: Two offspring resulting from the crossover.

    
`n_point_crossover(parent1: Finch3.genetics.Individual, parent2: Finch3.genetics.Individual, n=1) ‑> list`
:   Perform n-point crossover between two parents.
    
    Args:
        parent1 (Individual): The first parent.
        parent2 (Individual): The second parent.
        n (int): Number of crossover points.
    
    Returns:
        list of Individual: Offspring resulting from the crossover.

    
`uniform_crossover(parent1: Finch3.genetics.Individual, parent2: Finch3.genetics.Individual, probability=0.5) ‑> list`
:   Perform uniform crossover between two parents.
    
    Args:
        parent1 (Individual): The first parent.
        parent2 (Individual): The second parent.
        probability (float): The probability of selecting a gene from the first parent.
    
    Returns:
        list of Individual: Two offspring resulting from the crossover.

Classes
-------

`ParentNPoint(families, points=1, children=2, refit=True)`
:   Base parenting/crossover layer, not to be used directly.
    
    Initialize an n-point parent with crossover.
    
    Args:
        families: Selection method for individuals.
        points (int): Number of crossover points.
        children (int): Number of children to generate.
        refit (bool): If true will retest fitness in new children

    ### Ancestors (in MRO)

    * Finch3.layers.layer.Parent
    * Finch3.layers.layer.Layer
    * Finch3.genetics.Individual

    ### Methods

    `parent(self, parent1: Finch3.genetics.Individual, parent2: Finch3.genetics.Individual, environment) ‑> list`
    :   Generate children through n-point crossover.
        
        Args:
            parent1 (Individual): The first parent.
            parent2 (Individual): The second parent.
            environment: Additional environment information.
        
        Returns:
            list of Individual: Offspring resulting from the crossover.

`ParentSimple(families, children=2, refit=True)`
:   Base parenting/crossover layer, not to be used directly.
    
    Initialize a simple parent with crossover.
    
    Args:
        families: Selection method for individuals.
        children (int): Number of children to generate.
        refit (bool): If true will retest fitness in new children

    ### Ancestors (in MRO)

    * Finch3.layers.layer.Parent
    * Finch3.layers.layer.Layer
    * Finch3.genetics.Individual

    ### Methods

    `parent(self, parent1: Finch3.genetics.Individual, parent2: Finch3.genetics.Individual, environment) ‑> list`
    :   Generate children through crossover.
        
        Args:
            parent1 (Individual): The first parent.
            parent2 (Individual): The second parent.
            environment: Additional environment information.
        
        Returns:
            list of Individual: Offspring resulting from the crossover.