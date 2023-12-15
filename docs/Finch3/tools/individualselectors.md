Module Finch3.tools.individualselectors
=======================================

Classes
-------

`RandomSelection(percent_to_select=None, amount_to_select=None)`
:   Random selection strategy.
    
    Parameters:
    - percent_to_select: A callable returning the percentage of individuals to select.
    - amount_to_select: A callable returning the number of individuals to select.
    
    Usage:
    ```
    selector = RandomSelection(percent_to_select=lambda: 0.2)
    selected_individuals = selector.select(individuals)
    ```

    ### Ancestors (in MRO)

    * Finch3.tools.individualselectors.Select

    ### Methods

    `select(self, individuals)`
    :   Select individuals randomly.
        
        Parameters:
        - individuals: List of individuals to select from.
        
        Returns:
        - list[Individual]: Selected individuals.

`RankBasedSelection(factor, percent_to_select=None, amount_to_select=None)`
:   Rank-based selection strategy.
    
    Parameters:
    - factor: Selection pressure factor.
    - percent_to_select: A callable returning the percentage of individuals to select.
    - amount_to_select: A callable returning the number of individuals to select.
    
    Usage:
    ```
    selector = RankBasedSelection(factor=1.5, percent_to_select=lambda: 0.2)
    selected_individuals = selector.select(individuals)
    ```

    ### Ancestors (in MRO)

    * Finch3.tools.individualselectors.Select

    ### Methods

    `select(self, individuals)`
    :   Select individuals using rank-based selection.
        
        Parameters:
        - individuals: List of individuals to select from.
        
        Returns:
        - list[Individual]: Selected individuals.

`Select(percent_to_select=None, amount_to_select=None)`
:   Base class for selection strategies.
    
    Parameters:
    - percent_to_select: A callable returning the percentage of individuals to select.
    - amount_to_select: A callable returning the number of individuals to select.
    
    Usage:
    ```
    selector = Select(percent_to_select=lambda: 0.2)
    selected_individuals = selector.select(individuals)
    ```

    ### Descendants

    * Finch3.tools.individualselectors.RandomSelection
    * Finch3.tools.individualselectors.RankBasedSelection
    * Finch3.tools.individualselectors.TournamentSelection

    ### Methods

    `select(self, individuals: list[Finch3.genetics.Individual])`
    :   Abstract method for selecting individuals.
        
        Parameters:
        - individuals: List of individuals to select from.
        
        Returns:
        - list[Individual]: Selected individuals.

`TournamentSelection(percent_to_select=None, amount_to_select=None)`
:   Tournament selection strategy.
    
    Parameters:
    - percent_to_select: A callable returning the percentage of individuals to select.
    - amount_to_select: A callable returning the number of individuals to select.
    
    Usage:
    ```
    selector = TournamentSelection(percent_to_select=lambda: 0.2)
    selected_individuals = selector.select(individuals)
    ```

    ### Ancestors (in MRO)

    * Finch3.tools.individualselectors.Select

    ### Methods

    `select(self, individuals)`
    :   Select individuals using tournament selection.
        
        Parameters:
        - individuals: List of individuals to select from.
        
        Returns:
        - list[Individual]: Selected individuals.