# Finch.FinchGA.Layers docs:
### Age
        :param delay:
        :param every:
        :param years: The amount of years to age an individual
        
### Chromosome

        This is currently not useful in any way but it may eventually be utilized.
        :param genes: the genes in the chromosome
        
### Duplicate

        :param every: See Layer
        :param clones: Amount of clones
        :param delay: See Layer
        :param end: same
        
### Function

        :param fun: The function you want to apply
        :param every: See Layer
        :param delay: See Layer
        :param end: See Layer
        
### GenerateData

        :param gene_pool: The gene pool to be used
        :param population: The population to be generated
        :param array_length:
        
### Individual

        :param pool: the gene pool we want to use
        :param ar: The raw data to turn into Individual of Chromosome of List Gene
        :param fitness_func: The fitness function
        :param fitness: The default fitness
        :param calculate_on_start: Should the fitness be calculated on start? If so, fitness=0 is ignored
        
### KeepLength

        :param delay:
        :param every:
        :param years: The amount of years to age an individual
        
### Kill

        :param percent: The percent to kill (picks from the worst)
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        
### KillByAge

        :param percent: The percent to kill (picks from the worst)
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        
### Layer

        :param every: do this every x times
        :param delay: Delay until used, until then it will simply return what it is given
        :param native_run: The run function to be used from other classes
        :param end: stop after n epochs
        
### Mutate

        :param pool: The gene pool to look through
        :param every: Do this every n
        :param delay: Do this every n after delay in epochs
        :param select_percent: The percent of individuals to select for mutation
        :param likelihood: The likelihood of any individual gene mutating from within the individual (if selected)
        
### NarrowGRN

        :param gene_pool: The gene_pool to modify
        :param method: Can also be "all" defines how to calculate new weights. "all" recalculate
        all of them, "outer" will penalize the lowest fitness ones and reward the highest fitness. "best" will reward
        the best. "worst" will penalize the worst chromosome.
        :param amount: The amount of individuals to look at. Only relevant when the method is not "all".
        :param delay: The delay
        :param reward: The percentage to increase the weight of a gene
        :param penalty: Like reward
        
### Parent

        :param pool: The gene pool to use
        :param every: Do this every n epochs
        :param gene_size: The gene size will determine how to mux parents
        :param family_size: The amount of children to generate
        :param delay: The delay in epochs until this takes effect
        :param native_run: Ignore this
        :param end: When to stop in epochs
        
### Parents

        :param delay: The delay in epochs until this will come into affect
        :param every: Do this ever n epochs
        :param gene_size: The gene size will determine how to mux parents
        :param family_size: The amount of children to generate
        :param percent: The percent to select when method=random
        :param method: Right now only "random" TODO: add more methods
        
### SortFitness

        :param every: same in Layer
        :param delay: same
        :param end: same
        
### UpdateWeights

        :param pool: The gene pool to update
        :param every: Do this every n epochs
        :param delay: Do this after n epochs
        
# Finch.FinchGA.generic docs:
### Chromosome

        This is currently not useful in any way but it may eventually be utilized.
        :param genes: the genes in the chromosome
        
### Equation

        :param equation:
        :param args:
        
### Fittness

        :param funcs: The fitness functions
        :param thresh: The threshold of fitness before you want to move on to the next function
        
### Fuzzy

        TODO: Make it so that individuals that are similar are assigned the same fitness. Helps with optimization
        :param w:
        
### Gene

        :param gene: The raw value of the gene
        :param weight: How much this gene appears in the best individuals
        
### Generation

        :param individuals: List of individuals
        
### Individual

        :param pool: the gene pool we want to use
        :param ar: The raw data to turn into Individual of Chromosome of List Gene
        :param fitness_func: The fitness function
        :param fitness: The default fitness
        :param calculate_on_start: Should the fitness be calculated on start? If so, fitness=0 is ignored

# Finch.FinchGA.AutoGA docs:
### Adversarial

        Competes environments against each other
        :param environments:
        
### GenePool

        :param data: The "vocabulary" to make into genes
        :param fitness_func: The fitness function
        :param mx: The minimum weight
        :param mn: The maximum weight
        
### SequentialEnvironment

        :param every: Defines the environment
        :param layers: The layers of the environment
        
### TypedGenePool

        :param pools: The pools to be included in the mega gene pool
        
### ValueWeight

        :param items: A list of items that, for example could go into a backpack, each item is formatted as such ["item name", value, weight]
        where "value" is how much you "want" the item and "weight" is how much the item weighs. Both of these concepts can be applied to things
        outside of backpacks.
        :param max_weight: The max amount of weight allowed in a backpack
        :param stop_thresh: Stop when the value reaches this amount
        :param epochs: amount of iterations
        

# Finch.FinchGA.Environments docs:
### Adversarial

        Competes environments against each other
        :param environments:
        
### SequentialEnvironment

        :param every: Defines the environment
        :param layers: The layers of the environment

# Finch.FinchGA.EvolveRates docs:
### Rates

        Defines the initial rate and then the arg modifies it based on what function you choose. Please look
        at the file for more.
        :param rate: Initial rate
        :param arg: Modification