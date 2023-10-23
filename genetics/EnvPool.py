import random
from Finch.genetics.genepools import Pool, BinaryPool, FloatPool
from Finch.environmental.layers import mutation_layers, standard_layers, laysers_by_type
from Finch.tools import rates
from Finch.functions import selection
from Finch.environmental.environments import Adversarial, AdaptiveEnvironment, Sequential
import inspect


def gen_select(mn, mx):  # generate a random selection functions
    selection_functions = [selection.RandomSelection().select, selection.Select().select,
                           selection.TournamentSelection().select,
                           selection.RankBasedSelection(random.randint(mn, mx)).select]
    return random.choice(selection_functions)


def get_vals(obj):
    signature = inspect.signature(obj.__init__)
    init_params = signature.parameters
    args_with_types_and_defaults = [
        (param_name, param.default if param.default != inspect.Parameter.empty else None, param.annotation)
        for param_name, param in init_params.items()
    ]
    return args_with_types_and_defaults[1:-1]


class DynamicRate(rates.Rate):
    def __init__(self):
        super().__init__()


def gen_type(typeof, gene_pool, mn=1, mx=5, length_mn=None, length_mx=None):
    if typeof == int:
        return random.randint(mn, mx)
    elif typeof == float:
        return random.uniform(mn, mx)
    elif typeof == str:
        if length_mn is None:
            length_mn = mn
        if length_mx is None:
            length_mx = mx
        length = random.randint(length_mn, length_mx)
        return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length))
    elif typeof == bool:
        return random.choice([True, False])
    elif typeof == list:
        if length_mn is None:
            length_mn = mn
        if length_mx is None:
            length_mx = mx
        length = random.randint(length_mn, length_mx)
        return [random.randint(mn, mx) for _ in range(length)]
    elif typeof == selection.Select:
        return gen_select(mn, mx)
    elif typeof == Pool:
        return gene_pool
    else:
        raise ValueError(f"Invalid data type: {typeof}. Use 'int', 'float', 'str', 'bool', or 'list'.")


def mutate_arg(arg, mn=1, mx=5):
    typeof = type(arg)
    if typeof == int:
        return min(0, arg + random.randint(mn, mx))
    elif typeof == float:
        return min(0, arg + random.uniform(mn, mx))
    elif typeof == str:
        arg[random.randint(0, len(arg) - 1)] = random.choice("qwertyuiopasdfghjklzxcvbnm")
        return arg
    elif typeof == bool:
        return random.choice([arg, arg, True, False])  # weighted to remain
    elif typeof == list:
        arg[random.randint(0, len(arg) - 1)] = arg[random.randint(0, len(arg) - 1)] = random.randint(mn, mx)
        return
    elif typeof == selection.Select:
        return random.choice([arg, arg, arg, gen_select(mn, mx)])
    else:
        return arg


class DynamicLayerArgs(standard_layers.Layer):
    def __init__(self, pool: Pool, layer: standard_layers.Layer.__class__, args: list = None, kwargs: dict = None):
        super().__init__()
        self.layer = layer
        self.pool = pool
        print(self.layer.__class__)
        args, kwargs = self.create_args()
        self.args = args
        self.kwargs = kwargs
        if not isinstance(self.layer, type):
            self.update_layer_args()
        else:
            self.layer = layer(*args, **kwargs)  # Initialize layer with the provided args and kwargs

    def create_args(self):
        alls = get_vals(self.layer.__class__)
        args = []
        kwargs = {}
        for argument in alls:
            if argument[1] is None:
                args.append(gen_type(argument[2], self.pool))
            else:
                kwargs.update({argument[0]: gen_type(argument[2], self.pool)})
        return args, kwargs

    def mutate_args(self):
        self.args = [mutate_arg(arg, mn=0) for arg in self.args]
        for key in self.kwargs:
            self.kwargs[key] = mutate_arg(self.kwargs[key], mn=0)
        self.update_layer_args()  # Call update_layer_args after mutating args and kwargs

    def update_layer_args(self):
        alls = get_vals(self.layer.__class__)
        args = []
        kwargs = {}
        for argument in alls:
            if argument[1] is None:
                args.append(gen_type(argument[2], self.pool))
            else:
                kwargs.update({argument[0]: gen_type(argument[2], self.pool)})

        # Update the attributes of the existing layer instance without re-initialization
        for key, value in kwargs.items():
            setattr(self.layer, key, value)


class EnvironmentPool(Pool):
    def __init__(self, fitness_function, pool: Pool, valid_layers: list, max_pop=50):
        super().__init__()
        self.fitnes_function = fitness_function
        self.pool = pool
        self.valid_layers = valid_layers
        self.max_pop = max_pop
        self.n = -1

    def generate(self):
        start = [standard_layers.Populate(self.pool, random.randint(2, 10))]
        end = [standard_layers.SortByFitness(), standard_layers.CapPopulation(max_population=self.max_pop)]
        middle = random.choices(self.valid_layers)
        self.n += 1
        return Sequential([DynamicLayerArgs(layer=layer, pool=self.pool)
                           for layer in start + middle + end], "AutoGA: " + str(self.n))

    def generate_genes(self, num_genes):
        return random.choices([DynamicLayerArgs(layer=layer, pool=self.pool) for layer in self.valid_layers],
                              k=num_genes)


class AutoGA(standard_layers.Layer):
    def __init__(self, gene_pool: Pool, max_num_environments, fitness_function, valid_layers, max_pop, max_time):
        super().__init__()
        self.env_pool = EnvironmentPool(fitness_function, gene_pool, valid_layers, max_pop=max_pop)
        self.environment = Sequential(layers=[
            standard_layers.Populate(self.env_pool, 2),
            standard_layers.Function(self.mutate),  # how we will mutate
            standard_layers.Parent(min(1, int(max_num_environments*.2)), 2),
            standard_layers.SortByFitness(),
            standard_layers.CapPopulation(max_num_environments)
        ], name="AutoGA")

    def mutate(self, individuals):
        for individual in individuals:
            individual.mutate_args()
        return individuals

    def evolve(self, epochs, individuals: list[Sequential]=None, environment=None):
        for i in range(epochs):
            self.environment.evolve(1)
            for env in self.environment.individuals:
                print(env.name, env.fitness)


def fit(individual):
    return sum(individual.genes)
pool = FloatPool(0, 10, 10, fit)
autoga = AutoGA(pool, 10, fit, laysers_by_type.float_compatible, 20, 5)

autoga.evolve(10)