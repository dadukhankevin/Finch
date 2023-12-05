import copy
import numpy as np
from Finch.exceptions.population_exceptions import IndividualGenesNotArrayType
cp = None
set = False

do_gpu = False
NPCP = np
if do_gpu:
    try:
        import cupy as cp

        # Check if GPU is available
        if cp.cuda.runtime.getDeviceCount() > 0:
            print("GPU detected. Using CuPy.")
            NPCP = cp
        else:
            print("GPU not detected. Using NumPy.")
            cp = np
            NPCP = np

    except ImportError:
        print("CuPy not found. Using NumPy.")
        NPCP = np


def force_gpu_off():
    global NPCP
    import numpy as np
    NPCP = np


def can_use_cupy(array):
    global set
    set = True
    # Check if the data type of the array is compatible with CuPy
    if np.issubdtype(array.dtype, np.number):
        return True
    # Check if the data type of the array can be safely cast to a CuPy-supported data type
    elif np.can_cast(array.dtype, cp.float64):
        return True
    # Otherwise, return False
    else:
        return False


class Individual:
    def __init__(self, genes, fitness_function, as_array=True):
        self.as_array = as_array
        self.ar = NPCP
        if not as_array:
            self.genes = genes
        else:
            self.genes = NPCP.asarray(genes)
        self.fitness = 0
        self.fitness_function = fitness_function
        self.frozen_genes = NPCP.array([])

    def freeze(self, gene_indices):
        if self.as_array:
            self.frozen_genes = np.append(self.frozen_genes, gene_indices)
            self.frozen_genes = np.unique(self.frozen_genes)  # TODO: is this the worst way of doing this?
        else:
            raise IndividualGenesNotArrayType('- You are trying to freeze genes that are of type: '+str(type(self.genes)))

    def thaw(self, gene_indices):
        if self.as_array:
            self.frozen_genes = np.setdiff1d(self.frozen_genes, gene_indices)
        else:
            raise IndividualGenesNotArrayType('- You are trying to thaw genes that are of type: '+str(type(self.genes)))
    def fit(self):
        self.fitness = self.fitness_function(self.genes)

    def copy(self):  # this wont work with mutable fitness functions!!!
        if self.as_array:
            copied_genes = NPCP.copy(self.genes)
        else:
            copied_genes = copy.deepcopy(self.genes)
        copied_fitness_function = self.fitness_function
        copied_individual = Individual(copied_genes, copied_fitness_function, as_array=self.as_array)
        copied_individual.fitness = self.fitness
        copied_individual.frozen_genes = self.frozen_genes

        return copied_individual
