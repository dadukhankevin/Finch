import numpy as np

# Set the array manager
ARRAY_MANAGER = np

def use_cupy():
    global ARRAY_MANAGER
    import cupy
    ARRAY_MANAGER = cupy