import random
import numpy as np
ar = np.array([1]*100)
print("defined")
def test():
    """Stupid test function"""
    for i in range(5):
        [i for i in range(len(ar))]
if __name__ == '__main__':
    import timeit
    print(timeit.timeit("test()", setup="from __main__ import test"))