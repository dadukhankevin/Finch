import numpy as np
from PIL import Image, ImageDraw
from Finch.universal import ARRAY_MANAGER
from Finch.generic import GenePool, Individual, Layer, make_callable
from typing import Callable, List, Tuple, Union

class ImagePool(GenePool):
    def __init__(self, width: int, height: int, channels: int, fitness_function: Callable, device="cpu"):
        super().__init__(generator_function=self.generate_image, fitness_function=fitness_function)
        self.width = width
        self.height = height
        self.channels = channels
        self.device = device

    def generate_image(self):
        if self.device == "cpu":
            image = np.random.randint(0, 256, (self.height, self.width, self.channels), dtype=np.uint8)
            image = Image.fromarray(image)
        elif self.device == "gpu":
            image = ARRAY_MANAGER.random.randint(0, 256, (self.height, self.width, self.channels), dtype=ARRAY_MANAGER.uint8)
            image = Image.fromarray(image)
        return Individual(item=image, fitness_function=self.fitness_function)

