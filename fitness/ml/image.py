import numpy as np
import torch
from transformers import pipeline
from PIL import Image
from Finch.genetics import Individual

cp = None
try:
    import cupy as cp
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZeroShotImage:
    def __init__(self, target_labels, other_labels, shape, model="openai/clip-vit-large-patch14", criteria=sum,
                 base_image=None, denormalize=False, batch_size=1):
        """
        Initializes a ZeroShotImage object.

        Parameters:
        - target_labels (list): List of target labels for zero-shot image classification.
        - other_labels (list): List of additional labels for zero-shot image classification.
        - shape: (tuple): The shape the image should become
        - model (str): The name or path of the model to be used for zero-shot image classification.
        - criteria (function): The aggregation function used to calculate the overall score for labels.
        - base_image (PIL.Image.Image): An optional base image to be combined with generated images.
        - denormalize (bool): If true will multiply by 255
        - batch_size (int): passed to pipe(...)
        """
        self.model = model
        self.target_labels = target_labels
        self.other_labels = other_labels
        self.pipe = pipeline("zero-shot-image-classification", model=model, device=device)
        self.criteria = criteria
        self.base_image = base_image
        self.shape = shape
        self.denormalize = denormalize
        self.batch_size = batch_size

    def run(self, image, candidate_labels):
        """
        Runs zero-shot image classification on the given image and candidate labels.

        Parameters:
        - image (numpy.ndarray): The image data as a NumPy array.
        - candidate_labels (list): List of candidate labels for classification.

        Returns:
        - dict: Dictionary containing classification results.
        """
        return self.pipe(image, candidate_labels=candidate_labels, batch_size=self.batch_size)

    def search(self, terms, scores):
        """
        Searches for terms in the scores and applies the criteria function.

        Parameters:
        - terms (list): List of terms to search for in the scores.
        - scores (list): List of scores to be searched.

        Returns:
        - float: The result of applying the criteria function to the selected scores.
        """
        return self.criteria([term['score'] for term in scores if term['label'] in terms])

    def suppress_fit(self, individual: Individual):
        """
        Evaluates the fitness of an individual by suppressing target labels.

        Parameters:
        - individual (Finch.genetics.Individual): The individual to evaluate.

        Returns:
        - float: The negative score indicating the fitness of the individual.
        """
        if individual.device == 'gpu':
            image_array = cp.asnumpy(individual.genes)
        else:
            image_array = individual.genes
        if self.denormalize:
            image_array *= 255
        if self.base_image:
            image_array += self.base_image

        image_array = image_array.reshape(self.shape).astype(np.uint8)

        image = Image.fromarray(image_array)

        labels = self.run(image, self.target_labels + self.other_labels)
        return -self.search(self.target_labels, labels)

    def enhance_fit(self, individual: Individual):
        """
        Evaluates the fitness of an individual by enhancing target labels.

        Parameters:
        - individual (Finch.genetics.Individual): The individual to evaluate.

        Returns:
        - float: The negative score indicating the fitness of the individual.
        """
        return -self.suppress_fit(individual)

    def get_image(self, individual):
        if individual.device == 'gpu':
            image_array = cp.asnumpy(individual.genes)
        else:
            image_array = individual.genes
        if self.denormalize:
            image_array *= 255
        if self.base_image:
            image_array += self.base_image

        image_array = image_array.reshape(self.shape).astype(np.uint8)
        image = Image.fromarray(image_array)
        return image

    def batch_enhance(self, individuals):
        raw_images = [self.get_image(im) for im in individuals]
        scores = self.run(raw_images, candidate_labels=self.target_labels + self.other_labels)
        ind = 0
        for score in scores:
            individuals[ind].fitness = self.search(self.target_labels, score)

    def batch_supress(self, individuals):
        self.batch_enhance(individuals)
        for individual in individuals:
            individual.fitness *= -1

    def show(self, individual):
        image = self.get_image(individual)
        image.show()
        return image


class ObjectDetection:
    def __init__(self, target_labels, shape, denormalize=False, model="hustvl/yolos-tiny", criteria=sum,
                 base_image=None):
        """
        Initializes an ObjectDetection object.

        Parameters:
        - target_labels (list): List of target labels for object detection.
        - shape (tuple): The shape the image should conform to
        - model (str): The name or path of the model to be used for object detection.
        - denormalize (bool): If true will multiply by 255
        - criteria (function): The aggregation function used to calculate the overall score for labels.
        - base_image (PIL.Image.Image): An optional base image to be combined with input images.
        """
        self.pipe = pipeline("object-detection", model=model)
        self.criteria = criteria
        self.target_labels = target_labels
        self.base_image = base_image
        self.denormalize = denormalize
        self.shape = shape

    def search(self, terms, scores):
        """
        Searches for terms in the scores and applies the criteria function.

        Parameters:
        - terms (list): List of terms to search for in the scores.
        - scores (list): List of scores to be searched.

        Returns:
        - float: The result of applying the criteria function to the selected scores.
        """
        return self.criteria([term['score'] for term in scores if term['label'] in terms])

    def run(self, image):
        """
        Runs object detection on the given image.

        Parameters:
        - image (numpy.ndarray): The image data as a NumPy array.

        Returns:
        - list: List of dictionaries containing object detection results.
        """
        return self.pipe(image)

    def suppress_fit(self, individual: Individual):
        """
        Evaluates the fitness of an individual by suppressing target labels.

        Parameters:
        - individual (Finch.genetics.Individual): The individual to evaluate.

        Returns:
        - float: The negative score indicating the fitness of the individual.
        """
        if individual.device == 'gpu':
            image_array = cp.asnumpy(individual.genes)
        else:
            image_array = individual.genes
        if self.denormalize:
            image_array *= 255
        if self.base_image:
            image_array += self.base_image

        image_array = image_array.reshape(self.shape).astype(np.uint8)

        image = Image.fromarray(image_array)

        labels = self.run(image)
        return -self.search(self.target_labels, labels)

    def enhance_fit(self, individual: Individual):
        """
        Evaluates the fitness of an individual by enhancing target labels.

        Parameters:
        - individual (Finch.genetics.Individual): The individual to evaluate.

        Returns:
        - float: The negative score indicating the fitness of the individual.
        """
        return -self.suppress_fit(individual)

    def show(self, individual):
        if individual.device == 'gpu':
            image_array = cp.asnumpy(individual.genes)
        else:
            image_array = individual.genes
        if self.denormalize:
            image_array *= 255
        if self.base_image:
            image_array += self.base_image

        image_array = image_array.reshape(self.shape).astype(np.uint8)
        image = Image.fromarray(image_array)
        image.show()
        return image
