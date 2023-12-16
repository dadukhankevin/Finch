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
    def __init__(self, target_labels, other_labels, model="openai/clip-vit-large-patch14", criteria=sum,
                 base_image=None):
        """
        Initializes a ZeroShotImage object.

        Parameters:
        - target_labels (list): List of target labels for zero-shot image classification.
        - other_labels (list): List of additional labels for zero-shot image classification.
        - model (str): The name or path of the model to be used for zero-shot image classification.
        - criteria (function): The aggregation function used to calculate the overall score for labels.
        - base_image (PIL.Image.Image): An optional base image to be combined with generated images.
        """
        self.model = model
        self.target_labels = target_labels
        self.other_labels = other_labels
        self.pipe = pipeline("zero-shot-image-classification", model=model, device=device)
        self.criteria = criteria
        self.base_image = base_image

    def run(self, image, candidate_labels):
        """
        Runs zero-shot image classification on the given image and candidate labels.

        Parameters:
        - image (numpy.ndarray): The image data as a NumPy array.
        - candidate_labels (list): List of candidate labels for classification.

        Returns:
        - dict: Dictionary containing classification results.
        """
        return self.pipe(image, candidate_labels=candidate_labels)

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
        image = Image.fromarray(image_array)
        if self.base_image:
            image += self.base_image
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


class ObjectDetection:
    def __init__(self, target_labels, model="hustvl/yolos-tiny", criteria=sum, base_image=None):
        """
        Initializes an ObjectDetection object.

        Parameters:
        - target_labels (list): List of target labels for object detection.
        - model (str): The name or path of the model to be used for object detection.
        - criteria (function): The aggregation function used to calculate the overall score for labels.
        - base_image (PIL.Image.Image): An optional base image to be combined with input images.
        """
        self.pipe = pipeline("object-detection", model=model)
        self.criteria = criteria
        self.target_labels = target_labels
        self.base_image = base_image

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
        image = Image.fromarray(image_array)
        if self.base_image:
            image += self.base_image
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
