import numpy as np
import torch
from transformers import pipeline
from PIL import Image
from Finch.genetics import Individual
from diffusers import AutoPipelineForText2Image

cp = None
try:
    import cupy as cp
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZeroShotImage:
    def __init__(self, target_labels, other_labels, shape, model="openai/clip-vit-large-patch14", criteria=sum,
                 base_image=None, denormalize=False, batch_size=1):
        self.model = model
        self.target_labels = target_labels
        self.other_labels = other_labels
        self.pipe = pipeline("zero-shot-image-classification", model=model, device=device)
        self.criteria = criteria
        self.base_image = base_image
        self.shape = shape
        self.size = (shape[0], shape[1])
        self.denormalize = denormalize
        self.batch_size = batch_size

    def run(self, image, candidate_labels):
        return self.pipe(image, candidate_labels=candidate_labels, batch_size=self.batch_size)

    def search(self, terms, scores):
        return self.criteria([term['score'] for term in scores if term['label'] in terms])

    def suppress_fit(self, individual: Individual):
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

        return self.supress_pil_image(image)

    def supress_pil_image(self, image: Image):
        labels = self.run(image, self.target_labels + self.other_labels)
        return -self.search(self.target_labels, labels)

    def enhance_pil_image(self, image: Image):
        return -self.supress_pil_image(image)

    def enhance_fit(self, individual: Individual):
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

    def batch_enhance(self, individuals, pil_images=False):
        if not pil_images:
            raw_images = [self.get_image(im) for im in individuals]
        else:
            raw_images = pil_images

        scores = self.run(raw_images, candidate_labels=self.target_labels + self.other_labels)
        ind = 0
        for score in scores:
            individuals[ind].fitness = self.search(self.target_labels, score)
            ind += 1

    def batch_supress(self, individuals, pil_images):
        self.batch_enhance(individuals, pil_images)
        for individual in individuals:
            individual.fitness *= -1

    def show(self, individual):
        image = self.get_image(individual)
        image.show()
        return image


class ObjectDetection:
    def __init__(self, target_labels, other_labels, shape, model="hustvl/yolos-tiny", criteria=sum,
                 base_image=None, denormalize=False, batch_size=1):
        # Use meaningful names for the parameters
        self.model_name = model
        self.target_labels = target_labels
        self.other_labels = other_labels
        self.shape = shape
        self.criteria = criteria
        self.base_image = base_image
        self.denormalize = denormalize
        self.batch_size = batch_size

        # Use the built-in pipeline function to create the object detector
        self.pipe = pipeline("object-detection", model=self.model_name, device=device)

    def run(self, image):
        # Use the pipeline to get the labels and scores for the image
        return self.pipe(image)

    def search(self, terms, scores):
        # Use a list comprehension to get the scores for the terms
        return self.criteria([term['score'] for term in scores if term['label'] in terms])

    def suppress_fit(self, individual: Individual):
        # Use numpy to convert the genes to a numpy array
        image_array = np.array(individual.genes)

        # Use the denormalize and base_image attributes to modify the image array
        if self.denormalize:
            image_array *= 255
        if self.base_image:
            image_array += self.base_image

        # Reshape and cast the image array to uint8
        image_array = image_array.reshape(self.shape).astype(np.uint8)

        # Create an image from the array
        image = Image.fromarray(image_array)

        # Get the labels and scores for the image
        labels = self.run(image)

        # Return the negative score for the target labels
        return -self.search(self.target_labels, labels)

    def supress_pil_image(self, image: Image):
        # Return the negative score for the target labels
        return self.suppress_fit(image)

    def enhance_pil_image(self, image: Image):
        # Return the positive score for the target labels
        return -self.supress_pil_image(image)

    def enhance_fit(self, individual: Individual):
        # Return the positive score for the target labels
        return -self.suppress_fit(individual)

    def get_image(self, individual):
        # Use numpy to convert the genes to a numpy array
        image_array = np.array(individual.genes)

        # Use the denormalize and base_image attributes to modify the image array
        if self.denormalize:
            image_array *= 255
        if self.base_image:
            image_array += self.base_image

        # Reshape and cast the image array to uint8
        image_array = image_array.reshape(self.shape).astype(np.uint8)

        # Create and return an image from the array
        return Image.fromarray(image_array)

    def batch_enhance(self, individuals, pil_images=False):
        # Use a list comprehension to get the images from the individuals
        if not pil_images:
            raw_images = [self.get_image(im) for im in individuals]
        else:
            raw_images = pil_images

        # Get the labels and scores for the images
        scores = self.run(raw_images)

        # Use a for loop with enumerate to update the fitness of the individuals
        for ind, score in enumerate(scores):
            individuals[ind].fitness = self.search(self.target_labels, score)

    def batch_supress(self, individuals, pil_images):
        # Use the batch_enhance method to get the fitness of the individuals
        self.batch_enhance(individuals, pil_images)

        # Use a list comprehension to negate the fitness of the individuals
        for individual in individuals:
            individual.fitness *= -1

    def show(self, individual):
        # Use the get_image method to get the image from the individual
        image = self.get_image(individual)

        # Show the image
        image.show()

        # Return the image
        return image


class ImageGenerator:
    def __init__(self, recognizer: ZeroShotImage, model='stabilityai/sd-turbo', variant='fp16',
                 batch_size=1, guidance_scale=0.0, num_inference_steps=1, repeat_fitness=1, criteria=sum, seed=-1):
        self.pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16,
                                                              variant=variant)
        self.pipe.to("cuda")
        self.batch_size = batch_size
        self.recognizer = recognizer
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.repeat_fitness = repeat_fitness
        self.criteria = criteria
        self.seed = seed

    def generate(self, prompt):
        image = self.pipe(prompt=prompt, num_inference_steps=self.num_inference_steps,
                          guidance_scale=self.guidance_scale,
                          batch_size=self.batch_size, seed=self.seed)[0]
        return image

    def fit(self, individual: Individual):
        prompt = "".join(str(i) for i in individual.genes)
        image = self.generate(prompt)[0]
        return self.recognizer.enhance_pil_image(image)

    def repeated_fit(self, individual: Individual):
        assert self.seed == -1, "Repeated_fit should only be used when seed is randomized, please set it to -1"
        return self.criteria([self.fit(individual) for _ in range(self.repeat_fitness)])

    def batch_fit(self, individuals: list[Individual]):
        prompts = ["".join(str(i) for i in individual.genes) for individual in individuals]
        images = self.generate(prompts)
        self.recognizer.batch_enhance(individuals, pil_images=images)

    def batch_repeated_fit(self, individuals):
        fitness_values_list = []

        for _ in range(self.repeat_fitness):
            self.batch_fit(individuals)
            fitness_values = [individual.fitness for individual in individuals]
            fitness_values_list.append(fitness_values)

        aggregated_fitness_values = [self.criteria(values) for values in zip(*fitness_values_list)]

        for i, individual in enumerate(individuals):
            individual.fitness = aggregated_fitness_values[i]