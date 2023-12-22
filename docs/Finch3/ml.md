---
description: machine learning related fitness functions in Finch
---

# ML

### ZeroShotImage Class

The `ZeroShotImage` class manages image evolution through zero-shot image classification using the CLIP model.

**Parameters:**

* **target\_labels:** `list`
  * Labels representing features to be emphasized.
* **other\_labels:** `list`
  * Labels representing other features.
* **shape:** `tuple`
  * Image dimensions (height, width).
* **model:** `str`
  * CLIP model name for zero-shot image classification.
* **criteria:** `callable`
  * Fitness criteria function (default is `sum`).
* **base\_image:** `numpy array`, optional
  * Base image added to the evolved image.
* **denormalize:** `bool`
  * If `True`, denormalizes the image.
* **batch\_size:** `int`
  * Batch size for image processing.

**Methods:**

* **run(self, image, candidate\_labels):**
  * Runs zero-shot image classification on the given image.
* **search(self, terms, scores):**
  * Searches for specified terms in the scores and returns the result based on the fitness criteria.
* **suppress\_fit(self, individual):**
  * Suppresses the fitness of an individual based on its image.
* **suppress\_pil\_image(self, image: Image):**
  * Suppresses the fitness of a PIL image.
* **enhance\_pil\_image(self, image: Image):**
  * Enhances the fitness of a PIL image.
* **enhance\_fit(self, individual):**
  * Enhances the fitness of an individual.
* **get\_image(self, individual):**
  * Retrieves the image representation of an individual.
* **batch\_enhance(self, individuals, pil\_images=False):**
  * Enhances the fitness of a batch of individuals.
* **batch\_suppress(self, individuals, pil\_images):**
  * Suppresses the fitness of a batch of individuals.
* **show(self, individual):**
  * Displays the image representation of an individual.

### ObjectDetection Class

The `ObjectDetection` class orchestrates image evolution through object detection.

**Parameters:**

* **target\_labels:** `list`
  * Labels representing objects to be emphasized.
* **other\_labels:** `list`
  * Labels representing other objects.
* **shape:** `tuple`
  * Image dimensions (height, width).
* **model:** `str`
  * Object detection model name.
* **criteria:** `callable`
  * Fitness criteria function (default is `sum`).
* **base\_image:** `numpy array`, optional
  * Base image added to the evolved image.
* **denormalize:** `bool`
  * If `True`, denormalizes the image.
* **batch\_size:** `int`
  * Batch size for image processing.

**Methods:**

* **run(self, image):**
  * Runs object detection on the given image.
* **search(self, terms, scores):**
  * Searches for specified terms in the scores and returns the result based on the fitness criteria.
* **suppress\_fit(self, individual):**
  * Suppresses the fitness of an individual based on its image.
* **suppress\_pil\_image(self, image: Image):**
  * Suppresses the fitness of a PIL image.
* **enhance\_pil\_image(self, image: Image):**
  * Enhances the fitness of a PIL image.
* **enhance\_fit(self, individual):**
  * Enhances the fitness of an individual.
* **get\_image(self, individual):**
  * Retrieves the image representation of an individual.
* **batch\_enhance(self, individuals, pil\_images=False):**
  * Enhances the fitness of a batch of individuals.
* **batch\_suppress(self, individuals, pil\_images):**
  * Suppresses the fitness of a batch of individuals.
* **show(self, individual):**
  * Displays the image representation of an individual.

### ImageGenerator Class

The `ImageGenerator` class streamlines the generation and fitness evaluation of evolved images using a text-to-image model.

**Parameters:**

* **recognizer:** `ZeroShotImage`
  * Instance of the `ZeroShotImage` class for fitness evaluation.
* **model:** `str`
  * Text-to-image model name.
* **variant:** `str`
  * Model variant.
* **batch\_size:** `int`
  * Batch size for image generation.
* **guidance\_scale:** `float`
  * Guidance scale for the image generation model.
* **num\_inference\_steps:** `int`
  * Number of inference steps for image generation.
* **repeat\_fitness:** `int`
  * Number of times to repeat fitness evaluation.
* **criteria:** `callable`
  * Fitness criteria function (default is `sum`).
* **seed:** `int`
  * Seed for image generation.

**Methods:**

* **generate(self, prompt):**
  * Generates an image based on the given prompt using the text-to-image model.
* **fit(self, individual: Individual):**
  * Evaluates the fitness of an individual by generating an image and using the `ZeroShotImage` recognizer.
* **repeated\_fit(self, individual: Individual):**
  * Repeatedly evaluates the fitness of an individual and returns a list of fitness values.
* **batch\_fit(self, individuals: list\[Individual]):**
  * Evaluates the fitness
