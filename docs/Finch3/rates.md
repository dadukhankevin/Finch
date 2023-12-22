---
description: A helpful tool to change behavior over time
---

# Rates

Almost every int or float parameter in Finch can also take a Rate.

### Rate Class

The `Rate` class represents a dynamic value that changes over a specified number of epochs.

**Parameters:**

* **start:** `float` or `int`
  * The starting value.
* **end:** `float` or `int`
  * The target value after the specified epochs.
* **epochs:** `int`
  * The number of epochs to reach the target value.
* **return\_int:** `bool`
  * Whether to return only integers.

**Methods:**

* **next(self):**
  * Returns the current value and updates it by the change rate.
* **get(self):**
  * Returns the current value without updating it.
* **graph(self):**
  * Plots the history of the value over epochs.

Example:

```
rate = Rate(start=100, end=5, epochs=200)
rate.graph() # will show how it will change over time
ExampleLayer(individual_selection=rate.next)
ExampleLayer(individual_selection-rate.get) # here we still need next() to be called somewhere else
```
