# Tiny-Letter-Net-on-C

## Project Description

This project implements a simple **single-layer neural network** in **C**, capable of learning and performing binary classification on 5x5 matrix input data (e.g., recognizing letters or symbols). The model is minimalistic and designed for educational purposes, illustrating basic neural network concepts such as forward propagation, loss computation, and weight updates via backpropagation.

---

## Code Structure

The code is organized around the following core structures:

- **Neuron** — Represents a neuron with an array of weights, a bias term, and an output.
- **Layer** — Represents a layer of neurons (currently implemented as a single neuron).
- **NeuralNetwork** — Represents the overall neural network, consisting of one layer.

---

## Core Functions

- `initialize_neuron(Neuron *neuron)`  
  Initializes the neuron's weights and bias with default/random values.

- `sigmoid(float x)`  
  Sigmoid activation function: squashes input into the [0, 1] range.

- `forward(NeuralNetwork *nn, float input[INPUT_SIZE])`  
  Performs forward propagation through the neural network and returns the output.

- `mse(float predicted, float actual)`  
  Mean Squared Error loss function.

- `backpropagate(NeuralNetwork *nn, float input[INPUT_SIZE], float actual, float learning_rate)`  
  Performs error backpropagation and updates the neuron’s weights and bias.

- `read_dataset_from_file(const char *filename, float inputs[][INPUT_SIZE], float outputs[], int *num_examples)`  
  Reads the training dataset from a file into input and output arrays.

- `get_inputs_from_file(float input[INPUT_SIZE])`  
  Loads test input data from a file.

- `train(NeuralNetwork *nn, float inputs[][INPUT_SIZE], float outputs[], int num_examples, int epochs, float learning_rate)`  
  Trains the neural network over a number of epochs.

- `test_and_write_to_file(NeuralNetwork *nn, float input[INPUT_SIZE])`  
  Tests the neural network on input data and writes the result to a file.

---

## How to Use

 ### 1. Data preparation

Create a test.txt file for training and an input.txt file for testing.

The format of the test.txt file is:

On the first line, provide the number of examples of the first category (e.g., character).
This is followed by lines with input data for each example of the first category.
Then specify the number of examples of the second category.
The next lines contain the input data for each example of the second category.

### 2. Compiling and running
Use the gcc compiler to compile the program:

```bash
gcc main.c -o neural_network -lm
```

To run the program:
```bash
./neural_network
```

### 3. Result
After executing the program, an output.txt file will be created containing the predicted category and probability.

An example of the contents of output.txt:
```text
It's first symbol
0.734556
```


## Dependencies
- GCC compiler
- Math.h library (for math operations)
