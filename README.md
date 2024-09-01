# A-single-cell-creature-to-recognize-letters-on-a-5x5-matrix.
## Project Description
This project implements a simple single-layer neural network in programming language C. The neural network is capable of learning and performing classification on data presented as a set of input values.
## Code structure
Main components:
Neuron: Structure representing a neuron, contains an array of weights, bias and output.
Layer: A structure representing a layer of neurons. In this case, a layer with one neuron is implemented.
NeuralNetwork: A structure representing a neural network. Includes one layer of neurons.

## Basic functions
**initialize_neuron(Neuron *neuron)**: Initializes the neuron with the initial values of weights and bias.

**sigmoid(float x)**: Sigmoid activation function, converts the input value to a range from 0 to 1.

**forward(NeuralNetwork *nn, float input[INPUT_SIZE])**: Forward propagation (forward propagation) through a neural network, returns the output value of the neuron.

**mse(float predicted, float actual)**: Loss function, calculates RMS error.

**backpropagate(NeuralNetwork *nn, float input[INPUT_SIZE]**, float actual, float learning_rate): Error backpropagation, updates the weights and bias of the neuron based on the error and learning rate.

**read_dataset_from_file(const char *filename, float inputs[][INPUT_SIZE]**, float outputs[], int *num_examples): Read a dataset from a file and load it into arrays of inputs and expected outputs.

**get_inputs_from_file(float input[INPUT_SIZE])**: Read input data from a file for testing.

**train(NeuralNetwork *nn, float inputs[][INPUT_SIZE], float outputs[], int num_examples, int epochs, float learning_rate)**: Train a neural network on a dataset.

**test_and_write_to_file(NeuralNetwork *nn, float input[INPUT_SIZE])**: Test the trained neural network and write the result to a file.

## Instructions for use
1. Data preparation
Create a test.txt file for training and an input.txt file for testing.

The format of the test.txt file is:

On the first line, provide the number of examples of the first category (e.g., character).
This is followed by lines with input data for each example of the first category.
Then specify the number of examples of the second category.
The next lines contain the input data for each example of the second category.

2. Compiling and running
Use the gcc compiler to compile the program:


gcc main.c -o neural_network -lm
To run the program:

./neural_network
3. Result
After executing the program, an output.txt file will be created containing the predicted category and probability.

An example of the contents of output.txt:

It's first symbol
0.734556
## Dependencies
GCC compiler
Math.h library (for math operations)
