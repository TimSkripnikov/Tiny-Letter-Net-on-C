#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#define INPUT_SIZE 25
#define MIDDLE 0.5

typedef struct
{
    float weights[INPUT_SIZE];

    float bias;
    float output;
} Neuron;

typedef struct
{
    Neuron neurons[1];
} Layer;

typedef struct
{
    Layer layer;

} NeuralNetwork;

void initialize_neuron(Neuron *neuron)
{
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        neuron->weights[i] = 0.1; // We set the neuron weights close to zero
    }
    neuron->bias = 0.1; // We set the neuron's bias close to zero
}

float sigmoid(float x) // calculates the sigmoidal activation function
{
    return 1.0 / (1.0 + exp(-x)); // Converts the input value x to a range from 0 to 1 using the function y(x) = 1/(1+e^(-x))
}

float forward(NeuralNetwork *nn, float input[INPUT_SIZE]) // direct distribution
{
    float sum = 0.0;
    for (int i = 0; i < INPUT_SIZE; i++) // calculates the weighted sum of the input data and the offset
    {
        sum += nn->layer.neurons[0].weights[i] * input[i];
    }
    sum += nn->layer.neurons[0].bias;

    nn->layer.neurons[0].output = sigmoid(sum); // Applies the sigmoidal activation function to the weighted sum.

    return nn->layer.neurons[0].output; // Returns the output value of the neuron.
}

// Loss function (RMS error)
float mse(float predicted, float actual)
{
    return (predicted - actual) * (predicted - actual); // Calculates the square of the difference between the predicted value and the actual value.
}

// Backpropagation
void backpropagate(NeuralNetwork *nn, float input[INPUT_SIZE], float actual, float learning_rate) // performs back propagation of the error and updates the weights and offset.
{
    float predicted = nn->layer.neurons[0].output;

    float error = predicted - actual;

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        nn->layer.neurons[0].weights[i] -= learning_rate * error * predicted * (1 - predicted) * input[i];
    }
    nn->layer.neurons[0].bias -= learning_rate * error * predicted * (1 - predicted);
}

void read_dataset_from_file(const char *filename, float inputs[][INPUT_SIZE], float outputs[], int *num_examples)
{
    FILE *file = fopen(filename, "r");

    if (file == NULL)
    {
        printf("Error opening dataset file.\n");
        exit(1);
    }

    int num_a, num_b;

    fscanf(file, "%d", &num_a);

    *num_examples = 0;

    for (int i = 0; i < num_a; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            fscanf(file, "%1f", &inputs[*num_examples][j]);
        }
        outputs[*num_examples] = 1.0;

        (*num_examples)++;
    }

    fscanf(file, "%d", &num_b);

    for (int i = 0; i < num_b; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            fscanf(file, "%1f", &inputs[*num_examples][j]);
        }
        outputs[*num_examples] = 0.0;
        (*num_examples)++;
    }

    fclose(file);
}

void get_inputs_from_file(float input[INPUT_SIZE])
{
    FILE *file = fopen("input.txt", "r");
    if (file == NULL)
    {
        printf("Error opening input file.\n");
        exit(1);
    }

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        fscanf(file, "%1f", &input[i]);
    }

    fclose(file);
}

void train(NeuralNetwork *nn, float inputs[][INPUT_SIZE], float outputs[], int num_examples, int epochs, float learning_rate)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float total_loss = 0.0f;
        for (int i = 0; i < num_examples; i++)
        {
            float predicted = forward(nn, inputs[i]);                // For each training example, direct propagation is performed
            total_loss += mse(predicted, outputs[i]);                // the loss (mse) between the predicted value and the expected value is calculated
            backpropagate(nn, inputs[i], outputs[i], learning_rate); // Backpropagate is performed, where the weights and bias of the neural network are adjusted based on the obtained error and learning rate.
        }
    }
}

void test_and_write_to_file(NeuralNetwork *nn, float input[INPUT_SIZE])
{
    float output = forward(nn, input);

    const char *predicted_label;
    if (output >= MIDDLE)
    {
        predicted_label = "It's first symbol";
    }
    else
    {
        predicted_label = "It's second symbol";
    }

    FILE *file = fopen("output.txt", "w");
    if (file == NULL)
    {
        printf("Error opening output file.\n");
        exit(1);
    }

    fprintf(file, "%s\n", predicted_label);
    fprintf(file, "%f\n", output);

    fclose(file);
}

int main()
{
    NeuralNetwork nn;
    initialize_neuron(&nn.layer.neurons[0]); // NeuralNetwork structure is created, and its neuron is initialized using the initialize_neuron function, which sets the initial weights and bias

    float inputs[100][INPUT_SIZE]; // Arrays are declared to store input data (training examples), expected outputs and the number of examples

    float outputs[100];
    int num_examples;

    read_dataset_from_file("test.txt", inputs, outputs, &num_examples); // The data set is read from the test.txt file and loaded into the inputs and outputs arrays. The number of samples is stored in the variable num_examples

    int epochs = 1000; // The NN is trained on a set of data. The training process is repeated 1000 times using the specified learning rate of 0.01

    float learning_rate = 0.01;

    train(&nn, inputs, outputs, num_examples, epochs, learning_rate);

    float input[INPUT_SIZE]; // The input data is read from the input.txt file, and the trained neural network is tested. The result is written to the output.txt file.
    get_inputs_from_file(input);
    test_and_write_to_file(&nn, input);

    return 0;
}
