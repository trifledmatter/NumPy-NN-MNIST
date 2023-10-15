# Neural Network Implementation in Python

This project showcases a simple implementation of a MNIST NN in Python using only vanilla Python and `NumPy`, focusing on understanding the fundamental concepts and operations involved. The neural network is designed to learn from the given training data and make predictions accordingly.

## Usage

To use this implementation, create a `Network` instance with the desired architecture and training parameters. Then, utilize the `SGD` method to train the network using your training data.

```python
# Create a neural network with desired architecture
net = Network([input_size, hidden_layer_size, output_size])

# Train the network using Stochastic Gradient Descent
net.SGD(training_data, epochs, mini_batch_size, eta, test_data=test_data)
```

## Acknowledgments

This project was inspired by and based on the works of:

- [GeeksforGeeks - Implementation of Neural Network from Scratch using NumPy](https://www.geeksforgeeks.org/implementation-of-neural-network-from-scratch-using-numpy/)
- [Real Python - Python AI: Neural Network](https://realpython.com/python-ai-neural-network/)
