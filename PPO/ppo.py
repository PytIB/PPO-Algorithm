import numpy as np


import numpy as np

class Actor:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases for the hidden layer
        self.weights_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros(self.hidden_size)

        # Initialize weights and biases for the output layer
        self.weights_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros(self.output_size)

    def forward(self, state):
        # change the state to 1D array 
        state_one_hot = np.zeros(self.input_size)
        state_one_hot[state[0] * int(np.sqrt(self.input_size)) + state[1]] = 1

        # Forward pass through the hidden layer
        hidden_layer = np.maximum(0, np.dot(state_one_hot, self.weights_hidden) + self.bias_hidden)

        # Forward pass through the output layer with softmax activation
        logits = np.dot(hidden_layer, self.weights_output) + self.bias_output
        action_probs = self.softmax(logits)

        return action_probs

      

    def softmax(self, x):
        # Numerically stable softmax function
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class ValueNetwork:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases for the hidden layer
        self.weights_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros(self.hidden_size)

        # Initialize weights and biases for the output layer
        self.weights_output = np.random.randn(self.hidden_size, 1)
        self.bias_output = np.zeros(1)

    def forward(self, state):
        # Flatten the state to a 1D array
        state_flattened = state.reshape(-1)

        # Forward pass through the hidden layer
        hidden_layer = np.maximum(0, np.dot(state_flattened, self.weights_hidden) + self.bias_hidden)

        # Forward pass through the output layer (linear activation)
        value = np.dot(hidden_layer, self.weights_output) + self.bias_output

        return value


