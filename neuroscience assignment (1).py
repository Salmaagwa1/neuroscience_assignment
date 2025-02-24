
import numpy as np

def tanh_activation(x):
    return np.tanh(x)

def initialize_weights(shape):
    return np.random.uniform(-0.5, 0.5, shape)

def compute_error(target, output):
    return 0.5 * (target - output) ** 2 

# Define the structure of the network
input_size = 2 
hidden_size = 2 
output_size = 1  

# Initialize weights and biases
W1 = initialize_weights((input_size, hidden_size))
b1 = np.array([0.5, 0.5])
W2 = initialize_weights((hidden_size, output_size))
b2 = np.array([0.7, 0.7])  

# Example input
X = np.array([0.3, -0.2])
target = np.array([0.1, -0.3])

# Forward pass
hidden_input = np.dot(X, W1) + b1
hidden_output = tanh_activation(hidden_input)
output_input = np.dot(hidden_output, W2) + b2
output = tanh_activation(output_input)

# Compute error
error = compute_error(target, output)

# Print results
print("Hidden Layer Input:", hidden_input)
print("Hidden Layer Output:", hidden_output)
print("Output Layer Input:", output_input)
print("Final Output:", output)
print("Error:",error)



