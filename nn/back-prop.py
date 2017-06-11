import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## Calculate output error
error = target - output

# Calculate error term for output layer
output_error_term = error * output * (1- output)
# print("output_error_term ", output_error_term)

# Calculate error term for hidden layer
h_prime = hidden_layer_output * (1 - hidden_layer_output)
# print("h_primte ", h_prime)
hidden_error_term = weights_hidden_output * output_error_term * h_prime
print("hidden_error_term ", hidden_error_term)
print("x ", x)
print("x[:,None] ", x[:,None])


# TODO: Calculate change in weights for hidden layer to output layer
# learning rate, times the output unit error, times the hidden unit activation value

delta_w_h_o = learnrate * output_error_term * hidden_layer_output


# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * hidden_error_term * x[:,None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
