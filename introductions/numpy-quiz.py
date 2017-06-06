# Use the numpy library
import numpy as np


def prepare_inputs(inputs):
    input_array = np.array(inputs, ndmin=2)
    inputs_minus_min = input_array - input_array.min()
    inputs_div_max = inputs_minus_min / inputs_minus_min.max()

    # return the three arrays we've created
    return input_array, inputs_minus_min, inputs_div_max


def multiply_inputs(m1, m2):
    r1, c1 = m1.shape
    r2, c2 = m2.shape
    if not (c1 == r2 or c2 == r1):
        return False

    if c1 == r2:
        return m1.dot(m2)
    else:
        return m2.dot(m1)

def find_mean(values):
    return np.array(values).mean()


input_array, inputs_minus_min, inputs_div_max = prepare_inputs([-1,2,7])
print("Input as Array: {}".format(input_array))
print("Input minus min: {}".format(inputs_minus_min))
print("Input  Array: {}".format(inputs_div_max))

print("Multiply 1:\n{}".format(multiply_inputs(np.array([[1,2,3],[4,5,6]]), np.array([[1],[2],[3],[4]]))))
print("Multiply 2:\n{}".format(multiply_inputs(np.array([[1,2,3],[4,5,6]]), np.array([[1],[2],[3]]))))
print("Multiply 3:\n{}".format(multiply_inputs(np.array([[1,2,3],[4,5,6]]), np.array([[1,2]]))))

print("Mean == {}".format(find_mean([1,3,4])))
