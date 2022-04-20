
import numpy as np

# input_nodes is how many values per input, in this case 4
# weight is randomly assigned with input_nodes + 1 values for a bias
class percep:
    def __init__(self, input_nodes, learning_rate = 0.1):
        self.weight = np.random.randn(input_nodes + 1) / np.sqrt(input_nodes)
        self.learning_rate = learning_rate

# if x is positive returns 1 else returns 0
    def step_fun(self, x):
        return 1 if x > 0 else 0


# training function, requires input, answers, and number of epochs to run for
    def fit(self, input, answer, epochs=10):
        input = np.c_[input, np.ones((input.shape[0]))]
        for l in np.arange(0, epochs):
            for (x, target) in zip(input, answer):
# gets the dot product between the input and weight and then passes answer to step function for prediction
                p = self.step_fun(np.dot(x, self.weight))
                if p != target:
                    # determines the error _______________________________________
                    delta = p - target
                    # changes weights based on prediction
                    self.weight += -self.learning_rate * delta * x


    def prediction(self, input, add_bias = True):
        # turns input into a matrix
        input = np.atleast_2d(input)
        # check to see if the bias column should be added if yes, insert a column of 1's as the last entry in the matrix
        if add_bias:
            input = np.c_[input, np.ones((input.shape[0]))]
        # gets the dot product between the input and weight and then passes answer to step function for prediction
        return self.step_fun(np.dot(input, self.weight))