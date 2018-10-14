import numpy as np

class Sigmoid:
    """
    This class models an artificial neuron with sigmoid activation function.
    """

    def __init__(self, weights = np.array([1])):
        """
        Initialize weights based on input arguments. Note that no type-checking
        is being performed here for simplicity of code.
        """
        self.weights = weights

        # NOTE: You do not need to worry about these two attribues for this
        # programming quiz, but these will be useful for if you want to create
        # a network out of these sigmoid units!
        self.last_input = 0 # strength of last input
        self.delta      = 0 # error signal

    def dot(self, value):
        """
        Takes in @param values, a list of numbers equal to length of weights.
        @return the output of a sigmoid unit with given inputs based on unit
        weights.
        """

        # YOUR CODE HERE

        # First calculate the strength of the input signal.
        result = 1 / (1 + np.exp(-value))
        # TODO: Modify strength using the sigmoid activation function and
        # return as output signal.
        # HINT: You may want to create a helper function to compute the
        #   logistic function since you will need it for the update function.

        return result

    def derivative(self, value):
        f = 1/(1+np.exp(-value))
        df = f * (1 - f)
        return df