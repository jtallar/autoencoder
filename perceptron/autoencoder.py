import numpy as np
import complex as cp


# uses 2 complex perceptron
class AutoEncoder(object):

    def __init__(self, activation_function, activation_function_derived,
                 layout: [int], input_dim: int, latent_dim: int):
        self.encoder = cp.ComplexPerceptron(activation_function, activation_function_derived,
                                            layout, input_dim, latent_dim)

        self.decoder = cp.ComplexPerceptron(activation_function, activation_function_derived,
                                            layout.reverse(), latent_dim, input_dim)
