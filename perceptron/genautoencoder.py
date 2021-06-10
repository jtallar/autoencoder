import numpy as np
import complex as cp


# uses 2 complex perceptron
class GenAutoEncoder(object):

    def __init__(self, activation_function, activation_function_derived,
                 layout: [int], data_dim: int, latent_dim: int):

        self.data_dim: int = data_dim
        self.latent_dim: int = latent_dim
        self.encoderMean = cp.ComplexPerceptron(activation_function, activation_function_derived,
                                                layout, data_dim, latent_dim, encoder=True)

        self.encoderStd = cp.ComplexPerceptron(activation_function, activation_function_derived,
                                               layout, data_dim, latent_dim, encoder=True)

        self.decoder = cp.ComplexPerceptron(activation_function, activation_function_derived,
                                            reversed(layout), latent_dim, data_dim, encoder=False)

    # performs the training on the auto-encoder
    def train(self, data_in: np.ndarray, data_out: np.ndarray, eta: float) -> None:
        self.activation(data_in)
        self.retro(data_out, eta)

    # propagates input along the encoder and decoder
    # returns always the output of the encoder (latent space input)
    def activation(self, init_input: np.ndarray) -> np.ndarray:

        # propagate and return mean and std
        mean_out: np.ndarray = self.encoderMean.activation(init_input, training=True)
        std_out: np.ndarray = self.encoderStd.activation(init_input, training=True)

        # generate z
        z: np.ndarray = np.random.uniform(size=self.latent_dim) * std_out + mean_out

        # propagate z on decoder
        self.decoder.activation(z, training=True)
        return z

    # retro-propagates the difference with the expected out through the auto encoder
    # returns the latent space input on retro-propagation
    def retro(self, expected_out: np.ndarray, eta: float) -> (np.ndarray, np.ndarray):

        # retro-propagate on decoder
        out_dim: int = len(expected_out)
        sup_w, sup_delta = self.decoder.retro(expected_out, eta, np.empty(out_dim), np.empty(out_dim))

        # TODO connect here to the encoders

        return sup_w, sup_delta

    # initially the weights (w) start with 0, initialize/change them
    def randomize_w(self, ref: float) -> None:
        self.encoderStd.randomize_w(ref)
        self.encoderMean.randomize_w(ref)
        self.decoder.randomize_w(ref)

    # for epoch training updates each perceptron its weights
    def update_w(self) -> None:
        self.encoderStd.update_w()
        self.encoderMean.update_w()
        self.decoder.update_w()

    # calculates the error of the auto-encoder
    def error(self, data_in: np.ndarray, data_out: np.ndarray) -> float:
        return np.sum(np.abs((data_out[:, 1:] - self.activation(data_in[:, 1:])) ** 2)) / 2
