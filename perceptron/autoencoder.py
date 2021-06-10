import numpy as np
import complex as cp
import extras.functions as f


# uses 2 complex perceptron
class AutoEncoder(object):

    def __init__(self, activation_function, activation_function_derived,
                 layout: [int], data_dim: int, latent_dim: int):

        self.data_dim: int = data_dim
        self.latent_dim: int = latent_dim
        self.encoder = cp.ComplexPerceptron(activation_function, activation_function_derived,
                                            layout, data_dim, latent_dim, encoder=True)

        self.decoder = cp.ComplexPerceptron(activation_function, activation_function_derived,
                                            reversed(layout), latent_dim, data_dim, encoder=False)

    # performs the training on the auto-encoder
    def train(self, data_in: np.ndarray, data_out: np.ndarray, eta: float) -> None:
        self.activation(data_in, training=True)
        self.retro(data_out, eta)

    # propagates input along the encoder and decoder
    # returns always the output of the encoder (latent space input)
    def activation(self, init_input: np.ndarray, training: bool = False) -> np.ndarray:
        if not training:
            return self.encoder.activation(init_input, training=False)

        encoder_out: np.ndarray = self.encoder.activation(init_input, training=True)
        self.decoder.activation(encoder_out, training=True)
        return encoder_out

    # retro-propagates the difference with the expected out through the auto encoder
    # returns the latent space input on retro-propagation
    def retro(self, expected_out: np.ndarray, eta: float) -> (np.ndarray, np.ndarray):
        out_dim: int = len(expected_out)
        sup_w, sup_delta = self.decoder.retro(expected_out, eta, np.empty(out_dim), np.empty(out_dim))
        return sup_w, sup_delta

    # initially the weights (w) start with 0, initialize/change them
    def randomize_w(self, ref: float) -> None:
        self.encoder.randomize_w(ref)
        self.decoder.randomize_w(ref)

    # for epoch training updates each perceptron its weights
    def update_w(self) -> None:
        self.encoder.update_w()
        self.decoder.update_w()

    # calculates the error of the auto-encoder
    def error(self, data_in: np.ndarray, data_out: np.ndarray, trust: float, use_trust: bool) -> float:
        return np.sum(np.abs((data_out[:, 1:] -
                              f.discrete(self.activation(data_in[:, 1:]), trust, use_trust)) ** 2)) / 2
