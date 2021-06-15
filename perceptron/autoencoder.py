import numpy as np
import extras.functions as f
import perceptron.complex as cp


# uses 2 complex perceptron
class AutoEncoder(object):

    def __init__(self, activation_function, activation_function_derived,
                 layout: [int], data_dim: int, latent_dim: int,
                 momentum: bool = False, mom_alpha: float = 0.9):
        self.data_dim: int = data_dim
        self.latent_dim: int = latent_dim

        encoder_layout: [] = layout.copy()
        encoder_layout.append(latent_dim)
        self.encoder = cp.ComplexPerceptron(activation_function, activation_function_derived, encoder_layout,
                                            data_dim, full_hidden=True, momentum=momentum, mom_alpha=mom_alpha)

        decoder_layout: [] = layout[::-1]
        decoder_layout.append(data_dim)
        self.decoder = cp.ComplexPerceptron(activation_function, activation_function_derived, decoder_layout,
                                            latent_dim, full_hidden=False, momentum=momentum, mom_alpha=mom_alpha)

    # performs the training on the auto-encoder
    def train(self, data_in: np.ndarray, data_out: np.ndarray, eta: float) -> None:
        self.activation(data_in, training=True)
        self.retro(data_out, eta)

    # propagates input along the encoder and decoder
    # returns always the output
    def activation(self, init_input: np.ndarray, training: bool = False) -> np.ndarray:
        encoder_out: np.ndarray = self.encoder.activation(init_input, training=training)
        return self.decoder.activation(encoder_out, training=training)

    # returns the activation out from the latent space
    def activation_to_latent_space(self, init_input: np.ndarray) -> np.ndarray:
        return self.encoder.activation(init_input, training=False)

    # returns the activation value of the decoder from the latent space (generate things)
    def activation_from_latent_space(self, init_input: np.ndarray) -> np.ndarray:
        return self.decoder.activation(init_input, training=False)

    # retro-propagates the difference with the expected out through the auto encoder
    # returns the input on retro-propagation
    def retro(self, expected_out: np.ndarray, eta: float) -> (np.ndarray, np.ndarray):
        out_dim: int = len(expected_out)
        sup_w, sup_delta = self.decoder.retro(expected_out, eta, np.empty(out_dim), np.empty(out_dim))
        return self.encoder.retro(expected_out, eta, sup_w, sup_delta)

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
        act: np.ndarray = f.discrete(self.activation(data_in)[:, 1:], trust, use_trust)
        out: np.ndarray = data_out[:, 1:]

        return (np.linalg.norm(out - act) ** 2) / len(out)
