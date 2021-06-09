import multiprocessing.pool
import numpy as np
import simple as sp


class ComplexPerceptron(object):

    def __init__(self, activation_function, activation_function_derived,
                 layout: [int], input_dim: int, output_dim: int, encoder: bool = False):

        self.act_func = activation_function
        self.act_func_der = activation_function_derived
        self.network = None
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.__init_network(layout, encoder)

    # propagates input along the entire network
    # in case of training, saves  the input for later computation on retro propagation
    # returns the final activation value
    def activation(self, init_input: np.ndarray, training: bool = False) -> np.ndarray:
        if len(init_input) != self.in_dim:
            raise SystemExit('Bad input size for activation on complex perceptron')

        activation_values = init_input
        for layer in self.network:
            pool = multiprocessing.pool.ThreadPool(processes=len(layer))
            activation_values = pool.map(lambda s_p: s_p.activation(activation_values, training=training), layer)
            activation_values = np.transpose(np.asarray(activation_values))

        return activation_values

    # retro-propagates the error of the network given the true input
    # this method starts with an empty sup_w and sup_delta
    def simple_retro(self, expected_out: np.ndarray, eta: float, epoch: bool) -> (np.ndarray, np.ndarray):
        return self.retro(expected_out, eta, epoch, np.empty(self.out_dim), np.empty(self.out_dim))

    # retro-propagates the error of the network given the true input
    # takes the given suo_w and sup_delta as initial values
    def retro(self, expected_out: np.ndarray, eta: float, epoch: bool,
              init_sup_w: np.ndarray, init_sup_delta: np.ndarray) -> (np.ndarray, np.ndarray):
        if len(expected_out) != self.out_dim:
            raise SystemExit('Bad input size for retro on complex perceptron')

        sup_w: np.ndarray = init_sup_w
        sup_delta: np.ndarray = init_sup_delta
        for layer in reversed(self.network):
            pool = multiprocessing.pool.ThreadPool(processes=len(layer))
            sup_w, sup_delta = zip(*pool.map(lambda s_p: s_p.retro(expected_out, sup_w, sup_delta, eta, epoch), layer))
            # convert tuples to lists (used in the next layer)
            sup_w = np.asarray(sup_w)
            sup_delta = np.asarray(sup_delta)

        return sup_w, sup_delta

    # calculate the error on the perceptron
    def error(self, inp: np.ndarray, out: np.ndarray, error_enhance: bool = False) -> float:
        if not error_enhance:
            return np.sum(np.abs((out - self.activation(inp)) ** 2)) / 2

        return np.sum((1 + out) * np.log(np.divide((1 + out), (1 + self.activation(inp)))) / 2 +
                      (1 - out) * np.log(np.divide((1 - out), (1 - self.activation(inp)))) / 2)

    # resets the w to a randomize range if desired for the entire network
    # if randomize is false, then does nothing
    def randomize_w(self, ref: float) -> None:
        for layer in self.network:
            pool = multiprocessing.pool.ThreadPool(processes=len(layer))
            pool.map(lambda s_p: s_p.randomize_w(ref), layer)

    # for epoch training updates each w with its accum
    def update_w(self) -> None:
        for layer in self.network:
            pool = multiprocessing.pool.ThreadPool(processes=len(layer))
            pool.map(lambda s_p: s_p.update_w(epoch=True), layer)

    def __str__(self) -> str:
        out: str = "CPerceptron=("
        for i, layer in enumerate(self.network):
            out += f"\nlayer {i}=" + str(layer)
        return out + ")"

    def __repr__(self) -> str:
        out: str = "CPerceptron=("
        for i, layer in enumerate(self.network):
            out += f"\nlayer {i}=" + str(layer)
        return out + ")"

    # private methods

    # initializes the entire network of perceptron given a layout
    def __init_network(self, hidden_layout: [int], full_hidden: bool = False) -> None:
        # the final amount of perceptron depends on expected output dimension
        layout: np.ndarray = np.append(np.array(hidden_layout, dtype=int), self.out_dim)

        # initialize the length of the network
        self.network = np.empty(shape=len(layout), dtype=np.ndarray)

        # for each level, get its count of perceptron
        for level in range(len(layout)):

            # initialize (empty) level with its amount of perceptron
            self.network[level] = np.empty(shape=layout[level], dtype=sp.SimplePerceptron)

            # the dimension of the next level is set from the previous or the input data
            dim: int = layout[level - 1] if level != 0 else self.in_dim

            # if its a full hidden network all layers are hidden
            # if not check if it is the last layer
            hidden: bool = full_hidden or level != (len(layout) - 1)

            # create the corresponding amount of perceptron
            for index in range(layout[level]):
                # for each index and level, create the corresponding perceptron
                self.network[level][index] = \
                    sp.SimplePerceptron(self.act_func, self.act_func_der, dim, hidden, index)
