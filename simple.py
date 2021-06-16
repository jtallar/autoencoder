import json

import numpy as np

import extras.parser as parser
import extras.functions as functions
import extras.utils as utils
import perceptron.autoencoder as ae

with open("config.json") as file:
    config = json.load(file)

# static non changeable vars
error_threshold: float = config["error_threshold"]

# read the files and get the dataset. There is no need to normalize data at this exercise
full_dataset, _ = parser.read_file(config["file"], config["system_threshold"])

# activation function and its derived
act_funcs = functions.get_activation_functions(config["system"], config["beta"])

# normalize data
if config["normalize"]:
    full_dataset = parser.normalize_data(full_dataset)

# extract the last % of the dataset
dataset, rest = parser.extract_subset(full_dataset, config["training_ratio"])

# initializes the auto-encoder
auto_encoder = ae.AutoEncoder(*act_funcs, config["mid_layout"], len(dataset[0]), config["latent_dim"],
                              config["momentum"], config["alpha"])

# randomize w if asked
if bool(config["randomize_w"]):
    auto_encoder.randomize_w(config["randomize_w_ref"], config["randomize_w_by_len"])

# initialize plotter
utils.init_plotter()

# use minimizer if asked
if config["optimizer"] != "None" and config["optimizer"] != "":
    # randomize the dataset
    dataset = parser.randomize_data(dataset, config["data_random_seed"])
    # train with minimize
    auto_encoder.train_minimizer(dataset, dataset, config["trust"], config["use_trust"], config["optimizer"], config["optimizer_iter"], config["optimizer_fev"])
    # plot error vs opt step
    utils.plot_values(range(len(auto_encoder.opt_err)), 'opt step', auto_encoder.opt_err, 'error', sci_y=False)
else:
    # vars for plotting
    ep_list = []
    err_list = []

    # train auto-encoder
    for ep in range(config["epochs"]):

        # randomize the dataset everytime
        dataset = parser.randomize_data(dataset, config["data_random_seed"])

        # train for this epoch
        for data in dataset:
            auto_encoder.train(data, data, config["eta"])

        # apply the changes
        auto_encoder.update_w()

        # calculate error
        err = auto_encoder.error(dataset, dataset, config["trust"], config["use_trust"])

        if err < config["error_threshold"]:
            break

        if ep % 50 == 0:
            print(f'Iteration {ep}, error {err}')

        # add error to list
        ep_list.append(ep)
        err_list.append(err)

    # plot error vs epoch
    utils.plot_values(ep_list, 'epoch', err_list, 'error', sci_y=False)

# TODO: Algo de aca abajo me tira overflow si uso minimizer
print(auto_encoder.flatten_weights())

# labels for printing (use with full_dataset)
labels: [] = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']

# show latent space given the input
aux: [] = []
for data in full_dataset:
    aux.append(auto_encoder.activation_to_latent_space(data))
latent_space: np.ndarray = np.array(aux)
# plot latent space
utils.plot_latent_space(latent_space, labels, -1, 1)

# generate a new letter not from the dataset. Creates a new Z between the first two
new_latent_space: np.ndarray = np.sum([latent_space[0], latent_space[1]], axis=0)/2
new_letter: np.ndarray = auto_encoder.activation_from_latent_space(new_latent_space)

letter: np.ndarray = auto_encoder.activation(full_dataset[0])
print(dataset[0])
print(letter)

# hold execution
utils.hold_execution()
