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

# get pm from config
pm: float = config["denoising"]["pm"]

# use minimizer if asked
if config["optimizer"] != "None" and config["optimizer"] != "":
    # randomize the dataset
    dataset = parser.randomize_data(dataset, config["data_random_seed"])
    # train with minimize
    auto_encoder.train_minimizer(parser.add_noise_dataset(dataset, pm), dataset, config["trust"], config["use_trust"], config["optimizer"], config["optimizer_iter"], config["optimizer_fev"])
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
            auto_encoder.train(parser.add_noise(data, pm), data, config["eta"])

        # apply the changes
        auto_encoder.update_w()

        # calculate error
        error: float = auto_encoder.error(parser.add_noise_dataset(dataset, pm), dataset, config["trust"], config["use_trust"])
        if error < config["error_threshold"]:
            break

        if ep % 50 == 0:
            print(f'Iteration {ep}, error {err}')

        # add error to list
        ep_list.append(ep)
        err_list.append(err)
    
    # plot error vs epoch
    utils.plot_values(ep_list, 'epoch', err_list, 'error', sci_y=False)

# labels for printing (use with full_dataset)
labels: [] = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']

# check simple input how it performs
letter = full_dataset[0]
letter_act: np.ndarray = auto_encoder.activation(letter)
print(letter)
print(letter_act)
print(np.around(letter_act))

noisy_letter: np.ndarray = parser.add_noise(letter, pm)
noisy_letter_act: np.ndarray = auto_encoder.activation(noisy_letter)
print(noisy_letter)
print(noisy_letter_act)
print(np.around(noisy_letter_act))

# hold execution
utils.hold_execution()
