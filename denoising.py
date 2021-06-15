import json

import numpy as np

import extras.parser as parser
import extras.functions as functions
import perceptron.autoencoder as ae

with open("config.json") as file:
    config = json.load(file)

# static non changeable vars
error_threshold: float = config["error_threshold"]

# read the files and get the dataset. There is no need to normalize data at this exercise
full_dataset, _ = parser.read_file(config["file"], config["system_threshold"])

# activation function and its derived
act_funcs = functions.get_activation_functions(config["system"], config["beta"])

# extract the last % of the dataset
dataset, rest = parser.extract_subset(full_dataset, config["training_ratio"])

# initializes the auto-encoder
auto_encoder = ae.AutoEncoder(*act_funcs, config["mid_layout"], len(dataset[0]), config["latent_dim"],
                              config["momentum"], config["alpha"])

# randomize w if asked
if bool(config["randomize_w"]):
    auto_encoder.randomize_w(config["randomize_w_ref"])

# train auto-encoder
pm: float = config["denoising"]["pm"]
for _ in range(config["epochs"]):

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

# labels for printing (use with full_dataset)
labels: [] = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']

# check simple input how it performs
letter: np.ndarray = auto_encoder.activation(parser.add_noise(dataset[0], pm))

