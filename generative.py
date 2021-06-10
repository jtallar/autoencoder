import json

import numpy as np

import extras.parser as parser
import extras.functions as functions
import perceptron.genautoencoder as ae

with open("config.json") as file:
    config = json.load(file)

# static non changeable vars
error_threshold: float = config["error_threshold"]

# read the files and get the dataset. There is no need to normalize data at this exercise
full_dataset, _ = parser.read_file(config["file"], config["system_threshold"])

# activation function and its derived
act_funcs = functions.get_activation_functions(config["system"], config["beta"])

# randomize dataset order. if seed is "" then it is not used
full_dataset = parser.randomize_data(full_dataset, config["data_random_seed"])

# extract the last % of the dataset
dataset, rest = parser.extract_subset(full_dataset, config["training_ratio"])

# initializes the auto-encoder
auto_encoder = ae.GenAutoEncoder(*act_funcs, config["layout"], len(dataset[0]), config["latent_dim"])

# randomize w if asked
if bool(config["randomize_w"]):
    auto_encoder.randomize_w(config["randomize_w_ref"])

# train auto-encoder
for _ in range(config["epochs"]):

    # train for this epoch
    for data in dataset:
        auto_encoder.train(data, data, config["eta"])

    # apply the changes
    auto_encoder.update_w()

    # calculate error
    error: float = auto_encoder.error(dataset, dataset, config["trust"])
    if error < config["error_threshold"]:
        break

# show latent space given the input
aux: [] = []
for data in dataset:
    aux.append(auto_encoder.activation(data))
latent_space: np.ndarray = np.ndarray(aux)

# generate a new something not from the dataset. Creates a new Z between the first two
new_latent_space: np.ndarray = np.sum([latent_space[0], latent_space[1]], axis=0)/2
new_letter: np.ndarray = auto_encoder.decoder.activation(new_latent_space, training=False)

