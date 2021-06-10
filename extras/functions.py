import numpy as np
import math


def get_activation_functions(name: str, beta: float, complex_error_enhance: bool):
    act_funcs_dict = {
        "linear": lambda x: x,
        "tanh": lambda x: np.tanh(x * beta),
        "exp": lambda x: 1 / (1 + math.e ** (-2 * beta * x)),
    }

    # the derived activation func, if it cant be, then return always 1
    act_funcs_der_dict = {
        "linear": lambda x: 1,
        "tanh": lambda x: beta * (1 - act_funcs_dict["tanh"](x) ** 2),
        "exp": lambda x: 2 * beta * act_funcs_dict["exp"](x) * (1 - act_funcs_dict["exp"](x))
    }

    # the derived activation func but with the new error calculus, if it cant be, then return always 1
    act_funcs_der_2_dict = {
        "tanh": lambda x: beta,
    }

    # in case it is complex and a desired to make better error
    if complex_error_enhance:
        if name != "tanh":
            print("Cannot enhance error on activation function different than tanh")
            exit(1)
        return act_funcs_dict[name], act_funcs_der_2_dict[name]

    return act_funcs_dict[name], act_funcs_der_dict[name]


def adaptive_eta(new_delta_error: float, delta_error: float, delta_error_dec: bool, k: int, eta: float,
                 dec_k: int, inc_k: int, a, b):
    # the delta error changes direction, reset k (-1 +1 = 0)
    if (delta_error > new_delta_error) != delta_error_dec:
        k = -1

    if k == dec_k and delta_error_dec:
        eta += a * eta
        print(f"New update on adaptive eta to: {eta}, after {k} consecutive increase iterations")
        k = int(dec_k / 2)

    if k == inc_k and not delta_error_dec:
        eta -= b * eta
        print(f"New update on adaptive eta to: {eta}, after {k} consecutive decrease iterations")
        k = int(inc_k / 2)

    delta_error_dec = delta_error > new_delta_error
    delta_error = new_delta_error
    k += 1

    return delta_error, delta_error_dec, k, eta
