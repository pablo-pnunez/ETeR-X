# -*- coding: utf-8 -*-
import argparse
import pickle


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def to_pickle(path, name, data):
    with open(path + name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_pickle(path, name):
    with open(path + name, 'rb') as handle:
        data = pickle.load(handle)
    return data


def print_e(text):
    print(BColors.FAIL + str("[ERROR] ") + str(text) + BColors.ENDC)


def print_w(text, title=True):
    title = "[WARNING] " if title else ""
    print(BColors.WARNING + title + str(text) + BColors.ENDC)


def print_g(text, title=True):
    title = "[INFO] " if title else ""
    print(BColors.OKGREEN + title + str(text) + BColors.ENDC)


def print_b(text, bold=False):
    if bold:
        print(BColors.BOLD + BColors.OKBLUE + str(text) + BColors.ENDC + BColors.ENDC)
    else:
        print(BColors.OKBLUE + str(text) + BColors.ENDC)


def parse_cmd_args():
    """Obtener argumentos por linea de comandos"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-dst', type=str, help="Dataset")
    parser.add_argument('-sst', type=str, help="Subset")

    parser.add_argument('-gpu', type=int, help="GPU")
    parser.add_argument('-esp', type=int, help="Early stop patience")
    parser.add_argument('-sd', type=int, help="Seed")
    parser.add_argument('-ep', type=int, help="Epoch number")
    parser.add_argument('-lr', type=float, help="Learning rate")
    parser.add_argument('-bs', type=int, help="Batch size")
    parser.add_argument('-stg', type=int, help="stage")
    parser.add_argument('-mv', type=str, help="Model version")
    parser.add_argument('-mn', type=str, help="Model name")
    parser.add_argument('-bownws', type=int, help="BOW n words")

    args = parser.parse_args()
    return args

