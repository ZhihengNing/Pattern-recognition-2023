import json


def green(x):
    if type(x) is not str:
        x = str(x)
    return '\033[92m' + x + '\033[0m'


def blue(x):
    if type(x) is not str:
        x = str(x)
    return '\033[94m' + x + '\033[0m'


def red(x):
    if type(x) is not str:
        x = str(x)
    return '\033[91m' + x + '\033[0m'


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
