import pickle


def dump_pickle(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(path):
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj
