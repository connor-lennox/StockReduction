import torch


def pickle_model(model, filename):
    with open("TrainedModels" + '\\' + filename + '.model', 'wb+') as file:
        torch.save(model, file)


def unpickle_model(filename):
    with open("TrainedModels" + '\\' + filename, 'rb') as infile:
        return torch.load(infile)
