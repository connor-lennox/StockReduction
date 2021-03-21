import torch


def pickle_model(model, filename):
    with open("TrainedModels" + '\\' + filename + '.model', 'wb+') as file:
        torch.save(model, file)


def unpickle_model(filename, to_cpu=True):
    with open("TrainedModels" + '\\' + filename + '.model', 'rb') as infile:
        if to_cpu:
            return torch.load(infile, map_location=torch.device('cpu'))
        else:
            return torch.load(infile)
