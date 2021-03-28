import torch
from torch.utils.data import DataLoader


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def progress_string(done, total, bar_length=16, include_count=True):
    num_filled = int(done / total * bar_length)
    num_unfilled = bar_length - num_filled
    bar = "[" + "=" * num_filled
    if num_unfilled != 0:
        bar += '>'
    bar += " " * (num_unfilled-1) + "]"
    if include_count:
        bar += f" [{done}/{total}]"
    return bar


def generate_predictions(model, data):
    predictions = []

    print("Generating Predictions...")

    loader = DataLoader(data, batch_size=1)
    for i, elem in enumerate(loader):
        print("\r" + progress_string(i, len(loader)), end="")
        predictions.append(model(elem[0]).cpu().item())
    print("\r" + progress_string(len(loader), len(loader)))

    predictions = torch.tensor(predictions).to(DEVICE)
    return predictions
