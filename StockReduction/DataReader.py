import numpy as np
import pandas
import torch
from torch.utils.data import Dataset

import FeatureExtraction


_DATA_PATH = "data/chessData.csv"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _eval_to_int(eval_in):
    if eval_in.startswith("\ufeff"):
        eval_in = eval_in[1:]
    if eval_in.startswith("#"):
        eval_in = eval_in[1:]
    return float(eval_in)


def read_data(data_path, num_rows=10000):
    df = pandas.read_csv(data_path, nrows=num_rows)

    df['Evaluation'] = df['Evaluation'].map(_eval_to_int)
    df['Features'] = df['FEN'].map(FeatureExtraction.features_from_fen)

    return torch.vstack(tuple(df['Features'].values)).to(DEVICE), torch.tensor(df['Evaluation'].values).float().to(DEVICE)


# def construct_dataset(num_rows=10000, train_split=.8):
#     data = EvalDataset(*read_data(num_rows))
#     train_samples = int(num_rows * train_split)
#     train, test = torch.utils.data.random_split(data, [train_samples, len(data)-train_samples])
#     return train, test

def construct_dataset(data_path=_DATA_PATH, num_rows=10000, train_split=.8):
    xs, ys = read_data(data_path, num_rows)

    indices = np.arange(num_rows)
    np.random.shuffle(indices)
    train_samples = int(num_rows * train_split)

    return EvalDataset(xs[indices[:train_samples]], ys[indices[:train_samples]]), \
           EvalDataset(xs[indices[train_samples:]], ys[indices[train_samples:]])


class EvalDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]

    def __len__(self):
        return self.xs.shape[0]


if __name__ == '__main__':
    test_dataset = construct_dataset(num_rows=200000)
    with open("data/training_data.pt", 'wb+') as outfile:
        torch.save(test_dataset[0], outfile)
    with open("data/testing_data.pt", 'wb+') as outfile:
        torch.save(test_dataset[1], outfile)
