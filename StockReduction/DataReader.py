import numpy as np
import pandas
import torch
from torch.utils.data import Dataset

import FeatureExtraction


_DATA_PATH = "data/chessData.csv"


def _eval_to_int(eval_in):
    if str(eval_in).startswith("#"):
        return float(str(eval_in)[1:])
    return float(eval_in)


def read_data(num_rows=10000):
    df = pandas.read_csv(_DATA_PATH, nrows=num_rows)

    df['Evaluation'] = df['Evaluation'].map(_eval_to_int)
    df['Features'] = df['FEN'].map(FeatureExtraction.features_from_fen)

    return torch.vstack(tuple(df['Features'].values)), torch.tensor(df['Evaluation'].values).float()


# def construct_dataset(num_rows=10000, train_split=.8):
#     data = EvalDataset(*read_data(num_rows))
#     train_samples = int(num_rows * train_split)
#     train, test = torch.utils.data.random_split(data, [train_samples, len(data)-train_samples])
#     return train, test

def construct_dataset(num_rows=10000, train_split=.8):
    xs, ys = read_data(num_rows)

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
    test_dataset = EvalDataset(*read_data(num_rows=1000))
    print(test_dataset[0])
    print(len(test_dataset))
