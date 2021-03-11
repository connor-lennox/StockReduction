from math import ceil

from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

import DataReader
import TrainingUtil


def train_model(model, data, epochs=5, batch_size=64):
    train_loader = DataLoader(data, batch_size=batch_size)

    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    num_batches = ceil(len(data) / batch_size)
    for i in range(epochs):
        loss_sum = 0
        for batch_index, sample in enumerate(train_loader):
            print(f"\rEpoch {i+1} " + TrainingUtil.progress_string(batch_index, num_batches), end="")
            xs = sample[0]  # .to(device)
            ys = sample[1]  # .to(device)

            optim.zero_grad()
            result = model(xs).flatten()

            loss = criterion(result, ys)
            loss_sum += loss.item() / num_batches

            loss.backward()
            optim.step()
        print(f"\rEpoch {i+1} " + TrainingUtil.progress_string(num_batches, num_batches), end="")

        print(f"\n\tLoss={loss_sum}")


if __name__ == '__main__':
    m = torch.nn.Sequential(
        torch.nn.Linear(850, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1)
    )

    d_train, d_test = DataReader.construct_dataset(800)

    train_model(m, d_train, epochs=200, batch_size=64)

    predictions = TrainingUtil.generate_predictions(m, d_train)
    residuals = d_train.ys - predictions
    plt.scatter(d_train.ys, residuals, color='blue')

    predictions = TrainingUtil.generate_predictions(m, d_test)
    residuals = d_test.ys - predictions
    plt.scatter(d_test.ys, residuals, color='orange')

    plt.axhline(0, color='red')
    # plt.plot([-5000, 5000], [-5000, 5000], color='red')

    plt.show()
