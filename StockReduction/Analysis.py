from math import sqrt

import torch
from matplotlib import pyplot as plt

import ModelPersistence
import TrainingUtil


def do_analysis(model_file, data_file, draw):
    m = ModelPersistence.unpickle_model(model_file, to_cpu=True)

    with open(f"data/{data_file}", 'rb') as infile:
        data = torch.load(infile, map_location='cpu')

    preds = TrainingUtil.generate_predictions(m, data)

    residuals = data.ys - preds
    avg_y = torch.mean(data.ys)

    r2 = 1 - (torch.sum(torch.square(residuals)).item() / torch.sum(torch.square(data.ys - avg_y)).item())

    mse = torch.mean(torch.square(residuals)).item()
    rmse = sqrt(mse)

    mae = torch.mean(torch.abs(residuals))

    print(f" R^2 of {MODEL_TO_ANALYZE}: {r2}")
    print(f"RMSE of {MODEL_TO_ANALYZE}: {rmse}")
    print(f" MAE of {MODEL_TO_ANALYZE}: {mae}")

    if draw:
        plt.scatter(data.ys[:4000], preds)
        plt.plot([-5000, 5000], [-5000, 5000], color='red')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    MODEL_TO_ANALYZE = "colab_19_2"
    TEST_DATA = "temp_test.pt"
    DRAW_PLOTS = False

    do_analysis(MODEL_TO_ANALYZE, TEST_DATA, DRAW_PLOTS)