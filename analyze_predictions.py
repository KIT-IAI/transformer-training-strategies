import sys
import random
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm

from dataset import UCIDataset


if __name__ == "__main__":
    uci_dataset = UCIDataset()

    path = "results/predictions.pkl"
    with open(path, "rb") as f:
        predictions = pickle.load(f)
    buildings = list(predictions.keys())
    print(buildings)

    BUILDING = "MT_004"

    print(predictions[BUILDING])
    n_predictions = len(predictions[BUILDING])
    horizon = len(predictions[BUILDING][0])
    print(n_predictions)
    print(horizon)

    first_prediction_time = uci_dataset.df.index[-n_predictions - horizon]
    print(first_prediction_time)

    if "--plots" in sys.argv:
        while True:
            building = random.choice(buildings)
            time_step = random.randint(0, n_predictions)
            prediction_time = first_prediction_time + timedelta(hours=time_step)
            model_input = uci_dataset.df[building][prediction_time - timedelta(hours=167):prediction_time]
            prediction = predictions[building][time_step]
            truth = uci_dataset.df[building][prediction_time + timedelta(hours=1):prediction_time + timedelta(hours=horizon)]
            x = truth.index
            plt.plot(model_input, label="history")
            plt.plot(truth.index, prediction, label="prediction")
            plt.plot(truth, label="ground truth")
            plt.title(f"{building}, {str(prediction_time)}")
            plt.legend()
            plt.show()
    else:
        mae_array = []
        mse_array = []
        mape_array = []
        for building in tqdm(buildings):
            for time_step in range(n_predictions):
                prediction_time = first_prediction_time + timedelta(hours=time_step)
                prediction = np.array(predictions[building][time_step])
                truth = np.array(uci_dataset.df[building][
                                 prediction_time + timedelta(hours=1):prediction_time + timedelta(hours=horizon)])
                mae_array.append(mean_absolute_error(truth, prediction))
                mse_array.append(mean_squared_error(truth, prediction))
                mape_array.append(mean_absolute_percentage_error(truth, prediction))
        total_mae = np.mean(mae_array)
        total_mse = np.mean(mse_array)
        total_mape = np.mean(mape_array)
        print(f"MAE:  {total_mae:.4f}")
        print(f"MSE:  {total_mse:.4f}")
        print(f"MAPE: {total_mape:.4f}")
