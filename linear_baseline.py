import sys
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from dataset.electricity_dataset import ElectricityDataset


def get_x_y(time_series, features, input_length, output_length):
    X = sliding_window_view(time_series[:-output_length], window_shape=input_length)
    if features is not None:
        X = np.concatenate((X, features[input_length:-(output_length - 1)]), axis=1)
    Y = sliding_window_view(time_series[input_length:], window_shape=output_length)
    return X, Y


def unscale(time_series, mean, scale):
    return time_series * scale + mean


if __name__ == "__main__":
    METRICS = ["mae", "mse", "unscaled_mae", "unscaled_mse", "nmae"]

    DATASET_NAME = sys.argv[1]
    dataset = ElectricityDataset(DATASET_NAME)

    n_buildings = dataset.get_n_columns()
    input_length = 336
    output_length = int(sys.argv[2])

    training_data, training_features = dataset.get_training_data()
    test_data, test_features = dataset.get_test_data()

    results = {metric: [] for metric in METRICS}
    total_training_time = 0

    for building in range(n_buildings):
        print(f"* building {building} *")
        X_train, Y_train = get_x_y(training_data[:, building], training_features, input_length, output_length)
        print(X_train.shape, Y_train.shape)
        model = LinearRegression()
        start_time = time.time()
        model.fit(X_train, Y_train)
        total_training_time += time.time() - start_time
        X_test, Y_test = get_x_y(test_data[:, building], test_features, input_length, output_length)
        prediction = model.predict(X_test)
        print(X_test.shape, Y_test.shape, prediction.shape)
        mae = mean_absolute_error(Y_test, prediction)
        mse = mean_squared_error(Y_test, prediction)
        results["mae"].append(mae)
        results["mse"].append(mse)
        print(f"mae = {mae} (mean = {np.mean(results['mae'])})")
        print(f"mse = {mse} (mean = {np.mean(results['mse'])})")

        dataset.scaler: StandardScaler
        mean = dataset.scaler.mean_[building]
        scale = dataset.scaler.scale_[building]
        unscaled_Y = unscale(Y_test, mean, scale)
        unscaled_prediction = unscale(prediction, mean, scale)
        unscaled_mae = mean_absolute_error(unscaled_Y, unscaled_prediction)
        unscaled_mse = mean_squared_error(unscaled_Y, unscaled_prediction)
        nmae = unscaled_mae / mean * 100
        results["unscaled_mae"].append(unscaled_mae)
        results["unscaled_mse"].append(unscaled_mse)
        results["nmae"].append(nmae)
        print(f"unscaled mae = {unscaled_mae} (mean = {np.mean(results['unscaled_mae'])})")
        print(f"unscaled mse = {unscaled_mse} (mean = {np.mean(results['unscaled_mse'])})")
        print(f"nmae = {nmae} (mean = {np.mean(results['nmae'])})")
    print("* total *")
    for metric in METRICS:
        print(f"{metric} = {np.mean(results[metric])}")
    print(f"total training time = {total_training_time:.2f}")
