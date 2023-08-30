import argparse
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
from tqdm import tqdm

from dataset.electricity_dataset import ElectricityDataset
from linear_baseline import get_x_y, unscale
from trainable_parameters import num_trainable_parameters


DEVICE = "cuda"


class MLPRegressor(nn.Module):
    def __init__(self, input_dimensions: int, output_dimensions: int, n_layers: int, n_units: int):
        super(MLPRegressor, self).__init__()
        self.layers = nn.ModuleList()
        dim = input_dimensions
        for layer in range(n_layers):
            self.layers.append(nn.Linear(in_features=dim, out_features=n_units))
            dim = n_units
        self.layers.append(nn.Linear(in_features=dim, out_features=output_dimensions))

    def forward(self, X):
        for layer in self.layers[:-1]:
            X = torch.relu(layer(X))
        X = self.layers[-1](X)
        return X


class FlatDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float, device=DEVICE)
        self.Y = torch.tensor(Y, dtype=torch.float, device=DEVICE)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return int(self.X.shape[0])


METRICS = ["mae", "mse", "unscaled_mae", "unscaled_mse"]


def evaluate(model, data_loader: DataLoader, mean: Optional[float], scale: Optional[float]):
    model.eval()
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    results = {metric: 0 for metric in METRICS}
    for x, y in data_loader:
        with torch.no_grad():
            prediction = model(x)
        results["mse"] += mse_criterion(y, prediction)
        results["mae"] += mae_criterion(y, prediction)
        if mean is not None and scale is not None:
            y = y.cpu().numpy()
            prediction = prediction.cpu().numpy()
            y_unscaled = unscale(y, mean, scale)
            prediction_unscaled = unscale(prediction, mean, scale)
            results["unscaled_mse"] += mean_squared_error(y_unscaled, prediction_unscaled)
            results["unscaled_mae"] += mean_absolute_error(y_unscaled, prediction_unscaled)
    for metric in results:
        if isinstance(results[metric], torch.Tensor):
            results[metric] = results[metric].cpu().numpy()
        results[metric] = results[metric] / len(data_loader)
    if mean is not None:
        results["nmae"] = results["unscaled_mae"] / mean * 100
    return results


def main(args):
    random.seed(1)

    DATASET_NAME = args.dataset

    if DATASET_NAME == "electricity":
        n_buildings = 321
        TOTAL_BUILDINGS = 321
    else:
        n_buildings = 299
        TOTAL_BUILDINGS = 299

    GLOBAL = args.glob

    INPUT_LENGTH = 168
    OUTPUT_LENGTH = args.horizon
    N_LAYERS = args.layers
    N_UNITS = args.units

    BATCH_SIZE = 128
    EPOCHS = 1
    LEARNING_RATE = 0.001
    PATIENCE = 10
    GAMMA = 0.5

    dataset = ElectricityDataset(DATASET_NAME)
    training_data, training_features = dataset.get_training_data()
    validation_data, validation_features = dataset.get_validation_data()
    test_data, test_features = dataset.get_test_data()

    building_test_results = []
    total_training_time = 0

    if GLOBAL:
        buildings = [None]
    else:
        buildings = list(range(TOTAL_BUILDINGS))
        if n_buildings < TOTAL_BUILDINGS:
            random.shuffle(buildings)
            buildings = buildings[:n_buildings]
            buildings = sorted(buildings)

    for building in buildings:
        print(f"* building {building} *")

        if building is None:  # global model
            X_train, Y_train, X_valid, Y_valid, X_test, Y_test = [], [], [], [], [], []
            for column in range(TOTAL_BUILDINGS):
                x_train, y_train = get_x_y(training_data[:, column], training_features, input_length=INPUT_LENGTH,
                                           output_length=OUTPUT_LENGTH)
                x_valid, y_valid = get_x_y(validation_data[:, column], validation_features, input_length=INPUT_LENGTH,
                                           output_length=OUTPUT_LENGTH)
                x_test, y_test = get_x_y(test_data[:, column], test_features, input_length=INPUT_LENGTH,
                                         output_length=OUTPUT_LENGTH)
                X_train.append(x_train)
                Y_train.append(y_train)
                X_valid.append(x_valid)
                Y_valid.append(y_valid)
                X_test.append(x_test)
                Y_test.append(y_test)
            X_train = np.concatenate(X_train, axis=0)
            Y_train = np.concatenate(Y_train, axis=0)
            X_valid = np.concatenate(X_valid, axis=0)
            Y_valid = np.concatenate(Y_valid, axis=0)
            X_test = np.concatenate(X_test, axis=0)
            Y_test = np.concatenate(Y_test, axis=0)
        else:
            X_train, Y_train = get_x_y(training_data[:, building], training_features, input_length=INPUT_LENGTH,
                                       output_length=OUTPUT_LENGTH)
            X_valid, Y_valid = get_x_y(validation_data[:, building], validation_features, input_length=INPUT_LENGTH,
                                       output_length=OUTPUT_LENGTH)
            X_test, Y_test = get_x_y(test_data[:, building], test_features, input_length=INPUT_LENGTH,
                                     output_length=OUTPUT_LENGTH)
        train_dataset = FlatDataset(X_train, Y_train)
        validation_dataset = FlatDataset(X_valid, Y_valid)
        test_dataset = FlatDataset(X_test, Y_test)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        dataset.scaler: StandardScaler
        data_mean = None if building is None else dataset.scaler.mean_[building]
        data_scale = None if building is None else dataset.scaler.scale_[building]

        n_features = X_train.shape[1]
        model = MLPRegressor(n_features, OUTPUT_LENGTH, n_layers=N_LAYERS, n_units=N_UNITS)

        print(f"{n_features} features")
        print(f"model has {num_trainable_parameters(model)} parameters")

        model = model.to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)

        best_validation_loss = np.inf
        epochs_without_improvement = 0
        start_time = time.time()

        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0

            model.train()
            for x, y in tqdm(train_dataloader):
                optimizer.zero_grad()
                prediction = model(x)
                loss = criterion(y, prediction)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach()
            scheduler.step()

            validation_results = evaluate(model, validation_dataloader, data_mean, data_scale)

            print(f"epoch {epoch}, training loss = {epoch_loss / len(train_dataloader):.4f}, "
                  f"validation loss = {validation_results['mse']:.4f}, lr = {optimizer.param_groups[0]['lr']}")

            if validation_results["mse"] < best_validation_loss:
                best_validation_loss = validation_results["mse"]
                epochs_without_improvement = 0
                test_results = evaluate(model, test_dataloader, data_mean, data_scale)
                print("test:", test_results)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement == PATIENCE:
                print("early stopping")
                break

        total_training_time += time.time() - start_time
        building_test_results.append(test_results)

    mean_results = {}
    for metric in building_test_results[0]:
        mean_result = np.mean([result[metric] for result in building_test_results])
        print(f"{metric}: {mean_result}")
        mean_results[metric] = mean_result
    print(f"total training time: {total_training_time:.2f}")

    with open("mlp_results.txt", "a") as file:
        file.write(args)
        file.write(" ")
        file.write(str(mean_results))
        file.write(f" training_time={total_training_time:.2f}")
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--in", dest="input_length", type=int, required=False, default=168)
    parser.add_argument("--layers", type=int, required=False, default=2)
    parser.add_argument("--units", type=int, required=False, default=1024)
    parser.add_argument("--glob", action="store_true")
    args = parser.parse_args()
    main(args)
