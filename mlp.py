import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

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


def evaluate(model, data_loader: DataLoader, mean: float, scale: float):
    model.eval()
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    results = {metric: 0 for metric in METRICS}
    for x, y in data_loader:
        with torch.no_grad():
            prediction = model(x)
        results["mse"] += mse_criterion(y, prediction)
        results["mae"] += mae_criterion(y, prediction)
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
    results["nmae"] = results["unscaled_mae"] / mean * 100
    return results


if __name__ == "__main__":
    random.seed(1)

    DATASET_NAME = sys.argv[1]

    if DATASET_NAME == "electricity":
        n_buildings = 321
        TOTAL_BUILDINGS = 321
    else:
        n_buildings = 299
        TOTAL_BUILDINGS = 299

    INPUT_LENGTH = 168
    OUTPUT_LENGTH = int(sys.argv[2])
    N_LAYERS = int(sys.argv[3])
    N_UNITS = int(sys.argv[4])

    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 10
    GAMMA = 0.1

    dataset = ElectricityDataset(DATASET_NAME)
    training_data, training_features = dataset.get_training_data()
    validation_data, validation_features = dataset.get_validation_data()
    test_data, test_features = dataset.get_test_data()

    building_test_results = []
    total_training_time = 0

    buildings = list(range(TOTAL_BUILDINGS))
    if n_buildings < TOTAL_BUILDINGS:
        random.shuffle(buildings)
        buildings = buildings[:n_buildings]
        buildings = sorted(buildings)

    for building in buildings:
        print(f"* building {building} *")

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
        data_mean = dataset.scaler.mean_[building]
        data_scale = dataset.scaler.scale_[building]

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

        for epoch in range(1, EPOCHS):
            epoch_loss = 0

            model.train()
            for x, y in train_dataloader:
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
        file.write(str(sys.argv[1:]))
        file.write(" ")
        file.write(str(mean_results))
        file.write("\n")
