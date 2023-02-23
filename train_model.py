import argparse
import datetime

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import math
import pickle

from uci_dataset import UCIDataset, time_delta_hours, trim_leading_zeros
from transformer_network import TimeSeriesTransformer


INPUT_TIME_STEPS = 7 * 24
FORECASTING_HORIZON = 24


def get_samples(time_series: pd.Series, calendar_features: np.array):
    first_prediction_datetime = time_series.index[0]
    n = len(time_series)
    data = np.column_stack((time_series, calendar_features[-n:]))
    X = sliding_window_view(data[:-FORECASTING_HORIZON],
                            window_shape=INPUT_TIME_STEPS,
                            axis=0).transpose(0, 2, 1)
    Y = sliding_window_view(data[INPUT_TIME_STEPS:],
                            window_shape=FORECASTING_HORIZON,
                            axis=0).transpose(0, 2, 1)
    prediction_datetimes = pd.date_range(start=first_prediction_datetime, periods=X.shape[0], freq="H")
    return X, Y, prediction_datetimes


def main(args):
    autoformer_datapoints = time_delta_hours(datetime.datetime(2015, 1, 1), datetime.datetime(2012, 1, 1))
    n_test = int(autoformer_datapoints * 0.2)
    n_validation = int(autoformer_datapoints * 0.1)

    uci_dataset = UCIDataset()

    X_train, Y_train = [], []
    X_valid, Y_valid = [], []
    X_test, Y_test = [], []

    N_BUILDINGS = uci_dataset.number_of_buildings()
    buildings = list(uci_dataset.buildings())[1:(N_BUILDINGS + 1)]

    model_path = "saved_models/transformer"

    scalers = {}
    for building in buildings:
        time_series = trim_leading_zeros(uci_dataset.get_data(building))
        n_training = len(time_series) - n_test - n_validation
        scaler = StandardScaler()
        scalers[building] = scaler
        scaler.fit(np.array(time_series[:n_training]).reshape(-1, 1))
        time_series.iloc[:] = scaler.transform(np.array(time_series).reshape(-1, 1)).flatten()
        X, Y, prediction_times = get_samples(time_series, uci_dataset.calendar_features)

        n_train = X.shape[0] - n_test - n_validation
        X_train.append(X[:n_train])
        Y_train.append(Y[:n_train])
        X_valid.append(X[n_train:-n_test])
        Y_valid.append(Y[n_train:-n_test])
        X_test.append(X[-n_test:])
        Y_test.append(Y[-n_test:])

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    print(X_train.shape, Y_train.shape)
    X_valid = np.concatenate(X_valid, axis=0)
    Y_valid = np.concatenate(Y_valid, axis=0)
    print(X_valid.shape, Y_valid.shape)
    X_test = np.concatenate(X_test, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)
    print(X_test.shape, Y_test.shape)

    epochs = 100
    patience = 10
    batch_size = 32
    device = "cuda"

    criterion = nn.MSELoss()

    if args.training:
        model = TimeSeriesTransformer(d_model=160, input_features_count=X_train.shape[2], num_encoder_layers=2,
                                      num_decoder_layers=2, dim_feedforward=160, dropout=0.0, attention_heads=8)
        model = model.to(device)
        print(model)

        example_indices = list(range(X_train.shape[0]))
        n_batches = len(example_indices) // batch_size

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        best_validation_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            print(f"EPOCH {epoch}")
            random.shuffle(example_indices)
            batch_start = 0
            epoch_loss = 0
            for batch_i in tqdm(range(n_batches)):
                batch_end = batch_start + batch_size
                batch_indices = example_indices[batch_start:batch_end]
                X_enc_batch = torch.tensor(X_train[batch_indices], dtype=torch.float).to(device)
                X_dec_batch = torch.tensor(Y_train[batch_indices], dtype=torch.float).to(device)
                y_batch = X_dec_batch[:, :, 0].clone()
                X_dec_batch[:, :, 0] = 0

                optimizer.zero_grad()
                output = model.forward(X_enc_batch, X_dec_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                batch_start = batch_end
                epoch_loss += loss.detach()
            print("training loss:", epoch_loss.cpu().numpy() / n_batches)

            validation_loss = 0
            n_batches_dev = math.ceil(X_valid.shape[0] / batch_size)
            batch_start = 0
            for batch_i in tqdm(range(n_batches_dev)):
                batch_end = batch_start + batch_size
                X_enc_batch = torch.tensor(X_valid[batch_start:batch_end], dtype=torch.float).to(device)
                X_dec_batch = torch.tensor(Y_valid[batch_start:batch_end], dtype=torch.float).to(device)
                y_batch = X_dec_batch[:, :, 0].clone()
                X_dec_batch[:, :, 0] = 0
                with torch.no_grad():
                    output = model.forward(X_enc_batch, X_dec_batch)
                loss = criterion(output, y_batch)
                validation_loss += loss
                batch_start = batch_end
            validation_loss = validation_loss.cpu().numpy() / n_batches_dev
            print("validation loss:", validation_loss)

            if validation_loss < best_validation_loss:
                torch.save(model, model_path)
                print("model saved at", model_path)
                best_validation_loss = validation_loss
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == patience:
                    print("early stopping")
                    break
            scheduler.step()
    else:
        print("* testing *")
        model = torch.load(model_path)
        model: TimeSeriesTransformer
        n_batches = math.ceil(X_test.shape[0] / batch_size)
        batch_start = 0
        test_loss = 0
        predictions_per_building = X_test.shape[0] // N_BUILDINGS
        print(predictions_per_building, "predictions per building")
        test_prediction_times = prediction_times[-predictions_per_building:]
        collected_predictions = {building: [] for building in buildings}
        for batch_i in tqdm(range(n_batches)):
            batch_end = batch_start + batch_size
            X_enc_batch = torch.tensor(X_test[batch_start:batch_end], dtype=torch.float).to(device)
            X_dec_batch = torch.tensor(Y_test[batch_start:batch_end], dtype=torch.float).to(device)
            y_batch = X_dec_batch[:, :, 0].clone()
            X_dec_batch[:, :, 0] = 0
            with torch.no_grad():
                output = model(X_enc_batch, X_dec_batch)
                loss = criterion(output, y_batch)
                test_loss += loss.detach()
            output = output.cpu().numpy()
            for i in range(output.shape[0]):
                index = batch_start + i
                #print(index, predictions_per_building, len(buildings))
                building = buildings[index // predictions_per_building]
                prediction = output[i]
                prediction = scalers[building].inverse_transform(prediction.reshape(-1, 1)).flatten()
                collected_predictions[building].append(prediction)
            batch_start = batch_end
        print(test_loss.cpu().numpy() / n_batches)

        for building in buildings:
            collected_predictions[building] = np.stack(collected_predictions[building])
        with open("results/predictions.pkl", "wb") as f:
            pickle.dump(collected_predictions, f)

    # TODO:
    # test error
    # metriken
    # ! validation error !
    # ! early stopping !
    # example predictions
    # attention scores
    # baseline(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    main(args)
