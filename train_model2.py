import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset.electricity_dataset import ElectricityDataset
from dataset.transformer_dataset import TransformerDataset
from transformer_network import TimeSeriesTransformer


DEVICE = "cuda"


def evaluate_model(model, data_loader: DataLoader):
    model.eval()
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    total_mse = total_mae = 0
    for x_enc, x_dec, y in tqdm(data_loader):
        x_enc, x_dec, y = x_enc.to(DEVICE), x_dec.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            output = model(x_enc, x_dec)
        total_mse += mse_criterion(output, y)
        total_mae += mae_criterion(output, y)
    mean_mse = total_mse.cpu().numpy() / len(data_loader)
    mean_mae = total_mae.cpu().numpy() / len(data_loader)
    return {"mse": mean_mse,
            "mae": mean_mae}


def main(args):
    INPUT_LENGTH = 168
    HORIZON = args.horizon

    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 10

    model_name = f"transformer_h{HORIZON}"
    model_path = "saved_models/" + model_name

    electricity_dataset = ElectricityDataset()

    train_loads, train_features = electricity_dataset.get_training_data()
    dev_loads, dev_features = electricity_dataset.get_validation_data()
    test_loads, test_features = electricity_dataset.get_test_data()

    training_dataset = TransformerDataset(train_loads, train_features, INPUT_LENGTH, HORIZON)
    validation_dataset = TransformerDataset(dev_loads, dev_features, INPUT_LENGTH, HORIZON)
    test_dataset = TransformerDataset(test_loads, test_features, INPUT_LENGTH, HORIZON)
    print(len(training_dataset), len(validation_dataset), len(test_dataset))

    training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=BATCH_SIZE)
    validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = TimeSeriesTransformer(d_model=160, input_features_count=10, num_encoder_layers=2,
                                  num_decoder_layers=2, dim_feedforward=160, dropout=0.1, attention_heads=8)
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    n_batches = len(training_data_loader)
    best_validation_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"EPOCH {epoch}")
        model.train()
        epoch_loss = 0

        for x_enc, x_dec, y in tqdm(training_data_loader):
            x_enc, x_dec, y = x_enc.to(DEVICE), x_dec.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            prediction = model(x_enc, x_dec)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
        print("training loss:", epoch_loss.cpu().numpy() / n_batches)

        validation_result = evaluate_model(model, validation_data_loader)
        print(validation_result)
        validation_loss = validation_result["mse"]
        print("validation loss:", validation_loss)

        test_result = evaluate_model(model, test_data_loader)
        print(test_result)
        print("test loss:", test_result["mse"])

        if validation_loss < best_validation_loss:
            torch.save(model, model_path)
            print("model saved at", model_path)
            best_validation_loss = validation_loss
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == PATIENCE:
                print("early stopping")
                break
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    main(args)
