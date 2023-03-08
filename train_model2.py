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
    INPUT_LENGTH = args.input_length
    HORIZON = args.horizon
    N_WORKERS = 1

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = 100
    PATIENCE = 5

    ENCODER_LAYERS = args.encoder_layers
    DECODER_LAYERS = args.decoder_layers
    D_MODEL = args.d_model
    CONV_FILTER_WIDTH = args.conv_filter_width
    MAX_POOLING = args.max_pooling

    if args.model_name is None:
        model_name = f"transformer_h{HORIZON}_in{INPUT_LENGTH}_enc{ENCODER_LAYERS}_dec{DECODER_LAYERS}_dim{D_MODEL}" \
                     f"_bs{BATCH_SIZE}_lr{LEARNING_RATE}"
        if CONV_FILTER_WIDTH is not None:
            model_name += f"_conv{CONV_FILTER_WIDTH}"
        if MAX_POOLING is not None:
            model_name += f"_mp{MAX_POOLING}"
    else:
        model_name = args.model_name
    model_path = "saved_models/" + model_name

    electricity_dataset = ElectricityDataset()

    train_loads, train_features = electricity_dataset.get_training_data()
    dev_loads, dev_features = electricity_dataset.get_validation_data()
    test_loads, test_features = electricity_dataset.get_test_data()

    training_dataset = TransformerDataset(train_loads, train_features, INPUT_LENGTH, HORIZON)
    validation_dataset = TransformerDataset(dev_loads, dev_features, INPUT_LENGTH, HORIZON)
    test_dataset = TransformerDataset(test_loads, test_features, INPUT_LENGTH, HORIZON)
    print(len(training_dataset), len(validation_dataset), len(test_dataset))

    training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
    validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)

    if not args.test:
        model = TimeSeriesTransformer(d_model=D_MODEL, input_features_count=10, num_encoder_layers=ENCODER_LAYERS,
                                      num_decoder_layers=DECODER_LAYERS, dim_feedforward=D_MODEL, dropout=0.1,
                                      attention_heads=8, conv_filter_width=CONV_FILTER_WIDTH, max_pooling=MAX_POOLING)
        model = model.to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        n_batches = len(training_data_loader)
        best_validation_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(1, EPOCHS + 1):
            print(f"EPOCH {epoch}")
            model.train()
            epoch_loss = 0

            for batch_i, (x_enc, x_dec, y) in tqdm(enumerate(training_data_loader), total=len(training_data_loader)):
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

    else:
        model = torch.load(model_path)
        # print(model)
        test_result = evaluate_model(model, test_data_loader)
        print(test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--in", dest="input_length", type=int, required=False, default=168)
    parser.add_argument("--lr", dest="learning_rate", type=float, required=False, default=0.001)
    parser.add_argument("--bs", dest="batch_size", type=int, required=False, default=128)
    parser.add_argument("--enc", dest="encoder_layers", type=int, required=False, default=2)
    parser.add_argument("--dec", dest="decoder_layers", type=int, required=False, default=2)
    parser.add_argument("--d_model", type=int, required=False, default=160)
    parser.add_argument("--conv", dest="conv_filter_width", type=int, required=False, default=False)
    parser.add_argument("--max_pooling", type=int, required=False, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--name", dest="model_name", type=str, required=False, default=None)
    args = parser.parse_args()
    main(args)
