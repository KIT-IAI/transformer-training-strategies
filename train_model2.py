import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

from dataset.electricity_dataset import ElectricityDataset
from dataset.transformer_dataset import TransformerDataset
from dataset.multivariate_transformer_dataset import MultivariateTransformerDataset
from transformer_network import TimeSeriesTransformer
from transformer.informer import Informer


DEVICE = "cuda"


def evaluate_model(model, data_loader: DataLoader, n_batches=-1):
    model.eval()
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    total_mse = total_mae = 0
    if n_batches == -1:
        n_batches = len(data_loader)
    for i, (x_enc, x_dec, y) in tqdm(enumerate(data_loader), total=n_batches):
        x_enc, x_dec, y = x_enc.to(DEVICE), x_dec.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            output = model(x_enc, x_dec)
        total_mse += mse_criterion(output, y)
        total_mae += mae_criterion(output, y)
        if i + 1 == n_batches:
            break
    mean_mse = total_mse.cpu().numpy() / len(data_loader)
    mean_mae = total_mae.cpu().numpy() / len(data_loader)
    return {"mse": mean_mse,
            "mae": mean_mae}


def main(args):
    INPUT_LENGTH = args.input_length
    HORIZON = args.horizon
    N_WORKERS = 1
    SEED = args.seed

    MULTIVARIATE = args.mv
    IN_FEATURES = 330 if MULTIVARIATE else 10
    OUT_DIMENSIONS = 321 if MULTIVARIATE else 1
    CLIENT = args.client

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = 100
    PATIENCE = 10
    VALIDATE_EVERY_K_STEPS = 10000
    VALIDATION_BATCHES = -1
    GAMMA = 0.8
    MAX_GRADIENT_NORM = 1.0
    WARMUP_STEPS = 1000

    ENCODER_LAYERS = args.encoder_layers
    DECODER_LAYERS = args.decoder_layers
    D_MODEL = args.d_model
    CONV_FILTER_WIDTH = args.conv_filter_width
    MAX_POOLING = args.max_pooling

    TRAINING_TIME_FILE = "training_times.txt"

    if SEED is not None:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

    if args.model_name is None:
        model_name = "informer" if args.informer else "transformer"
        if MULTIVARIATE:
            model_name += "_mv"
        model_name += f"_h{HORIZON}_in{INPUT_LENGTH}_enc{ENCODER_LAYERS}_dec{DECODER_LAYERS}_dim{D_MODEL}" \
                      f"_bs{BATCH_SIZE}_lr{LEARNING_RATE}"
        if CONV_FILTER_WIDTH is not None:
            model_name += f"_conv{CONV_FILTER_WIDTH}"
        if MAX_POOLING is not None:
            model_name += f"_mp{MAX_POOLING}"
        if CLIENT is not None:
            model_name += f"_client{CLIENT}"
        if args.seed is not None:
            model_name += f"_seed{SEED}"
    else:
        model_name = args.model_name
    model_path = "saved_models/" + model_name

    electricity_dataset = ElectricityDataset(dataset=args.dataset, column=CLIENT)

    train_loads, train_features = electricity_dataset.get_training_data()
    dev_loads, dev_features = electricity_dataset.get_validation_data()
    test_loads, test_features = electricity_dataset.get_test_data()

    if MULTIVARIATE:
        DatasetType = MultivariateTransformerDataset
    else:
        DatasetType = TransformerDataset

    training_dataset = DatasetType(train_loads, train_features, INPUT_LENGTH, HORIZON)
    validation_dataset = DatasetType(dev_loads, dev_features, INPUT_LENGTH, HORIZON)
    test_dataset = DatasetType(test_loads, test_features, INPUT_LENGTH, HORIZON)
    print(len(training_dataset), len(validation_dataset), len(test_dataset))

    training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
    validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)

    if not args.test:
        if args.informer:
            model = Informer(input_features_count=IN_FEATURES, d_model=D_MODEL, n_heads=8, e_layers=ENCODER_LAYERS,
                             d_layers=DECODER_LAYERS, d_ff=D_MODEL, dropout=0.1, attn="full")
        else:
            model = TimeSeriesTransformer(d_model=D_MODEL, input_features_count=IN_FEATURES,
                                          num_encoder_layers=ENCODER_LAYERS, num_decoder_layers=DECODER_LAYERS,
                                          dim_feedforward=D_MODEL, dropout=0.1, attention_heads=8,
                                          conv_filter_width=CONV_FILTER_WIDTH, max_pooling=MAX_POOLING,
                                          output_dimensions=OUT_DIMENSIONS)
        model = model.to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)

        best_validation_loss = np.inf
        epochs_without_improvement = 0
        early_stopping = False
        step_i = 0
        lr_decay_start = False

        start_time = time.time()

        for epoch in range(1, EPOCHS + 1):
            if early_stopping:
                break

            print(f"EPOCH {epoch}")
            model.train()
            epoch_loss = 0

            for batch_i, (x_enc, x_dec, y) in tqdm(enumerate(training_data_loader), total=len(training_data_loader)):
                if step_i < WARMUP_STEPS:
                    lr = (step_i + 1) * LEARNING_RATE / WARMUP_STEPS
                    for g in optimizer.param_groups:
                        g["lr"] = lr
                x_enc, x_dec, y = x_enc.to(DEVICE), x_dec.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                prediction = model(x_enc, x_dec)
                loss = criterion(prediction, y)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=MAX_GRADIENT_NORM)
                optimizer.step()
                epoch_loss += loss.detach()
                step_i += 1

                if batch_i + 1 == len(training_data_loader) or (batch_i + 1) % VALIDATE_EVERY_K_STEPS == 0:
                    print("training loss:", epoch_loss.cpu().numpy() / (batch_i + 1))
                    validation_result = evaluate_model(model, validation_data_loader, n_batches=VALIDATION_BATCHES)
                    print(validation_result)
                    validation_loss = validation_result["mse"]
                    print("validation loss:", validation_loss)

                    if validation_loss < best_validation_loss:
                        torch.save(model, model_path)
                        print("model saved at", model_path)
                        best_validation_loss = validation_loss
                        epochs_without_improvement = -1

                        #test_result = evaluate_model(model, test_data_loader)
                        #print(test_result)
                        #print("test loss:", test_result["mse"])

                    if step_i > WARMUP_STEPS:
                        if lr_decay_start:
                            scheduler.step()
                        else:
                            lr_decay_start = True
                    print(f"learning rate set to {optimizer.param_groups[0]['lr']}")

                    epochs_without_improvement += 1
                    if epochs_without_improvement == PATIENCE:
                        print("early stopping")
                        early_stopping = True
                        break

        training_time = time.time() - start_time

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        with open(TRAINING_TIME_FILE, "a") as training_time_file:
            training_time_file.write(f"{model_name} {dt_string} {training_time}\n")

    else:
        model = torch.load(model_path)
        # print(model)
        test_result = evaluate_model(model, test_data_loader)
        print(test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--in", dest="input_length", type=int, required=False, default=168)
    parser.add_argument("--lr", dest="learning_rate", type=float, required=False, default=0.0001)
    parser.add_argument("--bs", dest="batch_size", type=int, required=False, default=128)
    parser.add_argument("--enc", dest="encoder_layers", type=int, required=False, default=3)
    parser.add_argument("--dec", dest="decoder_layers", type=int, required=False, default=3)
    parser.add_argument("--d_model", type=int, required=False, default=128)
    parser.add_argument("--conv", dest="conv_filter_width", type=int, required=False, default=None)
    parser.add_argument("--max_pooling", type=int, required=False, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--name", dest="model_name", type=str, required=False, default=None)
    parser.add_argument("--informer", action="store_true")
    parser.add_argument("--mv", action="store_true")
    parser.add_argument("--seed", type=int, required=False, default=None)
    parser.add_argument("--client", type=int, required=False, default=None)
    args = parser.parse_args()
    main(args)
