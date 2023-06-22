import numpy as np

from dataset.electricity_dataset import ElectricityDataset


if __name__ == "__main__":
    OFFSET = 168

    dataset = ElectricityDataset("ausgrid")
    test_data, _ = dataset.get_test_data()
    print(test_data.shape)

    prediction = test_data[:-OFFSET, :]
    ground_truth = test_data[OFFSET:, :]
    residuals = ground_truth - prediction
    print(residuals)
    print(residuals.shape)
    mae_building = np.mean(np.abs(residuals), axis=0)
    print(mae_building)
    print(mae_building.shape)
    print(np.mean(mae_building))
    mse_building = np.mean(np.square(residuals), axis=0)
    print(np.mean(mse_building))
