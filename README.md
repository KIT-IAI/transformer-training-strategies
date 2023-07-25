# Transformer Training Strategies for Forecasting Multiple Load Time Series

## Installation

Tested with Python 3.10.

```
pip install torch --index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Usage

Example call:

```
python train_model.py --dataset electricity --horizon 24
```

Arguments:

| Argument  | Explanation                                         |
|-----------|-----------------------------------------------------|
| `--dataset` | Choose the dataset from _electricity_ or _ausgrid_. |
| `--horizon` | Forecast horizon in hours.                          |
| `--in`    | Input length.                                       |
| `--test`  | Evaluates a trained model on the test data.         |
| `--name` | Specifies the model name.                           |
| `--mv` | Multivariate training strategy.                     |
| `--seed` | Sets the seed.                                      |
| `--client` | Local model for the specified client ID (integer, beginning with 0). |
| `--lstm` | Sets model architecture to LSTM. |

To list more arguments, including model hyperparameters, use `python train_model.py --help`.

## Comparison methods

### Persistence baseline

```
python recency_baseline.py [dataset] [offset]
```

Set the first argument to _electricity_ or _ausgrid_.
The second argument is 168 (1 week) for the 24 and 168 hours horizons, and 720 (1 month) for the 720 hours horizon.

### Local linear models

```
python linear_baseline.py [dataset] [horizon]
```

### Local multi-layer perceptrons

```
python mlp.py [dataset] [horizon]
```

### Models from related work

The code from [FEDformer](https://github.com/MAZiqing/FEDformer), [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) 
and [PatchTST](https://github.com/yuqinie98/PatchTST) was used to evaluate the models from related work.

### Additional results

Mean squared error, first three columns are for the _electricity_ dataset, last three for _ausgrid_.

| Model                 | 24h       | 96h       | 720h    | 24h       | 96h       | 720h      |
|-----------------------|-----------|-----------|---------|-----------|-----------|-----------|
| Informer (MV)         | 0.305     | 0.320     | 0.384   | 0.835     | 0.870     | 0.891     |
| Autoformer (MV)       | 0.166     | 0.201     | 0.254   | 0.732     | 0.754     | 0.800     |
| FEDformer (MV)        | _0.164_   | _0.183_   | _0.231_ | _0.707_   | _0.741_   | _0.798_   |
| LSTM (MV)             | 0.311     | 0.314     | 0.329   | 0.841     | 0.851     | 0.835     |
| Transformer (MV)      | 0.267     | 0.292     | 0.300   | 0.853     | 0.841     | 0.800     |
| Persistence (L)       | 0.214     | 0.214     | 0.490   | 1.163     | 1.163     | 1.353     |
| Linear regression (L) | 0.103     | _0.133_   | __0.194__ | _0.604_   | _0.659_   | _0.725_   |
| MLP (L)               | _0.097_   | 0.135     | 0.212   | 0.627     | 0.693     | 0.749     |
| LSTM (L)              | 0.146     | 0.169     | 0.234   | 0.647     | 0.693     | 0.751     |
| Transformer (L)       | 0.144     | 0.186     | 0.271   | 0.721     | 0.776     | 0.802     |
| LTSF-Linear (G)       | 0.110     | 0.140     | 0.203   | 0.598     | 0.647     | 0.705     |
| PatchTST (G)          | 0.094     | 0.129     | _0.197_ | __0.576__ | __0.641__ | __0.704__ |
| LSTM (G)              | 0.106     | 0.139     | 0.202   | 0.603     | 0.667     | 0.719     |
| Transformer (G)       | __0.090__ | __0.127__ | 0.219   | 0.599     | 0.665     | 0.716     |

## Paper

If you use this repository in your research, please cite our [paper](https://arxiv.org/abs/2306.10891).

```
@article{hertel2023transformer,
  title={Transformer Training Strategies for Forecasting Multiple Load Time Series},
  author={Hertel, Matthias and Beichter, Maximilian and Heidrich, Benedikt and Neumann, Oliver and Sch{\"a}fer, Benjamin and Mikut, Ralf and Hagenmeyer, Veit},
  journal={arXiv preprint arXiv:2306.10891},
  year={2023}
}
```