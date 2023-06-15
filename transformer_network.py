from typing import List

import math

import torch
from torch import nn, Tensor

from transformer.transformer import Transformer


class ValueEmbedding(nn.Module):

    def __init__(self, d_model: int, time_series_features: int, filter_width: int):
        super(ValueEmbedding, self).__init__()
        self.filter_width = filter_width
        if filter_width is None or filter_width == 0:
            self.projection = nn.Linear(time_series_features, d_model)
        else:
            self.projection = nn.Conv1d(time_series_features, d_model, kernel_size=filter_width, padding="same")

    def forward(self, x: Tensor) -> Tensor:
        """
        Creates from the given tensor a linear projection.

        :param x: the input tensor to project, shape: [batch_size, sequence_length, features]
        :return: the projected tensor of shape: [batch_size, sequence_length, model_dimension]
        """
        if self.filter_width is None or self.filter_width == 0:
            return self.projection(x)
        else:
            x = x.transpose(1, 2)
            x = self.projection(x)
            return x.transpose(2, 1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe.require_grad = False
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Creates a positional encoding for the given tensor.

        :param x: the tensor for which the pe is created, shape: [batch_size, sequence_length, model_dimension
        :return: positional encoding of dimension [1, sequence_length, model_dimension]
        """
        return self.pe[:, :x.size(1), :]


class TotalEmbedding(nn.Module):

    def __init__(self, d_model: int, value_features: int, time_features: int, dropout: float, conv_filter_width: int):
        super(TotalEmbedding, self).__init__()

        self.value_embedding = ValueEmbedding(d_model, value_features + time_features, filter_width=conv_filter_width)
        self.positional_encoding = PositionalEncoding(d_model)

        self.linear_embedding_weight = nn.Linear(2, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_embedding_weight.weight.data.fill_(1)  # initialize with 1 --> training is more stable

    def forward(self, x: Tensor):
        """
        Projects the given tensor x on the model_dimension (in the last dimension) and combines this with a positional
        encoding (PE). The PE is added with learned weights to the projected x tensor. Dropout is applied on the final
        result.

        :param x: tensor of dimension [Batch_Size, Sequence_Length, Features]
        :return: the embedded value of shape: [Batch_Size, Sequence_Length, model_dimension]
        """
        value_embedded = self.value_embedding(x)
        pe = self.positional_encoding(x).repeat(x.shape[0], 1, 1)

        # add the embedded tensor and positional encoding
        return self.dropout(self.linear_embedding_weight.weight[0][0] * value_embedded
                            + self.linear_embedding_weight.weight[0][1] * pe)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model: int, input_features_count: int, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, dropout: float, attention_heads: int, conv_filter_width: int = None,
                 max_pooling: int = None, output_dimensions: int = 1):
        super().__init__()
        self.output_dimensions = output_dimensions
        self.transformer = Transformer(d_model, attention_heads, num_encoder_layers, num_decoder_layers,
                                       batch_first=True, dim_feedforward=dim_feedforward, dropout=dropout,
                                       max_pooling=max_pooling)

        self.projection = nn.Linear(d_model, output_dimensions, bias=True)
        self.encoder_embedding = TotalEmbedding(d_model, 1, input_features_count - 1, dropout, conv_filter_width)
        self.decoder_embedding = TotalEmbedding(d_model, 1, input_features_count - 1, dropout, conv_filter_width)
        self.relu = nn.ReLU()

    def forward(self, x_enc, x_dec, src_mask=None, tgt_mask=None):
        """
        Executes the model for the given input. The raw encoder and decoder input is embedded to the model's dimension
        and a positional encoding added. Then, the transformer part with the encoder and decoder is executed and the
        prediction is generated with a linear layer.

        :param x_enc: the raw input for the encoder, shape: [batch_size, seq_enc_length, features]
        :param x_dec: the raw input for the decoder, shape: [batch_size, seq_dec_length, features]
        :param src_mask: mask for the encoder (optional, is normally not needed)
        :param tgt_mask: mask for the decoder (optional, normally needed)
        :returns: the predictions of shape: [batch_size, seq_dec_length]
        """
        enc_embedding = self.encoder_embedding(x_enc)
        dec_embedding = self.decoder_embedding(x_dec)
        out = self.transformer(enc_embedding, dec_embedding, src_mask=src_mask, tgt_mask=tgt_mask)
        out = self.projection(self.relu(out))
        if self.output_dimensions == 1:
            out = out[:, :, 0]
        return out

    def get_cross_attention_scores(self):
        return average_attention_scores([layer.multihead_attn.attention_weights
                                         for layer in self.transformer.decoder.layers])

    def get_self_attention_scores(self):
        return average_attention_scores([layer.self_attn.attention_weights
                                         for layer in self.transformer.decoder.layers])


def average_attention_scores(attention_scores: List[torch.Tensor]):
    return torch.mean(torch.stack(attention_scores), dim=0)
