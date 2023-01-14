import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt

softmax = torch.nn.Softmax(dim=1)


class DefaultParameters:
    DEFAULT_NUM_HEADS = 8
    DEFAULT_WORD_WIDTH = 512
    DEFAULT_PROJECTION_WIDTH = 64
    DEFAULT_SCALE_FACTOR = 100
    DEFAULT_FFNN_HIDDEN_LAYER_WIDTH = 2048
    DEFAULT_MAX_WORDS = 25


NUM_HEADS = 8
WORD_WIDTH = 512
PROJECTION_WIDTH = 64
SCALE_FACTOR = 100
FFNN_HIDDEN_LAYER_WIDTH = 2048


def encoder_stack(num_encoders, w_o):
    encoders = np.array(list(map(lambda x: Encoder(w_o,
                                                   DefaultParameters.DEFAULT_NUM_HEADS,
                                                   DefaultParameters.DEFAULT_WORD_WIDTH), range(num_encoders))))
    return nn.Sequential(*encoders)


class SelfAttentionLayer(nn.Module):
    def __init__(self, w_q, w_k, w_v):
        super(SelfAttentionLayer, self).__init__()
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v

    def forward(self, words):
        return self.attention_scores(qkvs(words, self.w_q, self.w_k, self.w_v))

    def attention_scores(self, qkvs):
        Q, K, V = list(range(3))
        return torch.matmul(
            softmax(torch.matmul(qkvs[Q], torch.transpose(qkvs[K], 0, 1)) / math.sqrt(qkvs[Q].shape[1])), qkvs[V])


class MultiheadedAttention(nn.Module):
    def __init__(self, w_o, num_heads=DefaultParameters.DEFAULT_NUM_HEADS):
        super(MultiheadedAttention, self).__init__()
        self.w_o = w_o
        self.attention_layers = list(map(lambda x: SelfAttentionLayer(W_Q, W_K, W_V), range(num_heads)))

    def forward(self, input):
        # Concatenating gives [num_words x num_heads * projection_width]
        attention_vectors = list(map(lambda attention_layer: attention_layer(input), self.attention_layers))
        concatenated_attention_vectors = torch.cat(attention_vectors, dim=1)
        scaled_concatenated_attention_vectors = torch.matmul(concatenated_attention_vectors, self.w_o)
        return scaled_concatenated_attention_vectors


class Encoder(nn.Module):
    def __init__(self, w_o, num_heads=8, word_width=512):
        super(Encoder, self).__init__()
        self.layer_norm = nn.LayerNorm(word_width)
        self.multiheaded_attention_layer = MultiheadedAttention(w_o, num_heads)
        self.feedforward_layer = nn.Sequential(
            nn.Linear(word_width, DefaultParameters.DEFAULT_FFNN_HIDDEN_LAYER_WIDTH, bias=True),
            nn.LeakyReLU(),
            nn.Linear(DefaultParameters.DEFAULT_FFNN_HIDDEN_LAYER_WIDTH, word_width, bias=True))

    def forward(self, input):
        mh_output = self.multiheaded_attention_layer(input)
        # Adds the residual connection to the output of the attention layer
        layer_normed_multihead_output = self.layer_norm(mh_output + input)
        ffnn_outputs = torch.stack(
            list(map(lambda attention_vector: self.feedforward_layer(attention_vector), layer_normed_multihead_output)))
        layer_normed_ffnn_output = self.layer_norm(ffnn_outputs + layer_normed_multihead_output)
        return layer_normed_ffnn_output


W_Q = torch.randn([WORD_WIDTH, PROJECTION_WIDTH]) / SCALE_FACTOR
W_K = torch.randn([WORD_WIDTH, PROJECTION_WIDTH]) / SCALE_FACTOR
W_V = torch.randn([WORD_WIDTH, PROJECTION_WIDTH]) / SCALE_FACTOR
W_O = torch.randn([NUM_HEADS * PROJECTION_WIDTH, WORD_WIDTH]) / SCALE_FACTOR


def qkvs(words, w_q, w_k, w_v):
    return torch.matmul(words, w_q), torch.matmul(words, w_k), torch.matmul(words, w_v)


def positional_encoding(num_dimensions):
    return lambda pos, dimension: (math.sin(
        pos / math.pow(10000, dimension / DefaultParameters.DEFAULT_WORD_WIDTH)) if (
            dimension % 2 == 0) else math.cos(pos / math.pow(10000, (dimension - 1) / num_dimensions))
)

def encoding_map(positional_encoding):
    num_words = DefaultParameters.DEFAULT_MAX_WORDS
    positions = range(num_words)
    dimensions = range(DefaultParameters.DEFAULT_WORD_WIDTH)
    mesh = np.zeros([len(positions), len(dimensions)])

    for dimension in dimensions:
        for position in positions:
            mesh[position, dimension] = positional_encoding(position, dimension)

    return mesh


def positionally_encoded(words, encoding_map):
    return (words + encoding_map[:len(words)]).float()


ENCODING_MAP = encoding_map(positional_encoding(DefaultParameters.DEFAULT_WORD_WIDTH))
num_words = 10
words = torch.randn([num_words, WORD_WIDTH])
qkv_words = qkvs(words, W_Q, W_K, W_V)
stack = encoder_stack(6, W_O)
values = stack(positionally_encoded(words, ENCODING_MAP))
print(values)
print(values.shape)
# encoder = EncoderCtor(W_O)
# encoder.eval()
# values = encoder(words)
# print(values)
# print(values.shape)
