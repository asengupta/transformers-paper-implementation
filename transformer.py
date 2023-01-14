import torch
import torch.nn as nn
import numpy as np
import math
import functools

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

W_Q = torch.randn([WORD_WIDTH, PROJECTION_WIDTH]) / SCALE_FACTOR
W_K = torch.randn([WORD_WIDTH, PROJECTION_WIDTH]) / SCALE_FACTOR
W_V = torch.randn([WORD_WIDTH, PROJECTION_WIDTH]) / SCALE_FACTOR
W_O = torch.randn([NUM_HEADS * PROJECTION_WIDTH, WORD_WIDTH]) / SCALE_FACTOR


def encoder_stack(num_encoders, w_o):
    encoders = np.array(list(map(lambda x: Encoder(WordSourcedQKVLayer(W_Q, W_K, W_V), w_o,
                                                   DefaultParameters.DEFAULT_NUM_HEADS,
                                                   DefaultParameters.DEFAULT_WORD_WIDTH), range(num_encoders))))
    return nn.Sequential(*encoders)


class EncoderStack:
    def __init__(self, num_encoders, w_o):
        self.encoders = np.array(list(map(lambda x: Encoder(WordSourcedQKVLayer(W_Q, W_K, W_V), w_o,
                                                            DefaultParameters.DEFAULT_NUM_HEADS,
                                                            DefaultParameters.DEFAULT_WORD_WIDTH),
                                          range(num_encoders))))
        self.stack = nn.Sequential(*encoders)

    def forward(self, input):
        return self.stack(input)


class DecoderStack:
    def __init__(self, num_decoders, w_o):
        self.root_decoder = Decoder(WordSourcedQKVLayer(W_Q, W_K, W_V), MultiSourcedQKVLayer(W_Q, W_K, W_V), W_O)
        self.decoders = np.array(list(
            map(lambda x: Decoder(WordSourcedQKVLayer(W_Q, W_K, W_V), MultiSourcedQKVLayer(W_Q, W_K, W_V), w_o),
                range(num_decoders - 1))))

    def forward(self, encoder_output, decoder_target):
        root_decoder_result = self.root_decoder((encoder_output, decoder_target))
        return functools.reduce(lambda result, decoder: decoder((encoder_output, result)), self.decoders, root_decoder_result)
        # for decoder in self.decoders:
        #     result = decoder((encoder_output, result))
        # return result


def decoder_stack(num_decoders, w_o):
    return DecoderStack(num_decoders, w_o)
    # root_decoder = [Decoder(WordSourcedQKVLayer(W_Q, W_K, W_V), MultiSourcedQKVLayer(W_Q, W_K, W_V), W_O)]
    # decoders = np.array(list(
    #     map(lambda x: Decoder(WordSourcedQKVLayer(W_Q, W_K, W_V), MultiSourcedQKVLayer(W_Q, W_K, W_V), w_o),
    #         range(num_decoders))))
    # return nn.Sequential(*(root_decoder + decoders))
    # return nn.Sequential(*decoders)


class Transformer:
    def __init__(self, encoders, decoders, embedding):
        self.embedding = embedding
        self.decoders = decoders
        self.encoders = encoders
        self.output_buffer = []

    def forward(self, words, decoder_target):
        encoder_block_output = self.encoders.forward(self.embedding(words))
        return self.decoders.forward(encoder_block_output, decoder_target)


class WordSourcedQKVLayer:
    def __init__(self, w_q, w_k, w_v):
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v

    def forward(self, words):
        return torch.matmul(words, self.w_q), torch.matmul(words, self.w_k), torch.matmul(words, self.w_v)


class MultiSourcedQKVLayer:
    def __init__(self, w_q, w_k, w_v):
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v

    def forward(self, inputs):
        encoder_output, decoder_output = inputs
        return torch.matmul(decoder_output, self.w_q), torch.matmul(encoder_output, self.w_k), torch.matmul(
            encoder_output, self.w_v)


class SelfAttentionLayer(nn.Module):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()

    def forward(self, input_qkv):
        return self.attention_scores(input_qkv)

    def attention_scores(self, qkvs):
        Q, K, V = list(range(3))
        return torch.matmul(
            softmax(torch.matmul(qkvs[Q], torch.transpose(qkvs[K], 0, 1)) / math.sqrt(qkvs[Q].shape[1])), qkvs[V])


class MultiheadedAttention(nn.Module):
    def __init__(self, w_o, num_heads=DefaultParameters.DEFAULT_NUM_HEADS):
        super(MultiheadedAttention, self).__init__()
        self.w_o = w_o
        self.attention_layers = list(map(lambda x: SelfAttentionLayer(), range(num_heads)))

    def forward(self, input_qkv):
        # Concatenating gives [num_words x num_heads * projection_width]
        attention_vectors = list(map(lambda attention_layer: attention_layer(input_qkv), self.attention_layers))
        concatenated_attention_vectors = torch.cat(attention_vectors, dim=1)
        scaled_concatenated_attention_vectors = torch.matmul(concatenated_attention_vectors, self.w_o)
        return scaled_concatenated_attention_vectors


class Encoder(nn.Module):
    def __init__(self, qkv_source, w_o, num_heads=8, word_width=512):
        super(Encoder, self).__init__()
        self.qkv_source = qkv_source
        self.layer_norm = nn.LayerNorm(word_width)
        self.multiheaded_attention_layer = MultiheadedAttention(w_o, num_heads)

        self.feedforward_layer = nn.Sequential(
            nn.Linear(word_width, DefaultParameters.DEFAULT_FFNN_HIDDEN_LAYER_WIDTH, bias=True),
            nn.LeakyReLU(),
            nn.Linear(DefaultParameters.DEFAULT_FFNN_HIDDEN_LAYER_WIDTH, word_width, bias=True))

    def forward(self, input):
        input_qkv = self.qkv_source.forward(input)
        mh_output = self.multiheaded_attention_layer(input_qkv)
        # Adds the residual connection to the output of the attention layer
        layer_normed_multihead_output = self.layer_norm(mh_output + input)
        # print(f"LNMH shape={layer_normed_multihead_output.shape}")
        ffnn_outputs = torch.stack(
            list(map(lambda attention_vector: self.feedforward_layer(attention_vector), layer_normed_multihead_output)))
        layer_normed_ffnn_output = self.layer_norm(ffnn_outputs + layer_normed_multihead_output)
        # print(f"FFNN Shape={layer_normed_ffnn_output.shape}")
        return layer_normed_ffnn_output


class Decoder(nn.Module):
    def __init__(self, previous_decoder_source, unmasked_qkv_source, w_o, num_heads=8, word_width=512):
        super(Decoder, self).__init__()
        self.unmasked_qkv_source = unmasked_qkv_source
        self.masked_qkv_source = previous_decoder_source
        self.layer_norm = nn.LayerNorm(word_width)
        self.masked_multiheaded_attention_layer = MultiheadedAttention(w_o, num_heads)
        self.multiheaded_attention_layer = MultiheadedAttention(w_o, num_heads)
        self.feedforward_layer = nn.Sequential(
            nn.Linear(word_width, DefaultParameters.DEFAULT_FFNN_HIDDEN_LAYER_WIDTH, bias=True),
            nn.LeakyReLU(),
            nn.Linear(DefaultParameters.DEFAULT_FFNN_HIDDEN_LAYER_WIDTH, word_width, bias=True))

    def forward(self, input):
        encoder_output, previous_stage_output = input
        # decoder_output = encoder_output
        masked_mh_output = self.masked_multiheaded_attention_layer(
            self.masked_qkv_source.forward(previous_stage_output))
        input_qkv = self.unmasked_qkv_source.forward((encoder_output, masked_mh_output))
        mh_output = self.multiheaded_attention_layer(input_qkv)
        # Adds the residual connection to the output of the attention layer
        layer_normed_multihead_output = self.layer_norm(mh_output + previous_stage_output)
        ffnn_outputs = torch.stack(
            list(map(lambda attention_vector: self.feedforward_layer(attention_vector), layer_normed_multihead_output)))
        layer_normed_ffnn_output = self.layer_norm(ffnn_outputs + layer_normed_multihead_output)
        return layer_normed_ffnn_output


def qkvs(words, w_q, w_k, w_v):
    return torch.matmul(words, w_q), torch.matmul(words, w_k), torch.matmul(words, w_v)


def encoding_seed(num_dimensions):
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


def embedding(encoding_map):
    return lambda words: (words + encoding_map[:len(words)]).float()


ENCODING_MAP = encoding_map(encoding_seed(DefaultParameters.DEFAULT_WORD_WIDTH))
num_words = 10
words = torch.randn([num_words, WORD_WIDTH])
decoder_target = torch.randn([num_words + 5, WORD_WIDTH])
# qkv_words = qkvs(words, W_Q, W_K, W_V)
# encoder_block = encoder_stack(1, W_O)
# encoder_block = Encoder(WordSourcedQKVLayer(W_Q, W_K, W_V), W_O)
# decoder_1 = Decoder(WordSourcedQKVLayer(W_Q, W_K, W_V), MultiSourcedQKVLayer(W_Q, W_K, W_V), W_O)
# encoder_output = encoder_block(embedding(ENCODING_MAP)(words))
# decoder_output = decoder_1(encoder_output)
# print(decoder_output)
# print(decoder_output.shape)
t = Transformer(encoder_stack(6, W_O), decoder_stack(6, W_O), embedding(ENCODING_MAP))
output = t.forward(words, decoder_target)
print(output)
print(output.shape)
# encoder = EncoderCtor(W_O)
# encoder.eval()
# values = encoder(words)
# print(values)
# print(values.shape)
