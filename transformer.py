from typing import List, Callable, Any

import math
import numpy as np
import torch
import torch.nn as nn
from enum import Enum

from torch import Tensor

softmax = torch.nn.Softmax(dim=1)


class Parameters:
    NUM_HEADS = 8
    WORD_WIDTH = 512
    PROJECTION_WIDTH = 64
    SCALE_FACTOR = 100
    FFNN_HIDDEN_LAYER_WIDTH = 2048
    MAX_WORDS = 25


class TransformerMode(Enum):
    INFERENCE = 1
    TRAINING = 2


class Tokens:
    START_TOKEN = torch.randn(Parameters.WORD_WIDTH)


W_Q = torch.randn([Parameters.WORD_WIDTH, Parameters.PROJECTION_WIDTH]) / Parameters.SCALE_FACTOR
W_K = torch.randn([Parameters.WORD_WIDTH, Parameters.PROJECTION_WIDTH]) / Parameters.SCALE_FACTOR
W_V = torch.randn([Parameters.WORD_WIDTH, Parameters.PROJECTION_WIDTH]) / Parameters.SCALE_FACTOR
W_O = torch.randn([Parameters.NUM_HEADS * Parameters.PROJECTION_WIDTH,
                   Parameters.WORD_WIDTH]) / Parameters.SCALE_FACTOR


def encoder_stack(num_encoders, w_o):
    return EncoderStack(num_encoders, w_o)


class EncoderStack:
    def __init__(self, num_encoders, w_o):
        self.encoders: list[Encoder] = list(map(lambda x: Encoder(SingleSourceQKVLayer(W_Q, W_K, W_V), w_o,
                                                       Parameters.NUM_HEADS,
                                                       Parameters.WORD_WIDTH),
                                     range(num_encoders)))
        self.stack = nn.Sequential(*self.encoders)

    def forward(self, input):
        return self.stack(input)


class DecoderStack:
    def __init__(self, num_decoders, w_o):
        self.decoders: list[Decoder] = list(
            map(lambda x: Decoder(SingleSourceQKVLayer(W_Q, W_K, W_V), MultiSourceQKVLayer(W_Q, W_K, W_V), w_o),
                range(num_decoders)))
        self.stack = nn.Sequential(*self.decoders)

    def forward(self, encoder_output, decoder_target):
        encoder_output, decoder_output = self.stack((encoder_output, decoder_target))
        return decoder_output

    def set_mode(self, mode):
        self.mode = mode


def decoder_stack(num_decoders, w_o, mode=TransformerMode.INFERENCE):
    return DecoderStack(num_decoders, w_o)


class Transformer:
    linear: Tensor
    embedding: Callable[[Tensor], Tensor]
    vector_buffer: list[Tensor]
    text_buffer: list[str]

    def __init__(self, vocabulary_map, mode=TransformerMode.INFERENCE):
        self.vocabulary_map = vocabulary_map
        self.mode = mode
        self.embedding = embedding(encoding_map(encoding_seed(Parameters.WORD_WIDTH)))
        self.decoders = decoder_stack(6, W_O)
        self.encoders = encoder_stack(6, W_O)
        self.linear = torch.randn([Parameters.WORD_WIDTH, Parameters.MAX_WORDS])
        # vector_buffer stores the vector representation of the output
        # text_buffer stores the text representation of the output
        self.vector_buffer = [Tokens.START_TOKEN]
        self.text_buffer = ["<SOS>"]

    # The decoder_target does not need to be passed in if the Transformer is in Inference mode, since
    # the outputs are fed back in as the decoder input, save the first time, where the fed token is
    # <SOS>=<Start of Sentence>
    def forward(self, words, decoder_target=None):
        if (decoder_target is None and self.mode == TransformerMode.TRAINING):
            raise Exception("Decoder Target must be provided during Training mode.")
        decoder_stack_input = decoder_target if self.mode == TransformerMode.TRAINING else torch.stack(
            self.vector_buffer)
        print(f"Transformer mode={self.mode}")
        encoder_block_output = self.encoders.forward(self.embedding(words))
        print(f"Stack input={decoder_stack_input.shape}")

        decoder_output = self.decoders.forward(encoder_block_output, decoder_stack_input)
        term_distributions = softmax(torch.matmul(decoder_output, self.linear))
        vocabulary_output = list(
            map(lambda distribution: self.vocabulary_map[distribution.argmax()], term_distributions))
        text_buffer = list(map(lambda vocabulary_entry: vocabulary_entry[0], vocabulary_output))
        vector_buffer = list(map(lambda vocabulary_entry: vocabulary_entry[1], vocabulary_output))
        if (self.mode == TransformerMode.INFERENCE):
            # The last word is chosen as the predicted word
            self.vector_buffer.append(vocabulary_output[-1][1])
            self.text_buffer.append(vocabulary_output[-1][0])
        # Returns the stored text_buffer, vector_buffer for Inference mode, otherwise outputs the Decoder
        # output as-is
        return [text_buffer, vector_buffer] if self.mode == TransformerMode.TRAINING else [self.text_buffer,
                                                                                           self.vector_buffer]

    def set_mode(self, mode):
        self.decoders.set_mode(mode)


# This QKV 'layer' is used to feed data to the encoder attention layer and the masked attention layer of the Decoder.
# The source of this layer is just the single set of words
class SingleSourceQKVLayer:
    def __init__(self, w_q, w_k, w_v):
        self.qkv = qkv(w_q, w_k, w_v)

    def forward(self, words):
        return self.qkv(words, words, words)


def qkv(w_q, w_k, w_v):
    return lambda query_input, key_input, value_input: (
        torch.matmul(query_input, w_q), torch.matmul(key_input, w_k), torch.matmul(
            value_input, w_v))


# This QKV 'layer' is used to feed data to the Encoder-Decoder attention layer. Because that layer uses the Query
# from the previous Decoder stage and the Key-Value from the Encoder output, the source of these embeddings
# is not a single one.
class MultiSourceQKVLayer:
    def __init__(self, w_q, w_k, w_v):
        self.qkv = qkv(w_q, w_k, w_v)

    def forward(self, inputs):
        encoder_output, decoder_output = inputs
        return self.qkv(decoder_output, encoder_output, encoder_output)


class SelfAttentionLayer(nn.Module):
    def __init__(self, mask=False):
        super(SelfAttentionLayer, self).__init__()
        self.mask = mask

    def forward(self, input_qkv):
        return self.attention_scores(input_qkv)

    def masked(self, q_dot_k):
        return q_dot_k.tril() + torch.full(q_dot_k.shape, - math.inf).triu(1)

    def attention_scores(self, qkvs):
        q, k, v = qkvs
        q_dot_k = torch.matmul(q, k.t()) / math.sqrt(q.shape[1])
        # The SelfAttentionLayer is used in both masked and unmasked variants. The mask value is used
        # to decide if the intermediate matrix should be masked or not before performing Softmax
        maybe_masked_q_dot_k = self.masked(q_dot_k) if self.mask else q_dot_k
        return torch.matmul(softmax(maybe_masked_q_dot_k), v)


class MultiheadedAttention(nn.Module):
    attention_layers: list[SelfAttentionLayer]

    def __init__(self, w_o, num_heads=Parameters.NUM_HEADS, mask=False):
        super(MultiheadedAttention, self).__init__()
        self.w_o = w_o
        self.attention_layers = list(map(lambda x: SelfAttentionLayer(mask=mask), range(num_heads)))

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
            nn.Linear(word_width, Parameters.FFNN_HIDDEN_LAYER_WIDTH, bias=True),
            nn.LeakyReLU(),
            nn.Linear(Parameters.FFNN_HIDDEN_LAYER_WIDTH, word_width, bias=True))

    def forward(self, input):
        input_qkv = self.qkv_source.forward(input)
        mh_output = self.multiheaded_attention_layer(input_qkv)
        # Adds the residual connection to the output of the attention layer
        layer_normed_multihead_output = self.layer_norm(mh_output + input)
        # print(f"LNMH shape={layer_normed_multihead_output.shape}")
        ffnn_outputs = torch.stack(
            list(map(lambda attention_vector: self.feedforward_layer(attention_vector), layer_normed_multihead_output)))
        layer_normed_ffnn_output = self.layer_norm(ffnn_outputs + layer_normed_multihead_output)
        print(f"FFNN Shape={layer_normed_ffnn_output.shape}")
        return layer_normed_ffnn_output


class Decoder(nn.Module):
    def __init__(self, previous_decoder_source, unmasked_qkv_source, w_o, num_heads=8, word_width=512):
        super(Decoder, self).__init__()
        self.unmasked_qkv_source = unmasked_qkv_source
        self.masked_qkv_source = previous_decoder_source
        self.layer_norm = nn.LayerNorm(word_width)
        self.masked_multiheaded_attention_layer = MultiheadedAttention(w_o, num_heads, mask=True)
        self.multiheaded_attention_layer = MultiheadedAttention(w_o, num_heads)
        self.feedforward_layer = nn.Sequential(
            nn.Linear(word_width, Parameters.FFNN_HIDDEN_LAYER_WIDTH, bias=True),
            nn.LeakyReLU(),
            nn.Linear(Parameters.FFNN_HIDDEN_LAYER_WIDTH, word_width, bias=True))

    # The encoder output is injected directly into the sublayer of every Decoder. To build up the chain of Decoders
    # in PyTorch, so that we can put the full stack inside a Sequential block, we simply inject the encoder output
    # to the root Decoder, and have it output the encoder output (together with the actual Decoder output) as part of
    # the Decoder's actual output to make it easy for the next Decoder in the stack to consume the Encoder and Decoder
    # outputs
    def forward(self, input):
        encoder_output, previous_stage_output = input
        masked_mh_output = self.masked_multiheaded_attention_layer(
            self.masked_qkv_source.forward(previous_stage_output))
        input_qkv = self.unmasked_qkv_source.forward((encoder_output, masked_mh_output))
        mh_output = self.multiheaded_attention_layer(input_qkv)
        # Adds the residual connection to the output of the attention layer
        layer_normed_multihead_output = self.layer_norm(mh_output + previous_stage_output)
        ffnn_outputs = torch.stack(
            list(map(lambda attention_vector: self.feedforward_layer(attention_vector), layer_normed_multihead_output)))
        layer_normed_ffnn_output = self.layer_norm(ffnn_outputs + layer_normed_multihead_output)
        return (encoder_output, layer_normed_ffnn_output)


def encoding_seed(num_dimensions):
    return lambda pos, dimension: (math.sin(
        pos / math.pow(10000, dimension / Parameters.WORD_WIDTH)) if (
            dimension % 2 == 0) else math.cos(pos / math.pow(10000, (dimension - 1) / num_dimensions))
                                   )


def encoding_map(positional_encoding):
    positions = range(Parameters.MAX_WORDS)
    dimensions = range(Parameters.WORD_WIDTH)
    mesh = np.zeros([len(positions), len(dimensions)])

    for dimension in dimensions:
        for position in positions:
            mesh[position, dimension] = positional_encoding(position, dimension)

    return mesh


def embedding(encoding_map):
    return lambda words: (words + encoding_map[:len(words)]).float()


VOCABULARY_MAP = []
for i in range(Parameters.MAX_WORDS):
    VOCABULARY_MAP.append(["Word", torch.randn(Parameters.WORD_WIDTH)])

num_words = 12
words = torch.randn([num_words, Parameters.WORD_WIDTH])
decoder_target = torch.randn([5, Parameters.WORD_WIDTH])
t = Transformer(VOCABULARY_MAP, mode=TransformerMode.TRAINING)
texts, embeddings = t.forward(words, decoder_target)
# output = t.forward(words, decoder_target)
# output = t.forward(words, decoder_target)
print(texts)
