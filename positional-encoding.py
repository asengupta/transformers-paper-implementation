import numpy as np
import math
import matplotlib.pyplot as plt

class DefaultParameters:
    DEFAULT_NUM_HEADS = 8
    DEFAULT_WORD_WIDTH = 512
    DEFAULT_PROJECTION_WIDTH = 64
    DEFAULT_SCALE_FACTOR = 100
    DEFAULT_FFNN_HIDDEN_LAYER_WIDTH = 2048

num_words = 10
positions = range(num_words)
dimensions = range(DefaultParameters.DEFAULT_WORD_WIDTH)

mesh = np.zeros([len(positions), len(dimensions)])
positional_encoding = lambda pos, dimension: math.sin(
    pos / math.pow(10000, dimension / DefaultParameters.DEFAULT_WORD_WIDTH)) if (
        dimension % 2 == 0) else math.cos(pos / math.pow(10000, (dimension - 1) / len(dimensions)))

plt.figure()

for dimension in dimensions:
    for position in positions:
        encoding = positional_encoding(position, dimension)
        greyscale_encoding = (encoding + 1) / 2
        mesh[position, dimension] = greyscale_encoding
        # plt.plot(dimension, position, marker="s", color=(greyscale_encoding, greyscale_encoding, greyscale_encoding))

plt.pcolormesh(mesh)
plt.show()
