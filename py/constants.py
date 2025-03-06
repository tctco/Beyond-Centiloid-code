import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RIGID_INSHAPE = (1, 64, 64, 64)
CHANNELS = (64, 64, 64, 128, 256)
STRIDES = (2, 2, 2, 2, 2)

ELASTIC_INSHAPE = (79 + 17, 95 + 33, 79 + 17)

RIGID_BTACH_SIZE = 128
ELASTIC_BATCH_SIZE = 16

DDF_REGULARIZATION = 0.1

NUM_WORKER = 4
