import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
import optax


class Simplest(nn.Module):
    def setup(self, nFeatures):
        self.dense_1 = nn.Dense(nFeatures)
        self.relu_1 = nn.relu

    def __call__(self, x):
        x = self.dense_1(x)
        x = self.relu_1(x)
        return x
