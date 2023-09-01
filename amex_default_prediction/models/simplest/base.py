import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
import optax
from flax.training import train_state


class Simplest(nn.Module):
    nFeatures = 189

    def setup(self):
        self.dense_1 = nn.Dense(Simplest.nFeatures)
        self.relu_1 = nn.relu
        # add more in the near future

    def __call__(self, x):
        x = self.dense_1(x)
        x = self.relu_1(x)
        return x


def create_train_state_Simplest(rng, learning_rate):
    model = Simplest()
    params = model.init(
        rng,
        jnp.empty(
            [  # These values will probably need to be changed
                1,
                189,
                189,
                1,
            ]
        ),
    )["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


if __name__ == "__main__":
    Simplest()
    print(
        create_train_state_Simplest(
            jax.random.PRNGKey(0),
            0.3,
        )
    )
