from typing import Tuple, Union, NamedTuple
from dataclasses import dataclass


import tensorflow as tf
from dataclasses_json import dataclass_json
from flytekit import Resources, task, workflow
from flytekit.types.directory import FlyteDirectory
import numpy as np


@dataclass_json
@dataclass
class Hyperparameters(object):
    """
    Args:
        batch_size: input batch size for training (default: 64)
        epochs: number of epochs to train (default: 10)
        learning_rate: gradient update step size (default: 0.01)
    """

    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.01


TrainingOutputs = NamedTuple(
    "TrainingOutputs",
    model_dir=FlyteDirectory,
)


def get_network(input_shape: Tuple[int, int]) -> tf.keras.Model:
    # Define a toy model
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Activation("relu")(x)

    outputs = tf.keras.layers.Dense(10)(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
    )


def get_data() -> Tuple[np.ndarray, np.ndarray]:
    return tf.keras.datasets.mnist.load_data()


@task(
    retries=2,
    cache=True,
    cache_version="1.0",
    requests=Resources(gpu="0", mem="1Gi", storage="1Gi"),
    limits=Resources(gpu="0", mem="1Gi", storage="1Gi"),
)
def tf_mnist_task(hp: Hyperparameters) -> TrainingOutputs:
    model = get_network(input_shape=(784,))

    (X_train, y_train), _ = get_data()
    X_train = X_train.reshape(60000, 784).astype("float32") / 255

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(lr=hp.learning_rate),
        metrics=["accuracy"],
    )

    _ = model.fit(X_train, y_train, batch_size=hp.batch_size, epochs=hp.epochs)

    model.save("my_model")
    return TrainingOutputs(model_dir=FlyteDirectory("my_model"))


@workflow
def tf_mnist_wf(hp: Hyperparameters) -> TrainingOutputs:
    return tf_mnist_task(hp=hp)


if __name__ == "__main__":
    print(
        tf_mnist_wf(hp=Hyperparameters(epochs=10, batch_size=16, learning_rate=10e-5))
    )
