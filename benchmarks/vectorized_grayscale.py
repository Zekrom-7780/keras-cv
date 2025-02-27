# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from keras_cv.layers import Grayscale
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


class OldGrayscale(BaseImageAugmentationLayer):
    """Grayscale is a preprocessing layer that transforms RGB images to
    Grayscale images.
    Input images should have values in the range of [0, 255].
    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Args:
        output_channels.
            Number color channels present in the output image.
            The output_channels can be 1 or 3. RGB image with shape
            (..., height, width, 3) will have the following shapes
            after the `Grayscale` operation:
                 a. (..., height, width, 1) if output_channels = 1
                 b. (..., height, width, 3) if output_channels = 3.
    Usage:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    to_grayscale = keras_cv.layers.preprocessing.Grayscale()
    augmented_images = to_grayscale(images)
    ```
    """

    def __init__(self, output_channels=1, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels
        # This layer may raise an error when running on GPU using auto_vectorize
        self.auto_vectorize = False

    def compute_image_signature(self, images):
        # required because of the `output_channels` argument
        if isinstance(images, tf.RaggedTensor):
            ragged_spec = tf.RaggedTensorSpec(
                shape=images.shape[1:3] + [self.output_channels],
                ragged_rank=1,
                dtype=self.compute_dtype,
            )
            return ragged_spec
        return tf.TensorSpec(
            images.shape[1:3] + [self.output_channels], self.compute_dtype
        )

    def _check_input_params(self, output_channels):
        if output_channels not in [1, 3]:
            raise ValueError(
                "Received invalid argument output_channels. "
                f"output_channels must be in 1 or 3. Got {output_channels}"
            )
        self.output_channels = output_channels

    def augment_image(self, image, transformation=None, **kwargs):
        grayscale = tf.image.rgb_to_grayscale(image)
        if self.output_channels == 1:
            return grayscale
        elif self.output_channels == 3:
            return tf.image.grayscale_to_rgb(grayscale)
        else:
            raise ValueError("Unsupported value for `output_channels`.")

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return segmentation_mask

    def get_config(self):
        config = {
            "output_channels": self.output_channels,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


(x_train, _), _ = keras.datasets.cifar10.load_data()
x_train = x_train.astype(float)

x_train.shape


images = []

num_images = [1000, 2000, 5000, 10000]

results = {}

for aug in [Grayscale, OldGrayscale]:
    c = aug.__name__

    layer = aug()

    runtimes = []
    print(f"Timing {c}")

    for n_images in num_images:
        # warmup
        layer(x_train[:n_images])

        t0 = time.time()
        r1 = layer(x_train[:n_images])
        t1 = time.time()
        runtimes.append(t1 - t0)
        print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")

    results[c] = runtimes

    c = aug.__name__ + " Graph Mode"

    layer = aug()

    @tf.function()
    def apply_aug(inputs):
        return layer(inputs)

    runtimes = []
    print(f"Timing {c}")

    for n_images in num_images:
        # warmup
        apply_aug(x_train[:n_images])

        t0 = time.time()
        r1 = apply_aug(x_train[:n_images])
        t1 = time.time()
        runtimes.append(t1 - t0)
        print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")

    results[c] = runtimes

plt.figure()
for key in results:
    plt.plot(num_images, results[key], label=key)
    plt.xlabel("Number images")

plt.ylabel("Runtime (seconds)")
plt.legend()
plt.show()

# So we can actually see more relevant margins
del results["OldGrayscale"]

plt.figure()
for key in results:
    plt.plot(num_images, results[key], label=key)
    plt.xlabel("Number images")

plt.ylabel("Runtime (seconds)")
plt.legend()
plt.show()
