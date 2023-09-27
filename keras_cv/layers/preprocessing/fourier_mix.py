# Copyright 2022 The KerasCV Authors
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

import tensorflow as tf
from keras_cv.api_export import keras_cv_export
from keras_cv.layers.preprocessing.base_image_augmentation_layer import BaseImageAugmentationLayer

@keras_cv_export("keras_cv.layers.FourierMix")
class FourierMix(BaseImageAugmentationLayer):
    """FourierMix implements the FMix data augmentation technique.

    Args:
        alpha: Float value for beta distribution. Inverse scale parameter for
            the gamma distribution. This controls the shape of the distribution
            from which the smoothing values are sampled. Defaults to 0.5, which
            is a recommended value in the paper.
        decay_power: A float value representing the decay power, defaults to 3,
            as recommended in the paper.
        seed: Integer. Used to create a random seed.
    References:
        - [FMix paper](https://arxiv.org/abs/2002.12047).

    Sample usage:
    ```python
    (images, labels, bounding_boxes), _ = load_data()  # Load your data including bounding boxes
    fourier_mix = FourierMix(alpha=0.5)
    augmented_inputs = fourier_mix({'images': images, 'labels': labels, 'bounding_boxes': bounding_boxes})
    # Now, 'augmented_inputs' contains the augmented images, labels, and bounding boxes.
    ```
    """

    def __init__(self, alpha=0.5, decay_power=3, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.alpha = alpha
        self.decay_power = decay_power
        self.seed = seed

    def _sample_from_beta(self, alpha, beta, shape):
        sample_alpha = tf.random.gamma(shape, alpha=alpha, seed=self._random_generator.make_legacy_seed())
        sample_beta = tf.random.gamma(shape, alpha=beta, seed=self._random_generator.make_legacy_seed())
        return sample_alpha / (sample_alpha + sample_beta)

    def _update_bounding_boxes(self, bounding_boxes, lambda_sample, permutation_order):
        reordered_bboxes = tf.gather(bounding_boxes, permutation_order)
        updated_bboxes = lambda_sample * bounding_boxes + (1.0 - lambda_sample) * reordered_bboxes
        return updated_bboxes

    def _batch_augment(self, inputs):
        images = inputs.get("images", None)
        labels = inputs.get("labels", None)
        segmentation_masks = inputs.get("segmentation_masks", None)
        bounding_boxes = inputs.get("bounding_boxes", None)

        if images is None or (labels is None and segmentation_masks is None):
            raise ValueError("FourierMix expects inputs in a dictionary with format {'images': images, 'labels': labels}. {'images': images, 'segmentation_masks': segmentation_masks}. Got: inputs = {inputs}")

        images, masks, lambda_sample, permutation_order = self._fourier_mix(images)

        if labels is not None:
            labels = self._update_labels(labels, lambda_sample, permutation_order)
            inputs["labels"] = labels

        if segmentation_masks is not None:
            segmentation_masks = self._update_segmentation_masks(segmentation_masks, masks, permutation_order)
            inputs["segmentation_masks"] = segmentation_masks

        if bounding_boxes is not None:
            bounding_boxes = self._update_bounding_boxes(bounding_boxes, lambda_sample, permutation_order)
            inputs["bounding_boxes"] = bounding_boxes

        inputs["images"] = images
        return inputs

    def _augment(self, inputs):
        raise ValueError(
            "FourierMix received a single image to `call`. The layer relies on "
            "combining multiple examples, and as such will not behave as "
            "expected. Please call the layer with 2 or more samples."
        )

    def _fourier_mix(self, images):
        shape = tf.shape(images)
        permutation_order = tf.random.shuffle(tf.range(0, shape[0]), seed=self.seed)
        lambda_sample = self._sample_from_beta(self.alpha, self.alpha, (shape[0],))

        masks = tf.map_fn(
            lambda x: self._sample_mask_from_transform(self.decay_power, shape[1:-1]),
            tf.range(shape[0], dtype=tf.float32),
        )

        masks = tf.map_fn(
            lambda i: self._binarise_mask(masks[i], lambda_sample[i], shape[1:-1]),
            tf.range(shape[0], dtype=tf.int32),
            fn_output_signature=tf.float32,
        )
        masks = tf.expand_dims(masks, -1)

        fmix_images = tf.gather(images, permutation_order)
        images = masks * images + (1.0 - masks) * fmix_images

        return images, masks, lambda_sample, permutation_order

    def _get_spectrum(self, freqs, decay_power, channel, h, w):
        scale = tf.ones(1) / tf.cast(
            tf.math.maximum(freqs, tf.convert_to_tensor(1e-7, dtype=freqs.dtype)),
            dtype=freqs.dtype,
        )
        x = tf.range(0, h, dtype=tf.float32)
        y = tf.range(0, w, dtype=tf.float32)

        x_cos, y_cos = tf.meshgrid(tf.cos(scale * x), tf.cos(scale * y))
        x_sin, y_sin = tf.meshgrid(tf.sin(scale * x), tf.sin(scale * y))

        x_cos = tf.reshape(x_cos, (-1,))
        y_cos = tf.reshape(y_cos, (-1,))
        x_sin = tf.reshape(x_sin, (-1,))
        y_sin = tf.reshape(y_sin, (-1,))

        spectrum_real = tf.concat([x_cos, y_cos], axis=0)
        spectrum_imag = tf.concat([x_sin, y_sin], axis=0)

        spectrum_real = spectrum_real / tf.norm(spectrum_real)
        spectrum_imag = spectrum_imag / tf.norm(spectrum_imag)

        return spectrum_real, spectrum_imag

    def _sample_mask_from_transform(self, decay_power, shape):
        freqs = self._random_generator.uniform(
            shape=(1,),
            minval=0.1,
            maxval=1.0,
            seed=self._random_generator.make_legacy_seed(),
        )
        channel = self._random_generator.uniform(
            shape=(1,),
            minval=0.0,
            maxval=1.0,
            seed=self._random_generator.make_legacy_seed(),
        )
        real_coeffs, imag_coeffs = self._get_spectrum(freqs, decay_power, channel, shape[0], shape[1])

        magnitude = self._random_generator.uniform(
            shape=(1,),
            minval=0.0,
            maxval=1.0,
            seed=self._random_generator.make_legacy_seed(),
        )
        phase = self._random_generator.uniform(
            shape=(1,),
            minval=0.0,
            maxval=2.0 * tf.constant(tf.math.pi, dtype=tf.float32),
            seed=self._random_generator.make_legacy_seed(),
        )

        spectrum_magnitude = tf.abs(
            tf.complex(
                magnitude * real_coeffs,
                magnitude * imag_coeffs,
            )
        )

        spectrum_magnitude = tf.reshape(spectrum_magnitude, (1, shape[0] * shape[1]))
        phase = tf.reshape(phase, (1,))

        return self._apply_mask(spectrum_magnitude, phase, shape[0], shape[1])

    def _binarise_mask(self, mask, sample_prob, shape):
        binary_mask = self._random_generator.uniform(
            shape=shape,
            minval=0.0,
            maxval=1.0,
            seed=self._random_generator.make_legacy_seed(),
        )
        binary_mask = tf.cast(binary_mask <= sample_prob, dtype=tf.float32)
        return mask * binary_mask

    def _apply_mask(self, spectrum_magnitude, phase, h, w):
        spectrum_magnitude = tf.reshape(spectrum_magnitude, (1, h, w))
        phase = tf.reshape(phase, (1, 1, 1))

        spectrum_real = spectrum_magnitude * tf.cos(phase)
        spectrum_imag = spectrum_magnitude * tf.sin(phase)

        complex_spectrum = tf.complex(spectrum_real, spectrum_imag)
        inverse_transform = tf.signal.ifft2d(complex_spectrum)
        mask = tf.abs(inverse_transform)
        mask = tf.reshape(mask, (h, w, 1))
        mask = tf.repeat(mask, axis=-1, repeats=3)
        return mask

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "decay_power": self.decay_power,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
