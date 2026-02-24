import tensorflow as tf
import qkeras

from tensorflow.keras.layers import Dense, Input, BatchNormalization
from qkeras.qlayers import QDense, QActivation

NUM_CLASSES = 5


def qkeras_dense_model(
    in_shape,
    dense_widths=[64],
    logit_total_bits=4,
    logit_int_bits=0,
    activation_total_bits=8,
    activation_int_bits=0,
    alpha=1,
    logit_quantizer="quantized_bits",
    activation_quantizer="quantized_relu",
):
    logit_quantizer = getattr(qkeras.quantizers, logit_quantizer)(
        logit_total_bits,
        logit_int_bits,
        alpha=alpha,
    )
    activation_quantizer = getattr(qkeras.quantizers, activation_quantizer)(
        activation_total_bits,
        activation_int_bits,
    )
    x = x_in = Input(in_shape, name="input1")

    for i, w in enumerate(dense_widths):
        x = QDense(
            w,
            kernel_quantizer=logit_quantizer,
            bias_quantizer=logit_quantizer,
            name=f"dense{i+1}",
        )(x)
        x = BatchNormalization()(x)
        x = QActivation(activation=activation_quantizer)(x)

    x = QDense(
        NUM_CLASSES,
        kernel_quantizer=logit_quantizer,
        bias_quantizer=logit_quantizer,
        name="dense_out",
    )(x)

    model = tf.keras.models.Model(inputs=x_in, outputs=x)
    return model