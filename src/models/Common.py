import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = K.cast(y_true, dtype=tf.float32)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


def create_weighted_binary_crossentropy_2(zero_weight, one_weight, class_weights):

    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = K.cast(y_true, dtype=tf.float32)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce * class_weights

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


def weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = K.cast(y_true, dtype=tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Compute cross entropy from probabilities.
        bce = y_true * tf.math.log(y_pred + epsilon)
        bce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
        bce = -bce

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_bce = weight_vector * bce

        # Return the mean error
        return tf.reduce_mean(weighted_bce)

    return weighted_binary_crossentropy
