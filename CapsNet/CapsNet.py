import numpy as np
import tensorflow as tf
import random

from Capslayers import Primary_caps_layer
from Capslayers import Digit_caps_layer

m_plus = 0.9
m_minus = 0.1
lambda_val = 0.5
epsilon = 1e-9

def get_predicting_units(image_shape, conv_filter_size, conv_stride, caps_filter_size, caps_stride, caps_units):

    conv_x = (image_shape[0]-conv_filter_size+1)/conv_stride
    conv_y = (image_shape[1]-conv_filter_size+1)/conv_stride
    caps_x = (conv_x-caps_filter_size+1)/caps_stride
    caps_y = (conv_y-caps_filter_size+1)/caps_stride
    predicting_units = caps_x*caps_y*caps_units    
    
    return predicting_units


def masking(digit_caps_layer, T_c, digit_caps_number, digit_caps_outs, true_input=True):

    T_c = tf.reshape(T_c, (-1, digit_caps_number, 1))

    filter = tf.tile(T_c, [1, 1, digit_caps_outs])
    filter_factor = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]] # this is used exclusively for tests
    filter = tf.add(filter, filter_factor)
    fixed_outs = tf.ones([100, 10, 16]) - 0.7

    if true_input:
        filtered_out = tf.multiply(tf.squeeze(digit_caps_layer), filter)
        decoder_input = tf.reshape(filtered_out,
                                  (-1, digit_caps_outs*digit_caps_number))

    else:
        filtered_out = tf.multiply(fixed_outs, filter)
        decoder_input = tf.reshape(filtered_out,
                                  (-1, digit_caps_outs*digit_caps_number))

    difference = tf.add(tf.squeeze(digit_caps_layer), (fixed_outs*(-1)))
    filtered_difference = tf.multiply(difference, filter)
    difference_error = tf.reduce_mean(tf.square(filtered_difference))

    return decoder_input, difference_error


def fake_masking(digit_caps_number, digit_caps_outs, out="real"):

    T_c = tf.reshape(tf.one_hot(tf.range(start=0, limit=10), 10), [10, digit_caps_number, 1]) # [10, 10, 1]
    filter = tf.tile(T_c, [1, 1, digit_caps_outs]) # [10, 10, 16]
    filter_factor = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]] # [1, 1, 16]
    filter = tf.add(filter, filter_factor)
    fixed_outs = tf.ones([10, 10, 16]) - 0.7
    filtered_out = tf.multiply(tf.squeeze(fixed_outs), filter)
    decoder_input = tf.reshape(filtered_out, (-1, digit_caps_outs*digit_caps_number))
   
    return decoder_input

def loss(X_image, digit_caps_layer, decoded, T_c, batch_size, aging, recon_scale=0.392):


    v_mean = tf.sqrt(tf.reduce_sum(tf.square(digit_caps_layer),axis=2, keepdims=True)+ epsilon) # ok
    v_softmax = tf.nn.softmax(v_mean, axis=1) # ok

    v_right = tf.square(tf.maximum(0., m_plus - v_mean)) # [cfg.batch_size, self.num_label, 1, 1]
    v_wrong = tf.square(tf.maximum(0., v_mean - m_minus)) # [cfg.batch_size, self.num_label, 1, 1]
    v_right = tf.reshape(v_right, [batch_size, -1]) # [cfg.batch_size, self.num_label]
    v_wrong = tf.reshape(v_wrong, [batch_size, -1]) # [cfg.batch_size, self.num_label]

    v_loss = T_c*v_right + lambda_val*(1 - T_c)*v_wrong

    caps_loss = tf.reduce_mean(tf.reduce_sum(v_loss, axis=1))

    squared = tf.square(decoded - X_image)
    aging = (25 + aging)/25
    reconstruction_err = tf.reduce_mean(squared)*recon_scale
    total_loss = caps_loss + reconstruction_err

    return(v_softmax, total_loss, reconstruction_err)

def diff_loss_fun(X_image, reconstructed_image, difference_error, recon_scale=0.05):

    decoder_loss = tf.square(reconstructed_image - X_image)
    decoder_loss = tf.reduce_mean(decoder_loss)*recon_scale
    difference_loss = tf.sqrt(tf.square(difference_error - decoder_loss))

    return difference_loss, decoder_loss

def avg_err_calc(reconstructed_image, X_image, T_c, batch_size, digitcaps_num):

    X_tiled = tf. reshape(X_image, [batch_size, 1, -1]) # [100, 1, 784]
    X_tiled = tf.tile(X_tiled, [1, digitcaps_num ,1]) # [100, 10, 784]
    filter = tf.reshape(T_c, [batch_size, digitcaps_num, 1])  # [100,10,1]
    filtered_X = tf.multiply(X_tiled, filter) # [100, 10, 784]
    filtered_X = tf.transpose(filtered_X, [1,2,0]) # [10, 784, 100]
    reduced_X = tf.reduce_mean(filtered_X, axis=2) # [10, 784]

    diff = tf.reduce_mean(tf.square(reconstructed_image - reduced_X))

    return diff


class CapsNetwork(object):

    def __init__(self, conv1_filters, conv1_kernel, conv1_stride, # input_shape,
                convcaps_num, caps1_filters, caps1_kernel, caps1_stride, caps2_neurons, 
                digitcaps_num, predicting_units, routing_it,
                fc1_units, fc2_units, image_dims, T_C):

        self.digitcaps_num = digitcaps_num
        self.caps2_neurons = caps2_neurons

        self.conv1_filters = conv1_filters
        self.conv1_kernel = conv1_kernel
        self.conv1_stride = conv1_stride

        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.image_dims = image_dims
        self.aging = 0

        self.conv_caps = Primary_caps_layer(caps1_filters*convcaps_num, caps1_kernel, caps1_stride)
        self.digit_caps = Digit_caps_layer(caps1_filters, caps2_neurons, digitcaps_num, predicting_units, routing_it)

    def compute(self, X_image, batch_size, T_c, X, input_kind="standard", true_input=True):


        if input_kind=="standard":
            with tf.compat.v1.variable_scope('Encoder', reuse = tf.compat.v1.AUTO_REUSE):
                convolutional_layer = tf.compat.v1.layers.conv2d(X_image, self.conv1_filters, self.conv1_kernel, 
                                                self.conv1_stride, padding="VALID", activation=tf.nn.relu)

            
                convcaps_layer = self.conv_caps.compute(convolutional_layer, batch_size)
                digitcaps_layer = self.digit_caps.compute(convcaps_layer, batch_size)

            decoder_input, difference_error = masking(digitcaps_layer, T_c, self.digitcaps_num, self.caps2_neurons, true_input=true_input)
        
        elif input_kind=="fake":

            decoder_input = fake_masking(self.digitcaps_num, self.caps2_neurons)


        with tf.compat.v1.variable_scope("Decoder", reuse = tf.compat.v1.AUTO_REUSE):
            fc1 = tf.compat.v1.layers.dense(decoder_input, units=self.fc1_units, activation=tf.nn.relu)
            fc2 = tf.compat.v1.layers.dense(fc1, units=self.fc2_units, activation=tf.nn.relu)
            reconstructed_image = tf.compat.v1.layers.dense(fc2, units=self.image_dims,
                                                        activation=tf.sigmoid)

        if input_kind=="standard":

            if true_input == True:
                v_softmax, total_loss, reconstruction_err = loss(X, digitcaps_layer, reconstructed_image, T_c, batch_size, self.aging)
                return v_softmax, total_loss, reconstruction_err, reconstructed_image
            
            else:
                difference_loss, decoder_loss = diff_loss_fun(X, reconstructed_image, difference_error)
                self.aging += 1
                return difference_loss, decoder_loss, difference_error

        elif input_kind=="fake":
            average_error = avg_err_calc(reconstructed_image, X_image, T_c, batch_size, self.digitcaps_num)
            
            return average_error, reconstructed_image



        