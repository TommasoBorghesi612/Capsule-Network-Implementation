import numpy as np
import tensorflow as tf
import random

epsilon = 1e-9

def squash(vector):

    vec_abs = tf.reduce_sum(tf.square(vector), axis=-2, keepdims=True)  # a scalar
    scalar_factor = vec_abs / (1 + vec_abs) / tf.sqrt(vec_abs + epsilon) 
    vec_squashed = vector * scalar_factor  # element-wise

    return(vec_squashed)


def init_weights(shape):
    init_random_dist = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

class Primary_caps_layer(object):

    def __init__(self, filters, kernel_size, stride, padding='VALID', activation_fn=tf.nn.relu):

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # self.primary_capsules = tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.stride, activation=tf.nn.relu)

    
    def compute(self, input_x, batch_size):

        with tf.compat.v1.variable_scope('ConvCaps_layer'):

            caps1 = tf.compat.v1.layers.conv2d(input_x, self.filters, self.kernel_size, 
                                            self.stride, padding="VALID", activation=tf.nn.relu)

        # caps1 = self.primary_capsules(input_x)
        caps1 = tf.reshape(caps1, [batch_size, -1, self.filters, 1])
        primary_capsules = squash(caps1)

        return primary_capsules



class Digit_caps_layer(object):
    
    def __init__(self, caps1_filters, caps2_neurons, digitcaps_num, predicting_units, routing_it):

        self.digitcaps_num = digitcaps_num

        with tf.compat.v1.variable_scope('Encoder'):
            self.weights = init_weights([1, predicting_units, caps2_neurons*digitcaps_num, caps1_filters, 1]) # [1, 1152, 160, 8, 1]
            # self.biases = init_bias([1, 1, digitcaps_num, caps2_neurons, 1]) # [1, 10, 16, 1]

        self.routing_it = routing_it
        self.predicting_units = predicting_units
        self.caps1_filters = caps1_filters
        self.caps2_neurons = caps2_neurons

    
    def routing(self, u_hat, batch_size):

        # new input : [batch_size, 1152, 10, 16]
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
        B = tf.zeros(shape=[batch_size, self.predicting_units, self.digitcaps_num, 1, 1], dtype=np.float32) # [batch_size, 1152, 10, 1, 1] 
        for i in range(self.routing_it):

            C = tf.nn.softmax(B, axis=2)                #C shape = [batch_size, self.predicting_units, 10, 1, 1] 
            if i == self.routing_it-1 :
                s = tf.reduce_sum(tf.multiply(u_hat, C),  #u_hat shape = [batch_size, self.predicting_units, 10, 16, 1]
                                axis=1, keepdims=True)                   
                # s += self.biases    #s shape = [batch_size, 1, 10, 16, 1]
                v = squash(s)    #v shape = [batch_size, 1, 10, 16, 1]

            else:
                s = tf.reduce_sum(tf.multiply(u_hat_stopped, C),  #u_hat shape = [batch_size, self.predicting_units, 10, 16]
                                axis=1, keepdims=True)
                # s += self.biases    #s shape = [batch_size, 1, 10, 16, 1]
                v = squash(s)    #v shape = [batch_size, 1, 10, 16, 1]

                # v = tf.reshape(v, [batch_size, 1, self.digitcaps_num,  self.caps2_neurons, 1])    # v shape =  [batch_size, 1, 10, 16, 1]
                v_tiled = tf.tile(v, [1, self.predicting_units, 1, 1, 1]) # v_tiled shape =  [batch_size, self.predicting_units, 10, 16, 1]
                u_mult_v = tf.reduce_sum(u_hat_stopped * v_tiled, axis=3, keepdims=True) # [batch_size, 1152, 10, 1, 1]
                # u_mult_v = tf.reshape(u_mult_v, [batch_size, self.predicting_units, self.digitcaps_num, 1]) # [batch_size, 1152, 10, 1] 
                B += u_mult_v

        return(v)

    
    def compute(self, input_x, batch_size):

        input_x = tf.reshape(input_x, [batch_size, self.predicting_units, 1, self.caps1_filters, 1]) # [batch_size, 1152, 1, 8, 1]
        input_x = tf.tile(input_x, [1, 1, self.digitcaps_num * self.caps2_neurons, 1, 1]) # [batch_size, 1152, 160, 8, 1]
        output = tf.reduce_sum(self.weights * input_x, axis=3, keepdims=True) # [batch_size, 1152, 160, 1]
        output = tf.reshape(output, shape=[batch_size, self.predicting_units, self.digitcaps_num, self.caps2_neurons, 1]) # [batch_size, 1152, 10, 16, 1]
        digitcaps_layer = self.routing(output, batch_size)
        digitcaps_layer = tf.squeeze(digitcaps_layer, axis=1)
        return digitcaps_layer


