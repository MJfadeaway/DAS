from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras import layers

#import numpy as np
# default initializer
#def default_initializer(std=0.05):
#    return tf.random_normal_initializer(0., std)
# def nn_initializer():
#     return tf.keras.initializers.GlorotNormal(seed=8)

def nn_initializer():
    return tf.keras.initializers.GlorotUniform(seed=8)

# def nn_initializer():
#      return tf.keras.initializers.GlorotNormal(seed=8)


def sin_act(x):
    """
    Sine activation function
    """
    return tf.math.sin(x)



# Basic operation for neural networks: linear layers
class Linear(layers.Layer):
    def __init__(self, name, n_hidden=32, **kwargs):
        super(Linear, self).__init__(name=name, **kwargs)
        self.n_hidden = n_hidden


    def build(self, input_shape):
        # running a initialization (pass the data to a network) will get input_shape
        n_length = input_shape[-1]
        self.w = self.add_weight(name='w', shape=(n_length, self.n_hidden),
                                 initializer=nn_initializer(),
                                 dtype=tf.float32, trainable=True)
        self.b = self.add_weight(name='b', shape=(self.n_hidden,),
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=True)


    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b



# Fully connected neural networks
class FCNN(tf.keras.Model):
    def __init__(self, name, n_out, depth, n_hidden, act='tanh', **kwargs):
        super(FCNN, self).__init__(name=name, **kwargs)
        self.n_out = n_out
        self.depth = depth
        self.n_hidden = n_hidden
        self.act = act
        self.hidden_layers = []
        for i in range(depth):
            self.hidden_layers.append(Linear(str(i), n_hidden))
        self.l_f = Linear('last',n_out)


    def call(self, inputs):
        x = inputs
        for i in range(self.depth):
            if self.act == 'relu':
                x = tf.nn.relu(self.hidden_layers[i](x))
            elif self.act == 'tanh':
                x = tf.nn.tanh(self.hidden_layers[i](x))
            # softplus function: log(1 + exp(x)) can be viewed as soft relu but it is smooth 
            elif self.act == 'softplus':
                x = tf.nn.softplus(self.hidden_layers[i](x))
            elif self.act == 'sin':
                x = sin_act(self.hidden_layers[i](x))
        x = self.l_f(x)

        return x


# Other architecture by invsitigating the underlying structure of PDEs 
## TODO



