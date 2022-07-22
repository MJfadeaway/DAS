from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import BR_lib.BR_layers as BR_layers
import BR_lib.BR_data as BR_data
import numpy as np

# DNN with fully connected layers.
class DNN(tf.keras.Model):
    def __init__(self, name, n_out, depth, width_list, **kwargs):
        super(DNN, self).__init__(name=name, **kwargs)
        self.n_out = n_out
        self.depth = depth
        self.width_list = width_list
        self.hidden_layers = []
        for i in range(depth):
            self.hidden_layers.append(BR_layers.Linear(str(i), width_list[i]))
        self.l_f = BR_layers.Linear('last',n_out)

    def call(self, inputs):
        x = inputs
        for i in range(self.depth):
            x = tf.nn.relu(self.hidden_layers[i](x))
        x = self.l_f(x)

        return x

    def get_gradient(self, inputs):
        x = tf.convert_to_tensor(inputs)
        with tf.GradientTape() as tape:
            tape.watch(x)
            prediction = self.call(x)
        return tape.gradient(prediction, x)

# DNN with the NN2 layers and actnorm layers.
class DNN_NN2(tf.keras.Model):
    def __init__(self, name, depth, width, n_out, **kwargs):
        super(DNN_NN2, self).__init__(name=name, **kwargs)
        self.n_out = n_out
        self.depth = depth
        self.width = width
        self.NN2_layers = []
        self.actnorm_layers = []
        for i in range(depth):
            self.NN2_layers.append(BR_layers.NN2('NN2_'+str(i), width, width))
            self.actnorm_layers.append(BR_layers.actnorm('actnorm_' + str(i)))
        self.l_NN2 = BR_layers.NN2('NN2_last', width, n_out)

    def call(self, inputs):
        x = inputs
        for i in range(self.depth):
            x =  x + self.NN2_layers[i](x)
            x = self.actnorm_layers[i](x)
            x = tf.nn.relu(x)
            #x = tf.nn.tanh(x)

        x = self.l_NN2(x)

        return x

    def get_gradient(self, inputs):
        x = tf.convert_to_tensor(inputs)
        with tf.GradientTape() as tape:
            tape.watch(x)
            prediction = self.call(x)
        return tape.gradient(prediction, x)

    def actnorm_data_initialization(self):
        for i in range(self.depth):
            #self.actnorm_layers[i].data_init = False
            self.actnorm_layers[i].reset_data_initialization()


# invertible mapping based on real NVP
class IM_rNVP(tf.keras.Model):
    def __init__(self, name, lb, hb, n_depth, n_split_at, n_width=32, flow_coupling=0,
	         rotation=False, bounded_supp=False, **kwargs):
        super(IM_rNVP, self).__init__(name=name,**kwargs)
        self.n_depth = n_depth
        self.n_split_at = n_split_at
        self.n_width = n_width
        self.flow_coupling = flow_coupling
        self.rotation = rotation
        self.bounded_supp = bounded_supp
        self.lb = lb
        self.hb = hb

	# two affine coupling layers are needed for each update of the vector
        assert n_depth % 2 == 0

        # add a logistic mapping to refine the data in a bounded support
        if bounded_supp:
            self.affine_linear = BR_layers.Affine_linear_mapping('affine_linear', self.lb, self.hb)
            self.logistic_mapping = BR_layers.Logistic_mapping('logi_mapping')

        # a rotation layer
        if rotation:
            self.W_LU = BR_layers.W_LU('w_lu')

        # stack affine coupling layers
        self.flow_mapping = BR_layers.flow_mapping('flow_mapping',
                                                   n_depth,
                                                   n_split_at,
                                                   n_width=n_width,
                                                   flow_coupling=flow_coupling)

        # the prior distribution
        self.log_prior = BR_data.log_standard_Gaussian
        #self.log_prior = BR_data.log_logistic

    # computing the logarithm of the estimated pdf on the input data.
    def call(self, inputs):
        objective = tf.zeros_like(inputs, dtype=tf.float32)[:,0]
        objective = tf.reshape(objective, [-1,1])

        z = inputs
        
        if self.bounded_supp:
            z, objective = self.affine_linear(z, objective)
            z, objective = self.logistic_mapping(z, objective, reverse=True)

        # unitary transformation
        if self.rotation:
            z, objective = self.W_LU(z, objective)

        # f(y) and log of jacobian
        z, objective = self.flow_mapping(z, objective)

        # logrithm of estimated pdf
        objective += self.log_prior(z)

        # add cross entropy to the loss function
        #CE = -tf.reduce_mean(objective)
        #self.add_loss(CE)
        return objective

    # return the first-order derivative wrt inputs
    def get_H1_regularization(self, inputs):
        x = tf.convert_to_tensor(inputs)
        with tf.GradientTape() as tape:
            tape.watch(x)
            objective = self.call(x)

        fx = tape.gradient(objective, x)
        #fx = BR_data.flatten_sum(tf.square(fx*tf.exp(objective/2.0)))
        #fx = BR_data.flatten_sum(tf.square(fx*tf.exp(-objective/2.0)))
        fx = BR_data.flatten_sum(tf.square(fx))
        return tf.reduce_mean(fx)

    # mapping from data to prior
    def mapping_to_prior(self, inputs, logdet=None):
        z = inputs
        if logdet is not None:
            if self.bounded_supp:
                z, logdet = self.affine_linear(z, logdet)
                z, logdet = self.logistic_mapping(z, logdet, reverse=True)
            if self.rotation:
                z, logdet = self.W_LU(z, logdet)
            z, logdet = self.flow_mapping(z, logdet)
            return z, logdet
        else:
            if self.bounded_supp:
                z = self.affine_linear(z)
                z = self.logistic_mapping(z, reverse=True)
            if self.rotation:
                z = self.W_LU(z)
            z = self.flow_mapping(z)
            return z

    # mapping from prior to data subject to the estimated distribution
    def mapping_from_prior(self, inputs):
        z = self.flow_mapping(inputs, reverse=True)
        if self.rotation:
            z = self.W_LU(z, reverse=True)
        if self.bounded_supp:
            z = self.logistic_mapping(z)
            z = self.affine_linear(z, reverse=True)
        return z

    def actnorm_data_initialization(self):
        self.flow_mapping.actnorm_data_initialization()

    def WLU_data_initialization(self):
        self.W_LU.reset_data_initialization()

    # return samples from prior
    def draw_samples_from_prior(self, n_samples, n_dim):
        return tf.random.normal((n_samples, n_dim))



# invertible mapping based on real NVP and KR rearrangement and CDF inverse
class IM_rNVP_KR_CDF(tf.keras.Model):
    def __init__(self, name, n_dim, lb, hb, n_step, n_depth,
                 n_width=32,
                 shrink_rate=1.0,
                 flow_coupling=0,
                 n_bins=16,
                 rotation=False,
                 bounded_supp=False,
                 **kwargs):
       super(IM_rNVP_KR_CDF, self).__init__(name=name,**kwargs)

       # two affine coupling layers are needed for each update of the vector
       assert n_depth % 2 == 0

       self.n_dim = n_dim # dimension of the data
       self.n_step = n_step # step size for dimension reduction
       self.n_depth = n_depth # depth for flow_mapping
       self.n_width = n_width
       self.n_bins = n_bins
       self.shrink_rate = shrink_rate
       self.flow_coupling = flow_coupling
       self.rotation = rotation
       self.bounded_supp = bounded_supp
       self.lb = lb
       self.hb = hb

       # the number of filtering stages
       self.n_stage = n_dim // n_step
       if n_dim % n_step > 0:
           self.n_stage += 1

       # the number of rotation layers
       self.n_rotation = self.n_stage
       #self.n_rotation = 1

       #self.affine_linear = BR_layers.Affine_linear_mapping('affine_linear', self.lb, self.hb)

       if rotation:
           self.rotations = []
           for i in range(self.n_rotation):
               # rotate the coordinate system for a better representation of data
               self.rotations.append(BR_layers.W_LU('rotation'+str(i)))

       # flow mapping with n_stage
       self.flow_mappings = []
       for i in range(self.n_stage):
           if i == (self.n_stage-1):
               self.flow_mappings.append(BR_layers.scale_and_CDF('scale_and_cdf', n_bins=n_bins))
           else:
               # flow_mapping given by such as real NVP
               n_split_at = n_dim - (i+1) * n_step
               self.flow_mappings.append(BR_layers.flow_mapping('flow_mapping'+str(i),
                                         n_depth,
                                         n_split_at,
                                         n_width=n_width,
                                         flow_coupling=flow_coupling,
                                         n_bins=n_bins))
               n_width = int(n_width*self.shrink_rate)

       # data will pass the squeezing layer at the end of each stage
       self.squeezing_layer = BR_layers.squeezing('squeezing', n_dim, n_step)

       # data will pass the bounded support mapping first (for general PDE not FP equation)
       self.bounded_support_layer = BR_layers.Bounded_support_mapping('bounded_for_pde', lb, hb)

       # the prior distribution is the Gaussian distribution
       self.log_prior = BR_data.log_standard_Gaussian
       #self.log_prior = BR_data.log_uniform

    # computing the logarithm of the estimated pdf on the input data.
    def call(self, inputs):
        objective = tf.zeros_like(inputs, dtype='float32')[:,0]
        objective = tf.reshape(objective, [-1,1])

        # f(y) and log of jacobian
        z, objective = self.mapping_to_prior(inputs, objective)

        # logrithm of estimated pdf
        objective += self.log_prior(z)

        return objective

    # mapping from data to prior
    def mapping_to_prior(self, inputs, logdet=None):
        z = inputs

        # data preprocessing using bounded support layer
        z = self.bounded_support_layer(z, logdet)
        if logdet is not None:
            z, logdet = z

        for i in range(self.n_stage):
            if logdet is not None:
                if self.rotation and i < self.n_rotation:
                    z, logdet = self.rotations[i](z, logdet)

                z, logdet = self.flow_mappings[i](z, logdet)
            else:
                if self.rotation and i < self.n_rotation:
                    z = self.rotations[i](z)

                z = self.flow_mappings[i](z)
            z = self.squeezing_layer(z)

        if logdet is not None:
            return z, logdet
        else:
            return z

    # mapping from prior to data
    def mapping_from_prior(self, inputs):
        z = inputs
        for i in reversed(range(self.n_stage)):
            z = self.squeezing_layer(z, reverse=True)
            z = self.flow_mappings[i](z, reverse=True)
                
            if self.rotation and i < self.n_rotation:
                z = self.rotations[i](z, reverse=True)

        # generate samples in domain [lb, hb]^d
        z = self.bounded_support_layer(z, reverse=True)
        return z

    # data initialization for actnorm layers
    def actnorm_data_initialization(self):
        for i in range(self.n_stage):
            self.flow_mappings[i].actnorm_data_initialization()

    # data initialization for rotation layers
    def WLU_data_initialization(self):
        #self.rotations[0].reset_data_initialization()
        for i in range(self.n_rotation):
            self.rotations[i].reset_data_initialization()

    # return the first-order derivative wrt inputs
    def get_H1_regularization(self, inputs):
        x = tf.convert_to_tensor(inputs)
        with tf.GradientTape() as tape:
            tape.watch(x)
            objective = self.call(x)
        fx = tape.gradient(objective, x)
        fx = BR_data.flatten_sum(tf.square(fx))
        return tf.reduce_mean(fx)

    # return samples from prior
    def draw_samples_from_prior(self, n_samples, n_dim):
        return tf.random.normal((n_samples, n_dim))
        #return tf.random.uniform([n_samples,n_dim], minval=-1, maxval=1, dtype=tf.float32)

