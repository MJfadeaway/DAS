from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
#import numpy as np

# default initializer
# def default_initializer(std=0.05):
#  return tf.random_normal_initializer(0., std)

def default_initializer():
  return tf.keras.initializers.GlorotNormal(seed=8)

#def default_initializer():
#  return tf.keras.initializers.Orthogonal()

# def default_initializer():
#  return tf.keras.initializers.GlorotUniform(seed=8)

#def default_initializer():
#  return tf.keras.initializers.VarianceScaling()


def entry_stop_gradients(target, mask):
    """
    mask specify which entries to be train
    """
    mask_stop = tf.logical_not(mask)
    mask = tf.cast(mask, dtype=target.dtype)
    mask_stop = tf.cast(mask_stop, dtype=target.dtype)

    return tf.stop_gradient(mask_stop * target) + mask * target

# This flow_mapping is for infinite support
# stacking actnorm, and affince coupling layers.
class flow_mapping(layers.Layer):
  def __init__(self, name, n_depth, n_split_at, n_width = 32, flow_coupling = 1, n_bins=16, **kwargs):
    super(flow_mapping, self).__init__(name=name,**kwargs)
    self.n_depth = n_depth
    self.n_split_at = n_split_at
    self.n_width = n_width
    self.flow_coupling = flow_coupling
    self.n_bins = n_bins

    # two affine coupling layers are needed for each update of the whole vector
    assert n_depth % 2 == 0

  def build(self, input_shape):
    self.n_length = input_shape[-1]
    self.scale_layers = []
    self.affine_layers = []

    sign = -1
    for i in range(self.n_depth):
      self.scale_layers.append(actnorm('actnorm_' + str(i)))
      sign *= -1
      i_split_at = (self.n_split_at*sign + self.n_length) % self.n_length
      self.affine_layers.append(affine_coupling('af_coupling_' + str(i),
                                                i_split_at,
                                                n_width=self.n_width,
                                                flow_coupling=self.flow_coupling))

    #self.cdf_layer = CDF_quadratic('cdf_layer', self.n_bins)

  # the default setting mapping the given data to the prior distribution
  # without computing the jacobian.
  def call(self, inputs, logdet=None, reverse=False):
    if not reverse:
      z = inputs
      for i in range(self.n_depth):
        z = self.scale_layers[i](z, logdet)
        if logdet is not None:
            z, logdet = z

        z = self.affine_layers[i](z, logdet)
        if logdet is not None:
            z, logdet = z

        z = z[:,::-1]

      # # # nolinear invertible mapping after affine coupling
      # i_split_at = self.n_split_at
      # z1 = z[:,:i_split_at]
      # z2 = z[:,i_split_at:]
      # z2 = self.cdf_layer(z2, logdet)
      # if logdet is not None:
      #   z2, logdet = z2
      # z = tf.concat([z1, z2], 1)
    else:
      z = inputs

      # # # nonlinear invertible mapping before affine coupling
      # i_split_at=self.n_split_at
      # z1 = z[:,:i_split_at]
      # z2 = z[:,i_split_at:]
      # z2 = self.cdf_layer(z2, logdet, reverse=True)
      # if logdet is not None:
      #   z2, logdet = z2
      # z = tf.concat([z1, z2], 1)

      for i in reversed(range(self.n_depth)):
        z = z[:,::-1]

        z = self.affine_layers[i](z, logdet, reverse=True)
        if logdet is not None:
            z, logdet = z

        z = self.scale_layers[i](z, logdet, reverse=True)
        if logdet is not None:
            z, logdet = z

    if logdet is not None:
        return z, logdet
    return z

  def actnorm_data_initialization(self):
    for i in range(self.n_depth):
        self.scale_layers[i].reset_data_initialization()

class scale_and_CDF(layers.Layer):
  def __init__(self, name, n_bins=16, **kwargs):
    super(scale_and_CDF, self).__init__(name=name,**kwargs)
    self.n_bins = n_bins

  def build(self, input_shape):
    self.scale_layer = actnorm('actnorm_s2c')
    self.cdf_layer = CDF_quadratic('cdf_s2c', self.n_bins)

  def call(self, inputs, logdet=None, reverse=False):
    z = inputs
    if not reverse:
      z = self.scale_layer(z, logdet)
      if logdet is not None:
        z, logdet = z

      z = self.cdf_layer(z, logdet)
      if logdet is not None:
        z, logdet = z
    else:
      z = self.cdf_layer(z, logdet, reverse=True)
      if logdet is not None:
        z, logdet = z

      z = self.scale_layer(z, logdet, reverse=True)
      if logdet is not None:
        z, logdet = z

    if logdet is not None:
      return z, logdet

    return z

  def actnorm_data_initialization(self):
    self.scale_layer.reset_data_initialization()

# Rotation layers: another version without solver
class W_LU(layers.Layer):
  def __init__(self, name, **kwargs):
    super(W_LU, self).__init__(name=name, **kwargs)
    self.data_init = True

  def build(self, input_shape):
    self.n_length = input_shape[-1]

    # lower- and upper-triangluar parts in one matrix.
    self.LU = self.add_weight(name='LU', shape=(self.n_length, self.n_length),
                              initializer=tf.zeros_initializer(),
                              dtype=tf.float32, trainable=True)

    # identity matrix
    self.LU_init = tf.eye(self.n_length, dtype=tf.float32)
    # permutation - identity matrix
    #self.P = tf.range(self.n_length)
    # inverse of self.P
    #self.invP = tf.math.invert_permutation(self.P)

  def call(self, inputs, logdet = None, reverse=False):
    x = inputs
    n_dim = x.shape[-1]

    # invP*L*U*x
    LU = self.LU_init + self.LU

    # upper-triangular matrix
    U = tf.linalg.band_part(LU,0,-1)

    # diagonal line
    U_diag = tf.linalg.tensor_diag_part(U)

    # trainable mask for U
    U_mask = (tf.linalg.band_part(tf.ones([n_dim, n_dim]), 0,-1) >= 1)
    U = entry_stop_gradients(U, U_mask)

    # lower-triangular matrix
    I = tf.eye(self.n_length,dtype=tf.float32)
    L = tf.linalg.band_part(I+LU,-1,0)-tf.linalg.band_part(LU,0,0)

    # trainable mask for L
    L_mask = (tf.linalg.band_part(tf.ones([n_dim, n_dim]), -1, 0) - tf.linalg.band_part(tf.ones([n_dim, n_dim]), 0, 0) >= 1)
    L = entry_stop_gradients(L, L_mask)

    if not reverse:
        x = tf.transpose(x)
        x = tf.linalg.matmul(U,x)
        x = tf.linalg.matmul(L,x)
        #x = tf.gather(x, self.invP)
        x = tf.transpose(x)
    else:
        x = tf.transpose(x)
        #x = tf.gather(x, self.P)
        x = tf.linalg.matmul(tf.linalg.inv(L), x)
        x = tf.linalg.matmul(tf.linalg.inv(U), x)

        #x = tf.linalg.triangular_solve(L, x, lower=True)
        #x = tf.linalg.triangular_solve(U, x, lower=False)
        x = tf.transpose(x)

    if logdet is not None:
        dlogdet = tf.reduce_sum(tf.math.log(tf.math.abs(U_diag)))
        if reverse:
            dlogdet *= -1.0
        return x, logdet + dlogdet

    return x

  def reset_data_initialization(self):
    self.data_init = False

# actnorm layer: centering and scaling layer - simplification of batchnormalization
class actnorm(layers.Layer):
  def __init__(self, name, scale = 1.0, logscale_factor = 3.0, **kwargs):
    super(actnorm, self).__init__(name=name,**kwargs)
    self.scale = scale
    self.logscale_factor = logscale_factor

    self.data_init = True
    #self.data_init = False

  def build(self, input_shape):
    self.n_length = input_shape[-1]
    self.b     = self.add_weight(name='b', shape=(1, self.n_length),
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=True)
    self.b_init = self.add_weight(name='b_init', shape=(1, self.n_length),
                                  initializer=tf.zeros_initializer(),
                                  dtype=tf.float32, trainable=False)

    self.logs  = self.add_weight(name='logs', shape=(1, self.n_length),
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=True)
    self.logs_init = self.add_weight(name='logs_init', shape=(1, self.n_length),
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=False)


  def call(self, inputs, logdet = None, reverse = False):
    # data initialization
    # by default, no data initialization is implemented.
    if not self.data_init:
        x_mean = tf.reduce_mean(inputs, [0], keepdims=True)
        x_var = tf.reduce_mean(tf.square(inputs-x_mean), [0], keepdims=True)

        self.b_init.assign(-x_mean)
        self.logs_init.assign(tf.math.log(self.scale/(tf.sqrt(x_var)+1e-6))/self.logscale_factor)

        self.data_init = True

    if not reverse:
      x = inputs + (self.b + self.b_init)
      #x = x * tf.exp(self.logs + self.logs_init)
      x = x * tf.exp(tf.clip_by_value(self.logs + self.logs_init, -5., 5.))
    else:
      #x = inputs * tf.exp(-self.logs - self.logs_init)
      x = inputs * tf.exp(-tf.clip_by_value(self.logs + self.logs_init, -5., 5.))
      x = x - (self.b + self.b_init)

    if logdet is not None:
      #dlogdet = tf.reduce_sum(self.logs + self.logs_init)
      dlogdet = tf.reduce_sum(tf.clip_by_value(self.logs + self.logs_init, -5., 5.))
      if reverse:
        dlogdet *= -1
      return x, logdet + dlogdet

    return x

  def reset_data_initialization(self):
    self.data_init = False

# affine coupling layer
class affine_coupling(layers.Layer):
  def __init__(self, name, n_split_at, n_width = 32, flow_coupling = 1, **kwargs):
    super(affine_coupling, self).__init__(name=name, **kwargs)
    # partition as [:n_split_at] and [n_split_at:]
    self.n_split_at = n_split_at
    self.flow_coupling = flow_coupling
    self.n_width = n_width

  def build(self, input_shape):
    n_length = input_shape[-1]
    if self.flow_coupling == 0:
      self.f = NN2('a2b', self.n_width, n_length-self.n_split_at)
    elif self.flow_coupling == 1:
      self.f = NN2('a2b', self.n_width, (n_length-self.n_split_at)*2)
    else:
      raise Exception()
    self.log_gamma  = self.add_weight(name='log_gamma',
                                      shape=(1, n_length-self.n_split_at),
                                      initializer=tf.zeros_initializer(),
                                      dtype=tf.float32, trainable=True)

  # the default setting performs a mapping of the data
  # without computing the jacobian
  def call(self, inputs, logdet=None, reverse=False):
    z = inputs
    n_split_at = self.n_split_at

    alpha = 0.6

    if not reverse:
      z1 = z[:,:n_split_at]
      z2 = z[:,n_split_at:]

      if self.flow_coupling == 0:
        shift = self.f(z1)
        shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
        z2 += shift
      elif self.flow_coupling == 1:
        h = self.f(z1)
        shift = h[:,0::2]

        # resnet-like trick
        # we suppressed both the scale and the shift.
        scale = alpha*tf.nn.tanh(h[:,1::2])
        #shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
        shift = tf.exp(tf.clip_by_value(self.log_gamma, -5.0, 5.0))*tf.nn.tanh(shift)
        z2 = z2 + scale * z2 + shift
        if logdet is not None:
           logdet += tf.reduce_sum(tf.math.log(scale + tf.ones_like(scale)),
                                   axis=[1], keepdims=True)
      else:
        raise Exception()

      z = tf.concat([z1, z2], 1)
    else:
      z1 = z[:,:n_split_at]
      z2 = z[:,n_split_at:]

      if self.flow_coupling == 0:
        shift = self.f(z1)
        shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
        z2 -= shift
      elif self.flow_coupling == 1:
        h = self.f(z1)
        shift = h[:,0::2]

        # resnet-like trick
        # we suppressed both the scale and the shift.
        scale = alpha*tf.nn.tanh(h[:,1::2])
        #shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
        shift = tf.exp(tf.clip_by_value(self.log_gamma, -5.0, 5.0))*tf.nn.tanh(shift)
        z2 = (z2 - shift) / (tf.ones_like(scale) + scale)
        if logdet is not None:
           logdet -= tf.reduce_sum(tf.math.log(scale + tf.ones_like(scale)),
                                   axis=[1], keepdims=True)
      else:
        raise Exception()

      z = tf.concat([z1, z2], 1)

    if logdet is not None:
        return z, logdet

    return z

# squeezing layer - KR rearrangement
class squeezing(layers.Layer):
    def __init__(self, name, n_dim, n_cut=1, **kwargs):
        super(squeezing, self).__init__(name=name, **kwargs)
        self.n_dim = n_dim
        self.n_cut = n_cut
        self.x = None

    def call(self, inputs, reverse=False):
        z = inputs
        n_length = z.get_shape()[-1]

        if not reverse:
            if n_length < self.n_cut:
                raise Exception()

            if self.n_dim == n_length:
                if self.n_dim > self.n_cut:
                    if self.x is not None:
                        raise Exception()
                    else:
                        self.x = z[:, (n_length - self.n_cut):]
                        z = z[:, :(n_length - self.n_cut)]
                else:
                    self.x = None
            elif n_length <= self.n_cut:
                z = tf.concat([z, self.x], 1)
                self.x = None
            else:
                cut = z[:, (n_length - self.n_cut):]
                self.x = tf.concat([cut, self.x], 1)
                z = z[:, :(n_length - self.n_cut)]
        else:
            if self.n_dim == n_length:
                n_start = self.n_dim % self.n_cut
                if n_start == 0:
                    n_start += self.n_cut
                self.x = z[:, n_start:]
                z = z[:, :n_start]

            else:
                x_length = self.x.get_shape()[-1]
                if x_length < self.n_cut:
                    raise Exception()

                cut = self.x[:, :self.n_cut]
                z = tf.concat([z, cut], 1)

                if (x_length - self.n_cut) == 0:
                    self.x = None
                else:
                    self.x = self.x[:, self.n_cut:]
        return z

# one linear layer with defaul width 32.
class Linear(layers.Layer):
  def __init__(self, name, n_width=32, **kwargs):
    super(Linear, self).__init__(name=name, **kwargs)
    self.n_width = n_width

  def build(self, input_shape):
    n_length = input_shape[-1]
    self.w = self.add_weight(name='w', shape=(n_length, self.n_width),
                             initializer=default_initializer(),
                             dtype=tf.float32, trainable=True)
    self.b = self.add_weight(name='b', shape=(self.n_width,),
                             initializer=tf.zeros_initializer(),
                             dtype=tf.float32, trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

# two-hidden-layer neural network
class NN2(layers.Layer):
  def __init__(self, name, n_width=32, n_out=None, **kwargs):
    super(NN2, self).__init__(name=name, **kwargs)
    self.n_width = n_width
    self.n_out = n_out

  def build(self, input_shape):
    self.l_1 = Linear('h1', self.n_width)
    self.l_2 = Linear('h2', self.n_width)
    #self.l_2 = Linear('h2', self.n_width//2)
    #self.l_3 = Linear('h3', self.n_width//2)
    #self.l_4 = Linear('h4', self.n_width)

    n_out = self.n_out or int(input_shape[-1])
    self.l_f = Linear('last', n_out)

  def call(self, inputs):
    # relu with low regularity
    x = tf.nn.relu(self.l_1(inputs))
    x = tf.nn.relu(self.l_2(x))

    # tanh with high regularity
    #x = tf.nn.tanh(self.l_1(inputs))
    #x = tf.nn.tanh(self.l_2(x))
    #x = tf.nn.relu(self.l_3(x))
    #x = tf.nn.relu(self.l_4(x))

    x = self.l_f(x)

    return x


# affine linear mapping from a bounded domain [lb, hb] to [0,1]^d
class Affine_linear_mapping(layers.Layer):
    def __init__(self, name, lb, hb, **kwargs):
        super(Affine_linear_mapping, self).__init__(name=name, **kwargs)
        self.lb = lb
        self.hb = hb

    def call(self, inputs, logdet=None, reverse=False):
        x = inputs
        n_dim = x.shape[-1]
        # mapping from [lb, hb] to [0,1]^d for PDE: y = (x-l) / (h - l)
        if not reverse:
            x = x / (self.hb - self.lb)
            x = x - self.lb / (self.hb - self.lb)
        else:
            x = x + self.lb / (self.hb - self.lb)
            x = x * (self.hb - self.lb)

        if logdet is not None:
            dlogdet = n_dim * tf.math.log(1.0/(self.hb-self.lb)) * tf.reshape(tf.ones_like(x[:,0], dtype=tf.float32), [-1,1])
            if reverse:
                dlogdet *= -1.0
            return x, logdet + dlogdet

        return x


class Logistic_mapping(layers.Layer):
    """
    Logistic mapping, (-inf, inf) --> (0, 1):
    y = (tanh(x/2) + 1) / 2 = e^x/(1 + e^x)
    derivate: dy/dx = y* (1-y)
    inverse: x = log(y) - log(1-y)

    For PDE, data to prior direction: [a,b] ---> (-inf, inf)
    So we need to use an affine linear mapping first and then use logistic mapping
    """
    def __init__(self, name, **kwargs):
        super(Logistic_mapping, self).__init__(name=name, **kwargs)
        self.s_init = 2.0

    def build(self, input_shape):
        n_length = input_shape[-1]
        self.s = self.add_weight(name='logistic_s', shape=(1, n_length),
                                 initializer = tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=False)

    # the direction of this mapping is not related to the flow
    # direction between the data and the prior
    def call(self, inputs, logdet=None, reverse=False):
        x = inputs

        if not reverse:
            x = tf.clip_by_value(x, 1.0e-6, 1.0-1.0e-6)
            tp1 = tf.math.log(x)
            tp2 = tf.math.log(1 - x)
            #x = (self.s_init + self.s) / 2.0 * (tp1 - tp2)
            x = tp1 - tp2
            if logdet is not None:
                #tp =  tf.math.log((self.s+self.s_init)/2.0) - tp1 - tp2
                tp =  - tp1 - tp2
                dlogdet = tf.reduce_sum(tp, axis=[1], keepdims=True)
                return x, logdet + dlogdet

        else:
            x = (tf.nn.tanh(x / (self.s_init + self.s)) + 1.0) / 2.0

            if logdet is not None:
                x = tf.clip_by_value(x, 1.0e-6, 1.0-1.0e-6)
                #tp = tf.math.log(x) + tf.math.log(1-x) + tf.math.log(2.0/(self.s+self.s_init))
                tp = tf.math.log(x) + tf.math.log(1-x)
                dlogdet = tf.reduce_sum(tp, axis=[1], keepdims=True)
                return x, logdet + dlogdet
 
        return x


## bounded support mapping: logistic mapping from (-inf, inf) to (0,1), affine linear (lb,hb) to (0,1)
## this is for resample of PDE domain
class Bounded_support_mapping(layers.Layer):
    def __init__(self, name, lb, hb, **kwargs):
        super(Bounded_support_mapping, self).__init__(name=name, **kwargs)
        self.lb = lb
        self.hb = hb

    def build(self, input_shape):
        self.n_length = input_shape[-1]
        self.logistic_layer = Logistic_mapping('logistic_for_pde')
        self.affine_linear_layer = Affine_linear_mapping('affine_linear_for_pde', self.lb, self.hb)

    def call(self, inputs, logdet=None, reverse=False):
        x = inputs

        if not reverse:
            x = self.affine_linear_layer(x, logdet)
            if logdet is not None:
                x, logdet = x
            x = self.logistic_layer(x, logdet)
            if logdet is not None:
                x, logdet = x

        else:
            x = self.logistic_layer(x, logdet, reverse=True)
            if logdet is not None:
                x, logdet = x
            x = self.affine_linear_layer(x, logdet, reverse=True)
            if logdet is not None:
                x, logdet = x

        if logdet is not None:
            return x, logdet

        return x

# mapping defined by a piecewise quadratic cumulative distribution function (CDF)
# Assume that each dimension has a compact support [0,1]
# CDF(x) maps [0,1] to [0,1], where the prior uniform distribution is defined.
# Since x is defined on (-inf,+inf), we only consider a CDF() mapping from
# the interval [-bound, bound] to [-bound, bound], and leave alone other points.
# The reason we do not consider a mapping from (-inf,inf) to (0,1) is the
# singularity induced by the mapping.
class CDF_quadratic(layers.Layer):
    def __init__(self, name, n_bins, r=1.2, bound=50.0, **kwargs):
        super(CDF_quadratic, self).__init__(name=name, **kwargs)

        assert n_bins % 2 == 0

        self.n_bins = n_bins

        # generate a nonuniform mesh symmetric to zero,
        # and increasing by ratio r away from zero.
        self.bound = bound
        self.r = r

        m = n_bins/2
        x1L = bound*(r-1.0)/(tf.math.pow(r, m)-1.0)

        index = tf.reshape(tf.range(0, self.n_bins+1, dtype=tf.float32),(-1,1))
        index -= m
        xr = tf.where(index>=0, (1.-tf.math.pow(r, index))/(1.-r),
                      (1.-tf.math.pow(r,tf.math.abs(index)))/(1.-r))
        xr = tf.where(index>=0, x1L*xr, -x1L*xr)
        xr = tf.reshape(xr,(-1,1))
        xr = (xr + bound)/2.0/bound

        self.x1L = x1L/2.0/bound
        self.mesh = tf.concat([tf.reshape([0.0],(-1,1)), tf.reshape(xr[1:-1,0],(-1,1)), tf.reshape([1.0],(-1,1))],0) 
        self.elmt_size = tf.reshape(self.mesh[1:] - self.mesh[:-1],(-1,1))

    def build(self, input_shape):
        self.n_length = input_shape[-1]
        self.p = self.add_weight(name='pdf', shape=(self.n_bins-1, self.n_length),
                                   initializer=tf.zeros_initializer(),
                                   dtype=tf.float32, trainable=True)

    def call(self, inputs, logdet=None, reverse=False):
        # normalize the PDF
        self._pdf_normalize()

        x = inputs
        if not reverse:
            # rescale such points in [-bound, bound] will be mapped to [0,1]
            x = (x + self.bound) / 2.0 / self.bound

            # cdf mapping
            x = self._cdf(x, logdet)
            if logdet is not None:
                x, logdet = x

            # maps [0,1] back to [-bound, bound]
            x = x * 2.0 * self.bound - self.bound
        else:
            # rescale such points in [-bound, bound] will be mapped to [0,1]
            x = (x + self.bound) / 2.0 / self.bound

            # cdf mapping
            x = self._cdf_inv(x, logdet)
            if logdet is not None:
                x, logdet = x

            # maps [0,1] back to [-bound, bound]
            x = x * 2.0 * self.bound - self.bound
        if logdet is not None:
            return x, logdet

        return x

    # normalize the piecewise representation of pdf
    def _pdf_normalize(self):
        # peicewise pdf
        p0 = tf.ones((1,self.n_length), dtype=tf.float32)
        self.pdf = p0
        px = tf.math.exp(self.p)*(self.elmt_size[:-1]+self.elmt_size[1:])/2.0
        px = (1 - self.elmt_size[0])/tf.reduce_sum(px, 0, keepdims=True)
        px = px*tf.math.exp(self.p)
        self.pdf = tf.concat([self.pdf, px], 0)
        self.pdf = tf.concat([self.pdf, p0], 0)

        # probability in each element
        cell = (self.pdf[:-1,:] + self.pdf[1:,:])/2.0*self.elmt_size
        # CDF - contribution from previous elements.
        r_zeros= tf.zeros((1,self.n_length), dtype=tf.float32)
        self.F_ref = r_zeros
        for i in range(1, self.n_bins):
            tp  = tf.math.reduce_sum(cell[:i,:], 0, keepdims=True)
            self.F_ref = tf.concat([self.F_ref, tp], 0)

    # the cdf is a piecewise quadratic function.
    def _cdf(self, x, logdet=None):
        x_sign = tf.math.sign(x-0.5)
        m = tf.math.floor(tf.math.log(tf.math.abs(x-0.5)*(self.r-1)/self.x1L + 1.0)/tf.math.log(self.r))
        k_ind = tf.where(x_sign >= 0, self.n_bins/2 + m, self.n_bins/2 - m - 1)
        k_ind = tf.cast(k_ind, tf.int32)

        cover = tf.where(k_ind*(k_ind-self.n_bins+1)<=0, 1.0, 0.0)

        k_ind = tf.where(k_ind < 0, 0, k_ind)
        k_ind = tf.where(k_ind > (self.n_bins-1), self.n_bins-1, k_ind)

        v1 = tf.reshape(tf.gather(self.pdf[:,0], k_ind[:,0]),(-1,1))
        for i in range(1, self.n_length):
            tp = tf.reshape(tf.gather(self.pdf[:,i], k_ind[:,i]),(-1,1))
            v1 = tf.concat([v1, tp], 1)

        v2 = tf.reshape(tf.gather(self.pdf[:,0], k_ind[:,0]+1),(-1,1))
        for i in range(1, self.n_length):
            tp = tf.reshape(tf.gather(self.pdf[:,i], k_ind[:,i]+1),(-1,1))
            v2 = tf.concat([v2, tp], 1)

        xmodi = tf.reshape(x[:,0] - tf.gather(self.mesh[:,0], k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = tf.reshape(x[:,i] - tf.gather(self.mesh[:,0], k_ind[:, i]), (-1, 1))
            xmodi = tf.concat([xmodi, tp], 1)

        h_list = tf.reshape(tf.gather(self.elmt_size[:,0], k_ind[:,0]),(-1,1))
        for i in range(1, self.n_length):
            tp = tf.reshape(tf.gather(self.elmt_size[:,0], k_ind[:,i]),(-1,1))
            h_list = tf.concat([h_list, tp], 1)

        F_pre = tf.reshape(tf.gather(self.F_ref[:, 0], k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = tf.reshape(tf.gather(self.F_ref[:, i], k_ind[:, i]), (-1, 1))
            F_pre = tf.concat([F_pre, tp], 1)

        y = tf.where(cover>0, F_pre + xmodi**2/2.0*(v2-v1)/h_list + xmodi*v1, x)
       
        if logdet is not None:
            dlogdet = tf.where(cover > 0, xmodi * (v2 - v1) / h_list + v1, 1.0)
            dlogdet = tf.reduce_sum(tf.math.log(dlogdet), axis=[1], keepdims=True)
            return y, logdet + dlogdet

        return y

    # inverse of the cdf
    def _cdf_inv(self, y, logdet=None):
        xr = tf.broadcast_to(self.mesh, [self.n_bins+1, self.n_length])
        yr1 = self._cdf(xr)

        p0 = tf.zeros((1,self.n_length), dtype=tf.float32)
        p1 = tf.ones((1,self.n_length), dtype=tf.float32)
        yr = tf.concat([p0, yr1[1:-1,:], p1], 0)

        k_ind = tf.searchsorted(tf.transpose(yr), tf.transpose(y), side='right')
        k_ind = tf.transpose(k_ind)
        k_ind = tf.cast(k_ind, tf.int32)
        k_ind -= 1

        cover = tf.where(k_ind*(k_ind-self.n_bins+1)<=0, 1.0, 0.0)

        k_ind = tf.where(k_ind < 0, 0, k_ind)
        k_ind = tf.where(k_ind > (self.n_bins-1), self.n_bins-1, k_ind)

        c_cover = tf.reshape(cover[:,0], (-1,1))
        v1 = tf.where(c_cover > 0, tf.reshape(tf.gather(self.pdf[:,0], k_ind[:,0]),(-1,1)), -1.0)
        for i in range(1, self.n_length):
            c_cover = tf.reshape(cover[:,i], (-1,1))
            tp = tf.where(c_cover > 0, tf.reshape(tf.gather(self.pdf[:,i], k_ind[:,i]),(-1,1)), -1.0)
            v1 = tf.concat([v1, tp], 1)

        c_cover = tf.reshape(cover[:,0], (-1,1))
        v2 = tf.where(c_cover > 0, tf.reshape(tf.gather(self.pdf[:,0], k_ind[:,0]+1),(-1,1)), -2.0)
        for i in range(1, self.n_length):
            c_cover = tf.reshape(cover[:,i], (-1,1))
            tp = tf.where(c_cover > 0, tf.reshape(tf.gather(self.pdf[:,i], k_ind[:,i]+1),(-1,1)), -2.0)
            v2 = tf.concat([v2, tp], 1)

        ys = tf.reshape(y[:, 0] - tf.gather(yr[:, 0], k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = tf.reshape(y[:, i] - tf.gather(yr[:, i], k_ind[:, i]), (-1, 1))
            ys = tf.concat([ys, tp], 1)

        xs = tf.reshape(tf.gather(xr[:, 0], k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = tf.reshape(tf.gather(xr[:, i], k_ind[:, i]), (-1, 1))
            xs = tf.concat([xs, tp], 1)

        h_list = tf.reshape(tf.gather(self.elmt_size[:,0], k_ind[:,0]),(-1,1))
        for i in range(1, self.n_length):
            tp = tf.reshape(tf.gather(self.elmt_size[:,0], k_ind[:,i]),(-1,1))
            h_list = tf.concat([h_list, tp], 1)

        tp = 2.0*ys*h_list*(v2-v1)
        tp += v1*v1*h_list*h_list
        tp = tf.sqrt(tp) - v1*h_list
        tp = tf.where(tf.math.abs(v1-v2)<1.0e-6, ys/v1, tp/(v2-v1))
        tp += xs

        x = tf.where(cover > 0, tp, y)

        if logdet is not None:
            tp = 2.0 * ys * h_list * (v2 - v1)
            tp += v1 * v1 * h_list * h_list
            tp = h_list/tf.sqrt(tp)

            dlogdet = tf.where(cover > 0, tp, 1.0)
            dlogdet = tf.reduce_sum(tf.math.log(dlogdet), axis=[1], keepdims=True)
            return x, logdet + dlogdet

        return x
