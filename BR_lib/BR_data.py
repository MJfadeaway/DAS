from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras import layers
import numpy as np
import random

# a wrapper for tensorflow.dataset
class dataflow(object):
    def __init__(self, x, buffersize, batchsize, y=None):
        self.x = x
        self.y = y
        self.buffersize = buffersize
        self.batchsize = batchsize

        if y is not None:
            dx = tf.data.Dataset.from_tensor_slices(x)
            dy = tf.data.Dataset.from_tensor_slices(y)
            self.dataset = tf.data.Dataset.zip((dx, dy))
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices(x)

        self.batched_dataset = self.dataset.batch(batchsize)
        self.shuffled_batched_dataset = self.dataset.shuffle(buffersize).batch(batchsize)
        #self.shuffled_batched_dataset = self.dataset.shuffle(buffersize).batch(batchsize).map(lambda x: x, num_parallel_calls=8).prefetch(buffer_size=1)

    def get_shuffled_batched_dataset(self):
        return self.shuffled_batched_dataset

    def get_batched_dataset(self):
        return self.batched_dataset

    def update_shuffled_batched_dataset(self):
        self.shuffled_batched_dataset = self.dataset.shuffle(self.buffersize).batch(self.batchsize)
        return self.shuffled_batched_dataset

    def get_n_batch_from_shuffled_batched_dataset(self, n):
        it = iter(self.shuffled_batched_dataset)
        xs = []
        for i in range(n):
            x = next(it)
            if isinstance(x, tuple):
              xs.append(x[0])
            else:
              xs.append(x)
        x = tf.concat(xs, 0)

        return x

    def get_n_batch_from_batched_dataset(self, n):
        it = iter(self.batched_dataset)
        xs = []
        for i in range(n):
            x = next(it)
            if isinstance(x, tuple):
              xs.append(x[0])
            else:
              xs.append(x)
        x = tf.concat(xs, 0)

        return x

# when threshold > 0, it returns a Gaussian with an ellipse hole
def gen_2d_Gaussian_w_hole(ndim, n_train, alpha, theta, threshold, weighted=False):
    assert ndim == 2
    m  = n_train

    x = np.zeros((m,ndim), dtype='float32')

    Rs = np.zeros((ndim,ndim), dtype='float32')
    Rs[0,0] = alpha * np.cos(theta)
    Rs[0,1] = -alpha * np.sin(theta)
    Rs[1,0] = np.sin(theta)
    Rs[1,1] = np.cos(theta)
 
    #tf.random.set_seed(12345)
    for i in range(m):
        while True:
            z = np.random.normal(0,1,ndim)
            y = np.matmul(Rs,z)
            if np.linalg.norm(y,2) >= threshold:
                x[i,:] = z
                break

    if weighted is True:
        w = np.ones((m,1), dtype='float32')
        beta = 0.5
        for i in range(m):
            y = np.matmul(Rs,x[i,:])
            w[i,0] = np.exp(beta*(np.linalg.norm(y,2)-threshold))

        s = np.sum(w)
        for i in range(m):
            w[i,0] = w[i,0]*m/s
        
        return x,w

    return x


def gen_2d_Logistic_w_hole(ndim, n_train, scale, alpha, theta, threshold, weighted=False):
    assert ndim == 2
    m  = n_train

    x = np.zeros((m,ndim), dtype='float32')

    Rs = np.zeros((ndim,ndim), dtype='float32')
    Rs[0,0] = alpha * np.cos(theta)
    Rs[0,1] = -alpha * np.sin(theta)
    Rs[1,0] = np.sin(theta)
    Rs[1,1] = np.cos(theta)

    #tf.random.set_seed(12345)
    for i in range(m):
        while True:
            z = np.random.logistic(0,scale,ndim)
            z[1] = np.sign(z[0])*np.power(np.abs(z[0]), 0.5, dtype='float32') + z[1]*2.0/5.0
            y = np.matmul(Rs,z)
            if np.linalg.norm(y,2) >= threshold:
                x[i,:] = z
                break

    if weighted is True:
        w = np.ones((m,1), dtype='float32')
        beta = 0.5
        for i in range(m):
            y = np.matmul(Rs,x[i,:])
            w[i,0] = np.exp(beta*(np.linalg.norm(y,2)-threshold))

        s = np.sum(w)
        for i in range(m):
            w[i,0] = w[i,0]*m/s

        return x,w

    return x

def gen_xd_Logistic_w_2d_hole(ndim, n_train, scale, alpha, theta, threshold, weighted=False):
    m  = n_train

    x = np.zeros((m,ndim), dtype='float32')

    Rs_even = np.zeros((2,2), dtype='float32')
    Rs_even[0,0] = alpha * np.cos(theta)
    Rs_even[0,1] = -alpha * np.sin(theta)
    Rs_even[1,0] = np.sin(theta)
    Rs_even[1,1] = np.cos(theta)

    Rs_odd = np.zeros((2,2), dtype='float32')
    Rs_odd[0,0] = alpha * np.cos(np.pi-theta)
    Rs_odd[0,1] = -alpha * np.sin(np.pi-theta)
    Rs_odd[1,0] = np.sin(np.pi-theta)
    Rs_odd[1,1] = np.cos(np.pi-theta)

    n = 0.0
    y = np.zeros((ndim-1,), dtype='float32')
    for i in range(m):
        while True:
            z = np.random.logistic(0, scale, ndim)
            n += 1.0

            for j in range(ndim-1):
                if j % 2 == 0:
                    tp1 = Rs_even[0,0]*z[j] + Rs_even[0,1]*z[j+1]
                    tp2 = Rs_even[1,0]*z[j] + Rs_even[1,1]*z[j+1]
                else:
                    tp1 = Rs_odd[0,0]*z[j] + Rs_odd[0,1]*z[j+1]
                    tp2 = Rs_odd[1,0]*z[j] + Rs_odd[1,1]*z[j+1]

                y[j] = np.sqrt(tp1**2+tp2**2)
            if np.amin(y) >= threshold:
                x[i,:] = z

                if (i+1) % 10000 == 0:
                    print("step {}:".format(i+1))

                break

    return x, float(m)/n

def gen_4d_Gaussian_w_hole(ndim, n_train, alpha, theta, threshold, weighted=False, loadfile=False):
    assert ndim == 4
    m = n_train

    x = np.zeros((m,ndim), dtype='float32')

    if loadfile is True:
        x = np.loadtxt('training_set_KR.dat').astype(np.float32)
    else:
        Rs = np.zeros((2,2), dtype='float32')
        Rs[0,0] = alpha * np.cos(theta)
        Rs[0,1] = -alpha * np.sin(theta)
        Rs[1,0] = np.sin(theta)
        Rs[1,1] = np.cos(theta)

        for i in range(m):
            while True:
                z = np.random.normal(0,1,2)
                y = np.matmul(Rs,z)
                if np.linalg.norm(y,2) >= threshold:
                    x[i,:2] = z
                    break

        Rs[0,0] = alpha * np.cos(theta+np.pi/2.0)
        Rs[0,1] = -alpha * np.sin(theta+np.pi/2.0)
        Rs[1,0] = np.sin(theta+np.pi/2.0)
        Rs[1,1] = np.cos(theta+np.pi/2.0)

        for i in range(m):
            while True:
                z = np.random.normal(0,1,2)
                y = np.matmul(Rs,z)
                if np.linalg.norm(y,2) >= threshold:
                    x[i,2:] = z
                    break


        np.savetxt('training_set_KR.dat'.format(), x)

    if weighted is True:
        w = np.ones((m,1), dtype='float32')
        return x,w

    return x

def gen_nd_Gaussian_w_hole(ndim, n_train, threshold):
    m = n_train

    x = np.zeros((m,ndim), dtype='float32')
    for i in range(m):
        while True:
            z = np.random.normal(0,1,ndim)
            if np.linalg.norm(z,2) > threshold:
                x[i,:] = z
                break
    return x


def gen_square_domain(ndim, n_train, bd=1.0, unitcube=False, hyperuniform=False):
    """
    random samples in square domain  default to [-1, 1]^d (len_edge=2)
    """
    # subfunction: generate samples uniformly at random in a ball
    def gen_nd_ball(n_sample, n_dim):
        x_g = np.random.randn(n_sample, n_dim)
        u_number = np.random.rand(n_sample, 1)
        x_normalized = x_g / np.sqrt(np.sum(x_g**2, axis=1, keepdims=True))
        x_sample = (u_number**(1/n_dim) * x_normalized).astype(np.float32)
        return x_sample

    if not unitcube:
        # if hyperuniform, half data points drawn from a unit ball and half from uniform distribution
        if hyperuniform:
            n_corner = n_train//2
            # n_circle = (1 - np.exp(-ndim/10.0)) * n_train
            n_circle = n_train - n_corner
            # generate samples uniformly at random from a unit ball
            x_circle = gen_nd_ball(n_circle, ndim)
            # most of these samples are lies in the corner of a hypercube
            x_corner = np.random.uniform(-bd, bd, [n_corner, ndim]).astype(np.float32)
            x = np.concatenate((x_circle, x_corner), axis=0)

        else:
            x = np.random.uniform(-bd, bd, [n_train, ndim]).astype(np.float32)

    else:
        x = np.random.uniform(0, 1, [n_train, ndim]).astype(np.float32)
    return x


def gen_square_domain_boundary(ndim, n_train, bd=1.0, unitcube=False):
    """
    random samples on the boundary of square domain [-1,1]^2 (default)
    """
    if ndim != 2:
        raise ValueError('ndim must be 2 for boundary of square domain')

    # boundary of square domain are four edges: top (0), bottom (1), left (2), and right (3).
    edges = [0,1,2,3]
    x_boundary = np.zeros((n_train, ndim), dtype='float32')
    for i in range(n_train):
        which_edge = random.choice(edges)
        if which_edge == 0:
            point_x = np.random.uniform(-bd, bd)
            point_y = bd
        elif which_edge == 1:
            point_x = np.random.uniform(-bd, bd)
            point_y = -bd
        elif which_edge == 2:
            point_x = -bd
            point_y = np.random.uniform(-bd, bd)
        elif which_edge == 3:
            point_x = bd
            point_y = np.random.uniform(-bd, bd)

        point = np.array([point_x, point_y])
        x_boundary[i, :] = point

    if unitcube:
        x_boundary = 1/(2*bd) * x_boundary + 0.5
        return x_boundary

    return x_boundary


def gen_nd_cube_boundary(ndim, n_train, unitcube=False):
    """
    random samples on the boundary of a hypercube, default to  [-1,1]^d (len_edge=2)
    """
    if not unitcube:
        x = np.random.randn(n_train, ndim).astype(np.float32)
        x = (x.T / np.max(np.abs(x), axis=1)).T
    # [0,1]^d unit cube
    else:
        x = np.random.randn(n_train, ndim).astype(np.float32)
        x = (x.T / np.max(np.abs(x), axis=1)).T
        x = 0.5*x + 0.5
    return x


def gen_2d_mesh_square_domain(ndim, n_train, unitcube=False):
    """
    mesh for square domain
    """
    assert ndim == 2
    n_nodes = int(np.sqrt(n_train))
    if not unitcube:
        nodes = np.linspace(-1,1, n_nodes)
    else:
        nodes = np.linspace(0,1, n_nodes)
    X, Y = np.meshgrid(nodes, nodes)
    xxnodes = np.reshape(X, (-1,1), order='F')
    yynodes = np.reshape(Y, (-1,1), order='F')
    xs = np.concatenate((xxnodes, yynodes), axis=1).astype(np.float32)
    return xs

def gen_Lshape_domain(ndim, n_train):
    """
    random samples in L-shape domain ([-1,1]^2 remove the left-bottom corner square) 
    """
    if ndim != 2:
        raise ValueError('ndim must be 2 for L-shape domain')
    
    x = np.zeros((n_train, ndim), dtype='float32')
    # L-shape domain is devidied into three parts with equal area
    #    --------
    #    | 0 | 1 |
    #    ---------
    #        | 2 |
    #        -----
    parts = [0,1,2]
    for i in range(n_train):
        which_part = random.choice(parts)
        if which_part == 0:
            point_x = np.random.uniform(-1, 0)
            point_y = np.random.uniform(0, 1)
        elif which_part == 1:
            point_x = np.random.uniform(0, 1)
            point_y = np.random.uniform(0, 1)
        elif which_part == 2:
            point_x = np.random.uniform(0, 1)
            point_y = np.random.uniform(-1, 0)

        point = np.array([point_x, point_y])
        x[i, :] = point

    return x


def gen_Lshape_domain_boundary(ndim, n_train):
    """
    random samples on the boundary of L-shape domain 
    """
    if ndim != 2:
        raise ValueError('ndim must be 2 for boundary of L-shape domain')

    x_boundary = np.zeros((n_train, ndim), dtype='float32')
    #        0
    #    --------
    #  5 | 4 |   |
    #    --------- 1
    #       3|   |
    #        -----
    #          2
    edges = [0,1,2,3,4,5]
    for i in range(n_train):
        which_edge = np.random.choice(np.arange(6), p=[0.25,0.25,0.125,0.125,0.125,0.125])
        if which_edge == 0:
            point_x = np.random.uniform(-1, 1)
            point_y = 1.0
        elif which_edge == 1:
            point_x = 1.0
            point_y = np.random.uniform(-1, 1)
        elif which_edge == 2:
            point_x = np.random.uniform(0, 1)
            point_y = -1.0
        elif which_edge == 3:
            point_x = 0
            point_y = np.random.uniform(-1, 0)
        elif which_edge == 4:
            point_x = np.random.uniform(-1, 0)
            point_y = 0
        elif which_edge == 5:
            point_x = -1.0
            point_y = np.random.uniform(0, 1)

        point = np.array([point_x, point_y])
        x_boundary[i,:] = point

    return x_boundary


# another version, do not leave out the data points that out of boundary. Instead, using projection to save all data points
def projection_onto_infunitball(x):
    """
    projection onto the infinity ball ||x||_inf <=1 
    y = prox(x), then
    y = 1, if x >= 1; x, if |x| < 1; -1, if x <= -1.

    Args:
    -----
        x: data points 
        probsetup: problem type

    Returns:
    --------
        projection onto infnity ball
    """
    assert len(x.shape) == 2
    n_sample = x.shape[0]
    boundary_idx = []

    for k in range(n_sample):
        # data point on the boundary
        if (x[k,:] > 1).any() or (x[k,:] < -1).any():
            # projection onto boundary
            boundary_idx.append(k)
            x[k,:] = np.sign(x[k,:]) * np.minimum(np.abs(x[k,:]), 1)

    x_boundary = x[boundary_idx,:]
    x = np.delete(x, boundary_idx, axis=0)

    return x, x_boundary


# for random variables:
def flatten_sum(logps):
    assert len(logps.get_shape()) == 2
    return tf.reduce_sum(logps, [1], keepdims=True)


def gaussian_diag():
    class o(object): pass
    o.logps = lambda x: -0.5*(tf.math.log(2.*np.pi)+x**2)
    o.logp = lambda x: flatten_sum(o.logps(x))
    o.logps_g = lambda x, mean, logsd: -0.5*(tf.math.log(2.*np.pi)+2.*logsd+(x-mean)**2/tf.exp(2.*logsd))
    o.logp_g  = lambda x, mean, logsd: flatten_sum(o.logps_g(x, mean, logsd))
    return o


# Gaussian distribution
def log_standard_Gaussian(x):
    return flatten_sum(-0.5*(tf.math.log(2.*np.pi)+x**2))

# Uniform distribution [-1,1]^d
def log_uniform(x):
    n_dim = x.shape[1]
    return tf.math.log(0.5**n_dim)


# logistic distribution
def log_logistic(x):
    s = 1.0
    return flatten_sum(-x/s-tf.math.log(s)-2.0*tf.math.log(1.0+tf.exp(-x/s)))

