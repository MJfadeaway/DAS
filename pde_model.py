from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np



def a_diff_v1(x):
    """
    TF format
    Coefficient function 

    Args:
    -----
         x: collocation points, a tensor

    Returns:
    --------
         coefficients: function values at collocation points,
         a tensor with size batch * 1
    """
    a = tf.ones([x.shape[0], 1], dtype=tf.float32)

    return a 


def compute_grads(f_out, x_inputs):
    """
    TF format
    Compute dy/dx where y = f(x), if y is a tensor, return d(sum(y))/dx

    Args:
    -----
        f_out: y, a tensor
        x_inputs: x, a tensor

    Returns:
    --------
        grads: dy/dx, a tensor with shape of x_inputs
    """
    grads = tf.gradients(f_out, [x_inputs])[0]

    return grads


def compute_div(f_out, x_inputs):
    """
    TF format
    Compute div(y) where y = grad(f(x))

    Args:
    -----
        f_out: y, a second order tensor where each row is a sample
        x_inputs: x, a second order tensor where each row is a sample

    Returns:
    --------
        div(y): a tensor where each row (scalar) is corresponding divergence
    """
    div_qx = tf.stack([tf.gradients(tmp, [x_inputs])[0][:, idx] for idx, tmp in enumerate(tf.unstack(f_out, axis=1))], axis=1)
    div_qx = tf.reduce_sum(div_qx, axis=1, keepdims=True)
    return div_qx


def f_source_peak(x):
    """
    TF format
    source function (right-hand term) for 2D peak problem

    Args:
    -----
        x: coordinates, a second order tensor where each row is a sample

    Returns:
    --------
        source function of 2D peak problem, a first order tensor
    """
    f_1 = -4000 * tf.math.exp(-1000*(x[:,0]**2-x[:,0]+x[:,1]**2-x[:,1]+0.5))
    f_2 = 1000 * x[:,0]**2 - 1000 * x[:,0] + 1000 * x[:,1]**2 - 1000 * x[:,1] + 499.0
    f_source = f_1 * f_2
    f_source = tf.reshape(f_source, [-1,1])
    return f_source


def f_source_exp(x):
    """
    TF format, source function (right-hand term) for exponential minus square norm

    Args:
    -----
        x: coordinates, a second order tensor where each row is a sample

    Returns:
    --------
        source function values at x of d dimensional problem, a first order tensor
    """
    n_dim = x.shape[-1]
    x_sum_square = tf.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
    f_source = tf.math.exp(-10.0*x_sum_square) * (20.0*n_dim - 400.0*x_sum_square)
    return f_source


def residual_peak(model, x):
    """
    TF format
    Compute the integral of square of residual for 2D peak problem

    Args:
    -----
        model: a neural network constructed with a defined model
        x: collocation points, a tensor with size batch_size * n_dim
        #a_diff: diffusion coefficient (function values at x), return 
        #a tensor with size batch_size * 1
        #f_source: source function or right-hand term 

    Returns:
    --------
        residual: residual  of peak problem, a tensor

    """
    fx = model(x)
    # constant coefficient function
    a_coeff = a_diff_v1(x)
    grads = compute_grads(fx, x)
    qx = a_coeff * grads
    # compute divergence 
    div_qx = compute_div(qx, x)

    f_source = f_source_peak(x)
    residual = tf.math.square(-div_qx - f_source)

    return residual


def boundary_loss_peak(model, x_boundary):
    """
    TF format
    Compute the boundary loss for 2D peak problem

    Args:
    -----
        model: a neural network constructed with a defined model
        x_boundary: collocation points on boundary, a tensor with size batch_size * n_dim

    Returns:
    --------
        residual on the boundary , a tensor
    """
    fx_boundary = model(x_boundary)
    u_boundary = diffusion_peak_boundary(x_boundary)
    residual_boundary = tf.math.square(fx_boundary-u_boundary)
    return residual_boundary


def residual_exp(model, x):
    """
    TF format
    Compute the integral of square of residual for exp minus square norm

    Args:
    -----
        model: a neural network constructed with a defined model
        x: collocation points, a tensor with size batch_size * n_dim
        #a_diff: diffusion coefficient (function values at x), return 
        #a tensor with size batch_size * 1
        #f_source: source function or right-hand term 

    Returns:
    --------
        residual: residual  of sinebump problem, a tensor
    """
    fx = model(x)
    # constant coefficient function
    a_coeff = a_diff_v1(x)
    grads = compute_grads(fx, x)
    qx = a_coeff * grads
    # compute divergence 
    div_qx = compute_div(qx, x)

    f_source = f_source_exp(x)
    residual = tf.math.square(-div_qx - f_source)

    return residual


def boundary_loss_exp(model, x_boundary):
    """
    TF format
    Compute the boundary loss for exponential minus square norm problem

    Args:
    -----
        model: a neural network constructed with a defined model
        x_boundary: collocation points on boundary, a tensor with size batch_size * n_dim

    Returns:
    --------
        residual on the boundary , a tensor
    """
    fx_boundary = model(x_boundary)
    u_boundary = diffusion_exp_boundary(x_boundary)
    residual_boundary = tf.math.square(fx_boundary-u_boundary)
    return residual_boundary


# 2D peak 
def diffusion_peak(x):
    """
    Numpy format
    the exact solution for diffusion peak problem is

    u(x,y) = exp(-1000 * (x^2 - x + y^2 - y + 0.5))

    Args:
    -----
        x: coordinates in the domain, a second order tensor where each row is a sample

    Returns:
    --------
        u(x,y): function values at coordinates x = (x,y)
    """
    ux = np.exp(-1000*(x[:,0]**2 - x[:,0] + x[:,1]**2 - x[:,1] + 0.5))
    ux = np.reshape(ux, (-1,1))
    return ux


def diffusion_peak_boundary(x_boundary):
    """
    TF format
    boundary condition: the exact solution on boundary 

    Args:
    -----
        x_boundary: boundary points, a second order tensor where each row is a sample

    Returns:
    --------
        u(x,y): function values on boundary points
    """
    x = x_boundary[:,0]
    y = x_boundary[:,1]
    u_boundary = tf.math.exp(-1000*(x**2 - x + y**2 - y + 0.5))
    u_boundary = tf.reshape(u_boundary, [-1,1])
    return u_boundary


# exponential minus square norm function
def diffusion_exp(x):
    """
    Numpy format
    the exact solution for diffusion exponential minus square norm problem is

    u(x,y) = exp(-10.0*(x1**2 + ... + xd**2))

    Args:
    -----
        x: coordinates in the domain, a second order tensor where each row is a sample

    Returns:
    --------
        u(x,y): function values at coordinates x = (x,y)
    """
    x_sum_square = np.sum(x**2, axis=1, keepdims=True)
    ux = np.exp(-10.0*x_sum_square)
    return ux


def diffusion_exp_boundary(x_boundary):
    """
    TF format
    boundary condition: the exact solution on boundary 

    Args:
    -----
        x_boundary: boundary points, a second order tensor where each row is a sample

    Returns:
    --------
        u(x,y): function values on boundary points
    """
    x_sum_square = tf.reduce_sum(tf.math.square(x_boundary), axis=1, keepdims=True)
    u_boundary = tf.math.exp(-10.0*x_sum_square)
    return u_boundary


def f_source_bimodal(x):
    """
    TF format
    Source function: bimodal function
    """
    assert len(x.shape) == 2

    x1 = x[:,0]
    x2 = x[:,1]

    ## div(grad(u))
    f1 = 4000.0 * tf.math.exp(-1000*((x1-0.5)**2 + (x2-0.5)**2)) * (1000*x1**2 - 1000*x1 + 1000*x2**2 - 1000*x2 + 499)
    f2 = 4000.0 * tf.math.exp(-500*(2*x1**2 + 2*x1 + 2*x2**2 + 2*x2 + 1)) * (1000*x1**2 + 1000*x1 + 1000*x2**2 + 1000*x2 + 499)

    ## -div(u grad(b(x,y))) where b(x,y) = x^2 + y^2
    f3 = 4.0 * tf.math.exp(-1000*((x1-0.5)**2 + (x2-0.5)**2)) * (1000*x1**2 - 500*x1 + 1000*x2**2 - 500*x2 - 1)
    f4 = 4.0 * tf.math.exp(-500*(2*x1**2 + 2*x1 + 2*x2**2 + 2*x2 + 1)) * (1000*x1**2 + 500*x1 + 1000*x2**2 + 500*x2 - 1)
    f_source = f1 + f2 + f3 + f4
    f_source = tf.reshape(f_source, [-1,1])

    return f_source


def residual_bimodal(model, x):
    """
    TF format
    Compute the integral of square of residual for bimodal function

    Args:
    -----
        model: a neural network constructed with a defined model
        x: collocation points, a tensor with size batch_size * n_dim
        #a_diff: diffusion coefficient (function values at x), return 
        #a tensor with size batch_size * 1
        #f_source: source function or right-hand term 

    Returns:
    --------
        residual: residual of bimodal function, a tensor

    """
    assert len(x.shape) == 2

    fx = model(x)

    ## compute div(grad(u))
    # constant coefficient function
    a_coeff = a_diff_v1(x)
    grads = compute_grads(fx, x)
    qx = a_coeff * grads
    # compute divergence 
    div_qx = compute_div(qx, x)

    ## compute -div(u grad(b(x,y)))
    mu1 = tf.reshape(2*x[:,0], [-1,1])
    mu2 = tf.reshape(2*x[:,1], [-1,1])
    mu = tf.concat([mu1, mu2], axis=1)
    mutf = mu*fx
    lfmu = tf.stack([tf.gradients(tmp, [x])[0][:, idx] for idx, tmp in enumerate(tf.unstack(mutf, axis=1))], axis=1)
    lfmu = tf.reduce_sum(lfmu, axis=1, keepdims=True)

    f_source = f_source_bimodal(x)
    residual = tf.math.square(-lfmu + div_qx - f_source)

    return residual


def boundary_loss_bimodal(model, x_boundary):
    """
    TF format
    Compute the boundary loss for bimodal problem

    Args:
    -----
        model: a neural network constructed with a defined model
        x_boundary: collocation points on boundary, a tensor with size batch_size * n_dim

    Returns:
    --------
        residual on the boundary , a tensor
    """
    fx_boundary = model(x_boundary)
    u_boundary = bimodal_boundary(x_boundary)
    residual_boundary = tf.math.square(fx_boundary-u_boundary)
    return residual_boundary


def bimodal_boundary(x_boundary):
    """
    TF format
    the exact solution on the boundary for bimodal problem

    Args:
    -----
        x_boundary: collocation points on boundary, a second order tensor where 
                    each row is a sample

    Returns:
    --------
        u(x,y): function values at coordinates x = (x, y)

    """
    assert len(x_boundary.shape) == 2

    x = x_boundary[:,0]
    y = x_boundary[:,1]
    ux = tf.math.exp(-1000*((x-0.5)**2+(y-0.5)**2)) + tf.math.exp(-1000*((x+0.5)**2 + (y+0.5)**2))
    ux = tf.reshape(ux, [-1,1])
    return ux


def bimodal_exact(x):
    """
    Numpy format
    the exact solution for bimodal problem

    Args:
    -----
        x: coordinates in the domain, a second order tensor where each row is a sample

    Returns:
    --------
        u(x,y): function values at coordinates x = (x,y)
    """
    x1 = x[:,0]
    x2 = x[:,1]

    ux = np.exp(-1000*((x1-0.5)**2+(x2-0.5)**2)) + np.exp(-1000*((x1+0.5)**2 + (x2+0.5)**2))
    ux = np.reshape(ux, (-1,1))
    return ux


