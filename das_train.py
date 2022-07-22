from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import time
import os
import das_pde

import argparse




def parse_args():
    desc = "Deep adaptive sampling method for PDEs"
    p = argparse.ArgumentParser(description=desc)

    ## Data arguments
    p.add_argument('--data_dir', type=str, default='./store_data', help='Path to preprocessed data files.')

    ## save parameters
    p.add_argument('--ckpts_dir', type=str, default='./pdf_ckpt', help='Path to the check points.')
    p.add_argument('--summary_dir', type=str, default='./pdf_summary', help='Path to the summaries.')
    p.add_argument('--ckpt_step', type=int, default=5, help='Save the model every n iterations.')

    ## Neural network hyperparameteris
    p.add_argument('--n_depth', type=int, default=6, help='The number of affine coupling layers.')
    p.add_argument('--n_width', type=int, default=24, help='The number of neurons for the hidden layers.')
    p.add_argument('--n_split_at', type=int, default=1, help='The position of splitting two parts for affine coupling layers.')
    p.add_argument('--n_step', type=int, default=1, help='The step size for dimension reduction in each squeezing layer.')
    p.add_argument('--rotation', action='store_true', help='Specify rotation layers or not?')
    p.add_argument('--bounded_supp', action='store_true', help='Assume a bounded support for data.')
    p.add_argument('--n_bins4cdf', type=int, default=32, help='The number of bins for uniform partition of the support of PDF.')
    p.add_argument('--flow_coupling', type=int, default=1, help='Coupling type: 0=additive, 1=affine.')
    p.add_argument('--h1_reg', action='store_false', help='H1 regularization of the PDF.')
    p.add_argument('--shrink_rate', type=float, default=1.0, help='The shrinking rate of the width of NN.')
    p.add_argument("--bd", type=float, default=1, help="maximum boundary value")
    p.add_argument("--n_block", type=int, default=5, help="The number of ResBlock for NN-PDE ")
    p.add_argument("--netu_depth", type=int, default=6, help="The number of FC layers for NN-PDE ")
    p.add_argument("--n_hidden", type=int, default=32, help="The number of hidden neurons for NN-PDE ")
    p.add_argument("--activation", type=str, default='tanh', help="The activation function for NN-PDE. tanh, relu ，softplus and sin ")
    p.add_argument("--probsetup", type=int, default=3, help='The problem setup: 3, Peak; 6, Exponential; 7, Bimodal.')

    ## adaptive hyperparameters
    #p.add_argument("--scale_factor", type=float, default=0.3, help='The percentage of adding new samples to the training set.')
    p.add_argument("--replace_all", type=int, default=0, help='Replace training set or add new samples: 0, add new samples; 1, replace all.') 
    p.add_argument('--max_stage',type=int, default=5, help='Start refine samples after every stage.')
    p.add_argument('--tol',type=float, default=1e-7, help='Tolerance for training loss, if loss < tol, no refine; else, refine.')
    p.add_argument('--lambda_bd',type=float, default=1.0, help='The penalty parameter for the boundary condition.')
    p.add_argument('--quantity_type',type=str, default='residual', help='Quantity type for adaptive procedure. residual or slope')
    p.add_argument('--scale_quantity',type=float, default=1.0, help='Scaling for quantity')

    ## optimization hyperparams:
    p.add_argument("--n_dim", type=int, default=2, help='The number of random dimension.')
    p.add_argument("--n_train", type=int, default=1000, help='The number of samples in the initial training set.')
    p.add_argument('--batch_size', type=int, default=1000, help='Batch size of training generator.')
    #p.add_argument('--rotation_epochs', type=int, default=1000, help='The rotation will be switched off after 100 epochs.')
    p.add_argument("--lr", type=float, default=0.0001, help='Base learning rate.')
    p.add_argument('--n_epochs',type=int, default=3000, help='Total number of training epochs for PDE.')
    p.add_argument('--flow_epochs',type=int, default=3000, help='Total number of training epochs for flow.')
    p.add_argument('--flow_batch_size',type=int, default=1000, help='Batchsize for flow.')
    p.add_argument('--epochs_decay',type=float, default=1.0, help='Epochs decay rate of the model.')

    p.add_argument('--if_IS_residual',type=int, default=0, help='Use importance sampling for computing residual: 0, No; 1, Yes.')

    p.add_argument('--gpu_id',type=int, default=0, help='GPU id')

    return p.parse_args()



def main(hps):

    if hps is None:
        exit()

    model = das_pde.DAS(hps)
    # Train model
    t0 = time.time()
    model.train()
    t1 = time.time()

    TrainTime = (t1 - t0)/3600
    print('train time is {:.4} hours'.format(TrainTime))



if __name__ == '__main__':
    hps = parse_args()

    # choose GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the GPU id specified by gpu_id
        try:
            tf.config.experimental.set_visible_devices(gpus[hps.gpu_id], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    main(hps)




