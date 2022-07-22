from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

import BR_lib.BR_model as BR_model
import BR_lib.BR_data as BR_data
import nn_model 
import pde_model

import os
import shutil



def gen_train_data(n_dim, n_sample, probsetup):
    """
    generate training data for the first stage
    Args:
    -----
        n_dim: dimension
        n_sample: number of samples
        probsetup: type of problems, see das_train.py file 

    Returns:
    --------
        x, x_boundary
        data points for training, including interior data points and boundary data points
    """
    if probsetup == 3:
        x = BR_data.gen_square_domain(n_dim, n_sample)
        x_boundary = BR_data.gen_square_domain_boundary(n_dim, n_sample)

    elif probsetup == 6:
        x = BR_data.gen_square_domain(n_dim, n_sample)
        x_boundary = BR_data.gen_nd_cube_boundary(n_dim, n_sample)

    elif probsetup == 7:
        x = BR_data.gen_square_domain(n_dim, n_sample)
        x_boundary = BR_data.gen_square_domain_boundary(n_dim, n_sample)

    else:
        raise ValueError('probsetup is not valid')

    return x, x_boundary


def load_valid_data(n_dim, probsetup):
    """
    load validation data for performance evaluation
    Args:
    -----
        n_dim: data dimension
        probsetup: type of problems

    Returns:
    --------
        true function values at the validation set, numpy format
    """
    if probsetup == 3:
        valid_dir = os.path.join('./dataset_for_validation', '{}d_square_problem.dat'.format(n_dim))
        sample_valid = np.loadtxt(valid_dir).astype(np.float32)
        u_true = pde_model.diffusion_peak(sample_valid)

    elif probsetup == 6:
        valid_dir = os.path.join('./dataset_for_validation', '{}d_exp_problem.dat'.format(n_dim))
        sample_valid = np.loadtxt(valid_dir).astype(np.float32)
        u_true = pde_model.diffusion_exp(sample_valid)

    elif probsetup == 7:
        valid_dir = os.path.join('./dataset_for_validation', '{}d_square_problem.dat'.format(n_dim))
        sample_valid = np.loadtxt(valid_dir).astype(np.float32)
        u_true = pde_model.bimodal_exact(sample_valid)

    else:
        raise ValueError('probsetp is not valid')

    return sample_valid, u_true



class DAS():
    """
    Deep adaptive sampling (DAS) for partial differential  equations
    ------------------------------------------------------------------------------------
    Here is the deep adaptive sampling method to solve partial differential equations. 
    Solving PDEs using deep nueral networks needs to compute a loss function with sample generation. 
    In general, uniform samples are generated, but it is not an optimal choice to efficiently train models. 
    However, flow-based generative models provide an opportunity for efficient sampling, and this is exactly what DAS 
    does. 
    
    Args:
    -----
        args: input parameters
    """
    def __init__(self, args):
        self.args = args
        self._set()
        self.build_nn()
        self.build_flow()
        self._restore()
 

    def _set(self):
        args = self.args
        pde_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        flow_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        self.pde_optimizer = pde_optimizer
        self.flow_optimizer = flow_optimizer

        # this is for creating folder for save training and validation loss history, checkpoint and summary
        if os.path.exists(args.ckpts_dir):
            shutil.rmtree(args.ckpts_dir)
        os.mkdir(args.ckpts_dir)

        if os.path.exists(args.summary_dir):
            shutil.rmtree(args.summary_dir)
        os.mkdir(args.summary_dir)

        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
        os.mkdir(args.data_dir)

        self.pdeloss_vs_iter = []
        self.residualloss_vs_iter = []
        self.entropyloss_vs_iter = []
        self.approximate_error_vs_iter = []
        self.resvar_vs_iter = []


    def _restore(self):
        args = self.args
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.pde_optimizer, net=self.net_u)
        self.manager = tf.train.CheckpointManager(self.ckpt, args.ckpts_dir, max_to_keep=5)


    def build_flow(self):
        args = self.args
        # build the PDF model

        # enlarge the computation domain slightly
        xlb = -args.bd - 0.01
        xhb = args.bd + 0.01

        pdf_model = BR_model.IM_rNVP_KR_CDF('pdf_model_rNVP_KR_CDF',
                                            args.n_dim,
                                            xlb, xhb,
                                            args.n_step,
                                            args.n_depth,
                                            n_width=args.n_width,
                                            shrink_rate=args.shrink_rate,
                                            flow_coupling=args.flow_coupling,
                                            n_bins=args.n_bins4cdf,
                                            rotation=args.rotation,
                                            bounded_supp=args.bounded_supp)

        self.pdf_model = pdf_model


    def build_nn(self):
        args = self.args
        # create a neural network to approximate the solution of PDEs
        net_u = nn_model.FCNN('FCNN', 1, args.netu_depth, args.n_hidden, args.activation)

        self.net_u = net_u 


    def resample(self):
        """
        resample from the trained flow model corresponding to mesh refinement in FEM
        This function will be excuted when residual loss of PDE is greater than a tolerance

        There are two strategies for resample. 
        The first one is to replace the current data points by samples generated from the flow model.
        The second one is that we add new samples generated from the flow model to the current set.

        Both of them need to sample n_train data points, so there is no difference for resample function, 
        but one should take this into account in the train function
        """
        args = self.args
        n_resample = args.n_train
        projection_operator = BR_data.projection_onto_infunitball

        x_prior = self.pdf_model.draw_samples_from_prior(n_resample, args.n_dim)
        x_candidate = self.pdf_model.mapping_from_prior(x_prior).numpy()

        # Since samples from the flow model may be out of boundary, one should do a projection
        # resample contains boundary data
        x_resample, x_bd = projection_operator(x_candidate)
        nv_sample = x_resample.shape[0]
        while nv_sample < n_resample:
            n_diff = n_resample - nv_sample
            x_prior_new = self.pdf_model.draw_samples_from_prior(n_diff, args.n_dim)
            x_candidate_new = self.pdf_model.mapping_from_prior(x_prior_new).numpy()
            x_candidate_new, _ = projection_operator(x_candidate_new)
            x_resample = np.concatenate((x_resample, x_candidate_new), axis=0)
            #x_bd = np.concatenate((x_bd, x_bd_new), axis=0)
            nv_sample = x_resample.shape[0]

        if x_bd.shape[0] < x_resample.shape[0]:
            n_add = x_resample.shape[0] - x_bd.shape[0]
            _, x_bd_add = gen_train_data(args.n_dim, n_add, args.probsetup)
            x_bd = np.concatenate((x_bd, x_bd_add), axis=0)
        else:
            n_s = x_resample.shape[0]
            x_bd = x_bd[:n_s,:]

        #return x_add
        x_new = np.concatenate((x_resample, x_bd), axis=1)
        return x_new


    def get_pde_loss(self, x, x_boundary, stage_idx):
        args = self.args

        if args.probsetup == 3:
            residual = pde_model.residual_peak(self.net_u, x)
            residual_boundary = pde_model.boundary_loss_peak(self.net_u, x_boundary)

        elif args.probsetup == 6:
            residual = pde_model.residual_exp(self.net_u, x)
            residual_boundary = pde_model.boundary_loss_exp(self.net_u, x_boundary)

        elif args.probsetup == 7:
            residual = pde_model.residual_bimodal(self.net_u, x)
            residual_boundary = pde_model.boundary_loss_bimodal(self.net_u, x_boundary)

        else:
            raise ValueError('probsetup is not valid')

        # When replace_all = 0, DAS-G; DAS-R, else
        # importance sampling may be used if replace all samples
        if stage_idx == 1:
            pde_loss = tf.reduce_mean(residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)

        else:

            if args.replace_all == 1:
                # importance sampling for computing residual
                # scaling to avoid numerical underflow issues
                if args.if_IS_residual == 0:
                    pde_loss = tf.reduce_mean(residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)
                else:
                    scaling = 1000.0
                    log_pdf = tf.clip_by_value(self.pdf_model(x), -23.02585, 5.0)
                    pdfx = tf.math.exp(log_pdf)
                    weight_residual = tf.math.divide(scaling*residual, scaling*pdfx)
                    pde_loss = tf.reduce_mean(weight_residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)

            else:
                if args.if_IS_residual == 0:
                    pde_loss = tf.reduce_mean(residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)

                else:
                    # importance sampling for computing residual
                    # scaling to avoid numerical underflow issues
                    scaling = 1000.0
                    log_pdf = tf.clip_by_value(self.pdf_model(x), -23.02585, 5.0)
                    pdfx = tf.math.exp(log_pdf)
                    weight_residual = tf.math.divide(scaling*residual, scaling*pdfx)
                    pde_loss = tf.reduce_mean(weight_residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)

        return pde_loss, residual


    def get_pdf(self, x):
        log_pdfx = self.pdf_model(x)
        pdfx = tf.math.exp(log_pdfx)
        return pdfx


    def get_entropy_loss(self, quantity, pre_pdf, x):
        log_pdf = tf.clip_by_value(self.pdf_model(x), -23.02585, 5.0)

        # scaling for numerical stability
        scaling = 1000.0
        pre_pdf = scaling*pre_pdf
        quantity = scaling*quantity

        # importance sampling
        ratio = tf.math.divide(quantity, pre_pdf)
        res_time_logpdf = ratio*log_pdf
        entropy_loss = -tf.reduce_mean(res_time_logpdf)
        return entropy_loss


    @tf.function
    def get_slope(self, x):
        """
        compute slope for nn
        """
        with tf.GradientTape() as tape:
            unn = self.net_u(x)
            grads = pde_model.compute_grads(unn, x)
            slopes = tf.reduce_sum(tf.math.square(grads), axis=1, keepdims=True) 

        return slopes


    @tf.function
    def residual_for_flow(self, inputs, inputs_boundary, stage_idx):
        with tf.GradientTape() as tape:
            pde_loss, residual = self.get_pde_loss(inputs, inputs_boundary, stage_idx)

        return residual


    @tf.function
    def train_pde(self, inputs, inputs_boundary, i, net_u_training_vars):
        # two neural networks: one for approximating PDE, and another for adaptive sampling
        with tf.GradientTape() as pde_tape:
            pde_loss, residual = self.get_pde_loss(inputs, inputs_boundary, i)

        grads_net_u = pde_tape.gradient(pde_loss, net_u_training_vars)
        self.pde_optimizer.apply_gradients(zip(grads_net_u, net_u_training_vars))
        return pde_loss, residual


    @tf.function
    def train_flow(self, inputs, quantity, pre_pdf, pdf_training_vars):
        # two neural networks: one for approximating PDE, and another for adaptive sampling
        with tf.GradientTape() as ce_tape:
            entropy_loss = self.get_entropy_loss(quantity, pre_pdf, inputs)

        grads_pdf_model = ce_tape.gradient(entropy_loss, pdf_training_vars)
        self.flow_optimizer.apply_gradients(zip(grads_pdf_model, pdf_training_vars))
        return entropy_loss


    def solve_pde(self, train_dataset, stage_idx, sample_valid, u_true):
        """train a neural network to approximate the pde solution"""
        args = self.args
        n_epochs = args.n_epochs
        for k in tf.range(1, n_epochs+1):
            for step, train_batch in enumerate(train_dataset):
                batch_x = train_batch[:,:args.n_dim]
                batch_boundary = train_batch[:,args.n_dim:]
                pde_loss, residual = self.train_pde(batch_x, batch_boundary, stage_idx, self.net_u.trainable_weights)

                residual_loss = tf.reduce_mean(residual)
                variance_residual = tf.math.reduce_variance(residual)
                print('stage: %s, epoch: %s, iter: %s, residual_loss: %s, pde_loss: %s ' % 
                      (stage_idx, k.numpy(), step+1, residual_loss.numpy(), pde_loss.numpy()))

                self.pdeloss_vs_iter += [pde_loss.numpy()]
                self.residualloss_vs_iter += [residual_loss.numpy()]
                self.resvar_vs_iter += [variance_residual.numpy()]

                ####################################################
                # evalute model performance using load test data every iteration
                if args.probsetup == 0 or args.probsetup == 6:
                    ## Error on data points
                    u_pred = self.net_u(sample_valid)
                    approximate_error = tf.norm(u_true - u_pred, ord=2)/tf.norm(u_true, ord=2)

                else:
                    u_pred = self.net_u(sample_valid)
                    approximate_error = tf.reduce_mean(tf.math.square(u_true - u_pred))

                self.approximate_error_vs_iter += [approximate_error.numpy()]
                #####################################################
            # Save model
            self.ckpt.step.assign_add(1)
            if int(self.ckpt.step) % args.ckpt_step == 0:
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

        # record the final five steps for the stopping criterion
        tol_pde = np.mean(np.array(self.pdeloss_vs_iter[-5:]))
        res_var = np.mean(np.array(self.resvar_vs_iter[-5:]))

        return u_pred, tol_pde, res_var


    def solve_flow(self, train_dataset, i):
        args = self.args
        flow_epochs = args.flow_epochs
        for k in tf.range(1, flow_epochs+1):
            for step, batch_x in enumerate(train_dataset):
                if i == 1:
                    batch_x_data = batch_x
                else:
                    # extract data points and its pdf value for i > 1
                    batch_x_data = batch_x[:, :args.n_dim]
                    batch_pre_pdf = tf.reshape(batch_x[:, -1], [-1,1])
                # generate boundary data only for quantity computation
                _, batch_boundary = gen_train_data(args.n_dim, args.flow_batch_size, args.probsetup)
                # using slopes or residual for adaptivity
                if args.quantity_type == 'slope':
                    quantity = self.get_slope(batch_x_data)
                    quantity = args.scale_quantity * quantity
                else:
                    quantity = self.residual_for_flow(batch_x_data, batch_boundary, i)
                    quantity = args.scale_quantity * quantity
                if i == 1:
                    pre_pdf = tf.ones_like(quantity, dtype=tf.float32)
                    entropy_loss = self.train_flow(batch_x_data, quantity, pre_pdf, self.pdf_model.trainable_weights)
                    quantity_loss = tf.reduce_mean(quantity)
                    print('stage: %s, flow_epoch: %s, iter: %s, quantity: %s, entropy_loss: %s ' % 
                          (i, k.numpy(), step+1, quantity_loss.numpy(), entropy_loss.numpy()))
                else:
                    # pdf values from the previous model are precomputed
                    entropy_loss = self.train_flow(batch_x_data, quantity, batch_pre_pdf, self.pdf_model.trainable_weights)
                    quantity_loss = tf.reduce_mean(quantity)
                    print('stage: %s, flow_epoch: %s, iter: %s, quantity: %s, entropy_loss: %s ' % 
                          (i, k.numpy(), step+1, quantity_loss.numpy(), entropy_loss.numpy()))

                self.entropyloss_vs_iter += [entropy_loss.numpy()]


    def train(self):
        """training procedure"""
        args = self.args
        max_stage = args.max_stage

        #################################################
        #load test data for evaluating model
        sample_valid, u_true = load_valid_data(args.n_dim, args.probsetup)

        # summary
        summary_writer = tf.summary.create_file_writer(args.summary_dir)

        print(' Quantity type for adaptive procedure: %s' % (args.quantity_type))
        print('====== Training process starting... ======')

        with summary_writer.as_default():

            # set random seed
            np.random.seed(23)
            tf.random.set_seed(23)
            # In the first step, data points are generated uniformly since there is no prior information 

            # starting from uniform distribution
            x_data, x_boundary = gen_train_data(args.n_dim, args.n_train, args.probsetup)
            x = np.concatenate((x_data, x_boundary), axis=1)

            data_flow_pde = BR_data.dataflow(x, buffersize=args.n_train, batchsize=args.batch_size)
            train_dataset_pde = data_flow_pde.get_shuffled_batched_dataset()

            data_flow_kr = BR_data.dataflow(x_data, buffersize=args.n_train, batchsize=args.flow_batch_size)
            train_dataset_flow = data_flow_kr.get_shuffled_batched_dataset()

            m = 1
            x_init_kr = data_flow_kr.get_n_batch_from_shuffled_batched_dataset(m)
            # pass data to networks to complete building process
            self.net_u(x_init_kr)
            self.pdf_model(x_init_kr)
            if args.rotation:
                self.pdf_model.WLU_data_initialization()
            self.pdf_model.actnorm_data_initialization()

            for i in range(1, max_stage+1):

                u_pred, tol_pde, res_var = self.solve_pde(train_dataset_pde, i, sample_valid, u_true)
                if tol_pde < args.tol and res_var < args.tol and i > 1:
                    print('===== stoppping criterion satisfies, finish training =====')
                    break

                if i < max_stage:

                    self.solve_flow(train_dataset_flow, i)
                    # resample contains two parts: data points in domain and boundary data
                    x_new = self.resample()

                    # two strategies: replace all samples or add new samples
                    if args.replace_all == 1:
                        # replace all samples
                        buffersize = x_new.shape[0]
                        data_flow_pde = BR_data.dataflow(x_new, buffersize=buffersize, batchsize=args.batch_size)
                        train_dataset_pde = data_flow_pde.get_shuffled_batched_dataset()

                        x_prior = self.pdf_model.draw_samples_from_prior(buffersize, args.n_dim)
                        x_flow = self.pdf_model.mapping_from_prior(x_prior).numpy()

                        pre_pdf = tf.clip_by_value(self.get_pdf(x_flow), 1.0e-10, 148.4131)
                        pre_pdf = tf.stop_gradient(pre_pdf)
                        x_flow = np.concatenate((x_flow, pre_pdf), axis=1)
                        data_flow_kr = BR_data.dataflow(x_flow, buffersize=buffersize, batchsize=args.flow_batch_size)
                        train_dataset_flow = data_flow_kr.get_shuffled_batched_dataset()

                    else:
                        # add new samples to the current set
                        x = np.concatenate((x, x_new), axis=0)
                        buffersize = x.shape[0]

                        # refine data points for the next stage
                        data_flow_pde = BR_data.dataflow(x, buffersize=buffersize, batchsize=args.batch_size)
                        train_dataset_pde = data_flow_pde.get_shuffled_batched_dataset()

                        x_prior = self.pdf_model.draw_samples_from_prior(x.shape[0], args.n_dim)
                        x_flow = self.pdf_model.mapping_from_prior(x_prior).numpy()

                        pre_pdf = tf.clip_by_value(self.get_pdf(x_flow), 1.0e-10, 148.4131)
                        pre_pdf = tf.stop_gradient(pre_pdf)
                        x_flow = np.concatenate((x_flow, pre_pdf), axis=1)
                        data_flow_kr = BR_data.dataflow(x_flow, buffersize=args.n_train, batchsize=args.flow_batch_size)
                        train_dataset_flow = data_flow_kr.get_shuffled_batched_dataset()
                    # save resample data points every stage
                    x_resample_stg = x_new[:, :args.n_dim]
                    np.savetxt(os.path.join(args.data_dir, 'stage_{}_resample.dat'.format(i)), x_resample_stg)


            # save data for visualization after training
            np.savetxt(os.path.join(args.data_dir, 'pdeloss_vs_iter.dat'), np.array(self.pdeloss_vs_iter))

            np.savetxt(os.path.join(args.data_dir, 'residualloss_vs_iter.dat'), np.array(self.residualloss_vs_iter))
            validation_error = np.array(self.approximate_error_vs_iter).reshape(-1,1)
            np.savetxt(os.path.join(args.data_dir, 'validation_error.dat'), validation_error)

            # save u_true and u_pred on the validation set
            np.savetxt(os.path.join(args.data_dir, 'u_true.dat'), u_true)
            np.savetxt(os.path.join(args.data_dir, 'u_pred.dat'), u_pred)
            if args.max_stage > 1:
                np.savetxt(os.path.join(args.data_dir, 'entropyloss_vs_iter.dat'), np.array(self.entropyloss_vs_iter))


