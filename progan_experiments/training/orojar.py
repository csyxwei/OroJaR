"""
Implementation of a multi-layer OroJaR in TensorFlow (Heavily borrowed from Hessian Penalty).
Steps of multi_layer_orojar() function:
    (0a) Construct graph of fake images G(z) (done in loss.py)
    (0b) Find the deepest activation R(z) we want to regularize (name of this activation is found in layers_to_reg[-1])
    (1) Grab R(z) out of G(z)'s graph using find_op() (we already computed R(z) as part of computing G(z)!)
    (2) Using graph_replace, compute R(z+x), where x are Rademacher vectors scaled by epsilon
    (3) Iterating over all intermediate layers I up-to and including R (as specified by layers_to_reg):
            (4) Grab I(z+x) and I(z) from R(z+x) and R(z)'s graphs, respectively (by using find_op())
            (5) Compute the Orthogonal Jacobian Regularization OroJaR(I) for layer I by using I(z+x) and I(z).
            (6) Only compute OroJaR(I) if I has been grown-in via progressive growth (done by wrapping OroJaR(I) in a tf.cond)
    (7) Return the sum of all layers' Orthogonal Jacobian Regularizations (i.e., sum of all OroJaR(I))
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import re
from dnnlib.tflib.autosummary import autosummary


def multi_layer_orojar(G_z, z, num_rademacher_samples, epsilon=0.1,
                       reduction=tf.reduce_max, layers_to_reg=None, current_lod=None,
                       max_lod=7, gpu_ix=None, verbose=False, log=True):
    """
    :param G_z: Tensor of shape (N, C, H, W): represents fake images G(z).
    :param z: Tensor of shape (N, nz): Latent z vector used to compute G_z.
    :param num_rademacher_samples: Number of Rademacher vectors x to sample to estimate OroJaR.
    :param epsilon: Finite differences hyperparameter for estimating Jacobian.
    :param reduction: Function used to reduce the OroJaR a scalar.
    :param layers_to_reg: The names of layers whose OroJaRs should be computed. We sum over all of them.
    :param current_lod: Placeholder indicating the current max_lod - log2(resolution) being output by ProgressiveGAN.
    :param max_lod: The maximum log2(resolution) of the dataset (e.g., for 128x128, max_lod=7)
    :param gpu_ix: The GPU index that this OroJaR is being computed on (needed for graph ops).
    :param verbose: Whether or not to print various debugging-related information to the console for graph construction.
    :param log: Whether or not to add individual per-layer OroJaRs to TensorBoard/ WandB
    :return: A differentiable scalar representing the sum of each layer's OroJaR.
    """

    def lod_fn(name):
        """
        Used for Step (6) as described above.
        Given a layer's name, this function returns the minimum level of detail that ProgressiveGAN
        must have grown to in order for that layer to start being regularized by the OroJaR.
        For example, if we are training on 128x128 real images, then we can only start regularizing
        layers in G that output 64x64 resolution features when lod < 2.0 (as that is when the
        64x64 layers we want to regularize are added as part of progressive growth).
        """

        if 'images_out' in name:  # final RGB conv layer is grown-in when lod < 1.0
            return 1.0
        res_x_res = re.match(r'\d+x\d+', name).group(0)
        x_ix = res_x_res.find('x')
        res = int(res_x_res[:x_ix])  # For example, res=64 when name="64x64/...":
        lod_required = max_lod - np.log2(res) + 1
        assert lod_required > 0, f'Unreachable LOD of {lod_required} computed for layer {name}'
        return lod_required

    def find_op(tensor, name, pos_constraints=None, neg_constraints=None):
        """
        Used for Steps (1) and (4) as described above.
        This is a helper function that tries to find intermediate activations in the graph
        belonging to "tensor" that contain the substring "name" in their name.
        When using multiple GPUs, sometimes there will be multiple tensors containing the same substring.
        We use pos_constraints and neg_constraints to filter-out tensors so we can find the correct intermediate
        activation.
        """
        def filter_fn(op):
            this_layer = name in op.name
            this_gpu = f'GPU{gpu_ix}' in op.name
            this_model = 'G_loss' in op.name
            if pos_constraints:
                sat_pos_constraints = all(pconstraint in op.name for pconstraint in pos_constraints)
            else:  # If we didn't specify any pos_constraints, just ignore them:
                sat_pos_constraints = True
            if neg_constraints:  #
                sat_neg_constraints = all(nconstraint not in op.name for nconstraint in neg_constraints)
            else:  # If we didn't specify any neg_constraints, just ignore them:
                sat_neg_constraints = True
            op_found = this_layer and this_gpu and this_model and sat_pos_constraints and sat_neg_constraints
            if op_found and verbose:
                print('-----')
                print(op.name)
                print(name)
                print('-----')
            return op_found

        assert tensor is not None
        ops = tensor.graph.get_operations()
        R_op = list(filter(filter_fn, ops))
        assert len(R_op) == 1, f'Found {len(R_op)} ops with name {name}'
        R = tensor.graph.get_tensor_by_name(f'{R_op[0].name}:0')
        return R

    def orojar(layer_name, R_z_perturbed, R_z, log=True):
        """
        Performs Steps (4) and (5) as described above.
        Computes the OroJaR w.r.t. input z vector of the G layer with name "layer_name."
        :param layer_name: String identifying the name of the activation we are taking the OroJaR of.
        :param R_z_perturbed: List of R(z+x) (num_rademacher_samples). The only difference between outs[0], outs[1], outs[2], ... is that x is
                              re-sampled.
        :param R_z: Tensor of shape (N, C, H, W): R(z).
        :param log: Whether or not to report layer-specific OroJaRs in TensorBoard/ WandB.
        :return: A differentiable scalar representing the OroJaR of this activation.
        """

        # Note: when I refer to (C, H, W) below, C=channels in activation, H=height of activation, W=width of activation
        jacobians = []
        I_z = find_op(R_z, layer_name, None, ['Ahead_'])  # (N, C, H, W)
        for k, r_ahead in enumerate(R_z_perturbed):
            I_z_plus_x = find_op(r_ahead, layer_name, [f'Ahead_k{k:05}'])
            # Estimating the Jacobian vector (Jv)
            jacobians.append((I_z_plus_x - I_z) / epsilon)  # Appends a (N, C, H, W) tensor
        assert len(jacobians) == num_rademacher_samples
        # calculate and stack the v^TJ^TJv
        jacobians = tf.stack([tf.matmul(tf.reshape(x, (tf.shape(x)[0], 1, -1)),
                                            tf.reshape(x, (tf.shape(x)[0], -1, 1))) for x in jacobians])  # (num_rademacher_samples, N, 1)
        variances = tf.nn.moments(jacobians, (0,))[1]  # (N, 1)
        full_orojar = reduction(variances)  # (1,) (scalar)
        if log:  # Optionally log individual layer OroJaR to TensorBoard/ WandB:
            full_orojar = autosummary(f'OroJaRs/{layer_name}', full_orojar)
        return full_orojar

    if verbose:
        print(f'Applying OroJaR to {", ".join(layers_to_reg)}')

    final_layer = layers_to_reg[-1]  # Step (0b)
    with tf.name_scope('OroJaR'):
        R_z = find_op(G_z, final_layer)  # Step (1)
        R_z_perturbed = []
        for k in range(num_rademacher_samples):  # Step (2) (we repeat for multiple samples of x):
            x = epsilon * tfp.math.random_rademacher(tf.shape(z))
            with tf.name_scope(f'Ahead_k{k:05}'):  # Compute the deepest intermediate activation R(z+x):
                R_z_perturbed.append(tf.contrib.graph_editor.graph_replace(R_z, {z: z + x}))
        assert len(R_z_perturbed) == num_rademacher_samples

        every_orojar = 0  # Stores the sum of all layers' OroJaR
        for layer_name in layers_to_reg:  # Step (3):
            lod_req = lod_fn(layer_name)
            if verbose:
                print(f'LOD required for layer {layer_name}: {lod_req}')
                print(f'Computing OroJaR for layer {layer_name}')
            # Steps (4), (5) and (6):
            # If this layer hasn't been grown-in yet, we simply don't compute its OroJaR (instead return 0.0).
            # The way progressive growth works, newly-grown layers are linearly phased-in. Similarly, we linearly
            # phase-in the OroJaR for that newly-grown layer (hence the tf.minimum() line below).
            every_orojar += tf.cond(current_lod < lod_req,
                                             true_fn=lambda: orojar(layer_name, R_z_perturbed, R_z, log=log) * tf.minimum(lod_req - current_lod, 1.0),
                                             false_fn=lambda: 0.0)
        return every_orojar  # Step (7)


def get_current_hessian_penalty_loss_weight(max_lambda, hp_start_iter, t, T):
    """
    Computes the current loss weighting of the Hessian Penalty.

    max_lambda: Maximum loss weighting for the Hessian Penalty
    hp_start_iter: the first training iteration where we start applying the Hessian Penalty
    t: current training iteration
    T: number of "warm-up" training iterations over which the Hessian Penalty should be linearly ramped-up
    """
    if t > hp_start_iter:  # If we're training/fine-tuning with the Hessian Penalty...
        if T > 0:  # If we want to linearly ramp-up the loss weighting of the Hessian Penalty:
            cur_hessian_weight = max_lambda * min(1.0, (t - hp_start_iter) / T)
        else:  # If we're training-from-scratch/ don't want to smoothly ramp-up the loss weighting:
            cur_hessian_weight = max_lambda
    else:  # If we aren't training with the Hessian Penalty (yet):
        cur_hessian_weight = 0.0
    return cur_hessian_weight
