"""
Official PyTorch implementation of the Orthogonal Jacobian Regularization term (Heavily based on Hessian Penalty)
Author: Yuxiang Wei
PyTorch Implementation (GPU + Multi-Layer): orojar_pytorch.py

Simple use case where you want to apply the OroJaR to the output of net w.r.t. net_input:
>>> from orojar_tf import orojar
>>> net = MyNeuralNet()
>>> net_input = sample_input()
>>> loss = orojar(net, z=net_input)  # Compute orojar of net's output w.r.t. net_input

If your network takes multiple inputs, simply supply them to orojar as you do in the net's forward pass. In the
following example, we assume BigGAN takes a second input argument "y". Note that we always take the OroJaR w.r.t. the z argument supplied to orojar:
>>> from orojar_tf import orojar
>>> net = BigGAN()
>>> z_input = sample_z_vector()
>>> class_label = sample_class_label()
>>> loss = orojar(net, z=net_input, y=class_label)
"""

import tensorflow as tf
import tensorflow_probability as tfp


def orojar(G, z, k=2, epsilon=0.1, reduction=tf.reduce_max, return_separately=False, G_z=None, **G_kwargs):
    """
    Official Tensorflow OroJaR implementation.

    Note: If you want to regularize multiple network activations simultaneously, you need to
    make sure the function G you pass to orojar returns a list of those activations when it's called with
    G(z, **G_kwargs). Otherwise, if G returns a tensor the OrojaR will only be computed for the final
    output of G.

    :param G: Function that maps input z to either a tensor or a list of tensors (activations)
    :param z: Input to G that the Regularization will be computed with respect to
    :param k: Number of Jacobian directions to sample (must be >= 2)
    :param epsilon: Amount to blur G before estimating Jacobian (must be > 0)
    :param reduction: TODO
    :param return_separately: If False, regularizations for each activation output by G are automatically summed into
                              a final loss. If True, the regularizations for each layer will be returned in a list
                              instead. If G outputs a single tensor, setting this to True will produce a length-1
                              list.
    :param G_z: [Optional small speed-up] If you have already computed G(z, **G_kwargs) for the current training
                iteration, then you can provide it here to reduce the number of forward passes of this method by 1
    :param G_kwargs: Additional inputs to G besides the z vector. For example, in BigGAN you
                     would pass the class label into this function via y=<class_label_tensor>

    :return: A differentiable scalar (the orojar), or a list of regularizations if return_separately is True
    """
    if G_z is None:
        G_z = G(z, **G_kwargs)
    first_orders = []
    for _ in range(k):  # Iterate over each (N, z.shape) tensor in xs
        x = epsilon * rademacher(tf.shape(z))
        first_order = multi_layer_first_directional_derivative(G, z, x, G_z, epsilon, **G_kwargs)
        first_orders.append(first_order)  # Appends a tensor with shape equal to G(z).shape
    loss = multi_stack_var_and_reduce(first_orders, reduction, return_separately)  # (k, G(z).shape) --> scalar
    return loss


def rademacher(shape):
    """Creates a random tensor of size [shape] under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    tfp_major_version, tfp_minor_version = tfp.__version__.split('.')[:2]  # Handle annoying TF deprecation...
    if int(tfp_major_version) == 0 and int(tfp_minor_version) < 11:
        return tfp.math.random_rademacher(shape)
    else:  # tfp.math.random_rademacher was deprecated in favor of the following:
        return tfp.random.rademacher(shape)


def multi_layer_first_directional_derivative(G, z, x, G_z, epsilon, **G_kwargs):
    """Estimates the first directional derivative of G w.r.t. its input at z in the direction x"""
    G_to_x = G(z + x, **G_kwargs)
    G_to_x = listify(G_to_x)
    G_z = listify(G_z)

    fdd = [(G2x - G_z_base) / epsilon for G2x, G_z_base in zip(G_to_x, G_z)]
    return fdd


def stack_var_and_reduce(list_of_activations, reduction=tf.reduce_max):
    dots = [tf.matmul(tf.reshape(x, (tf.shape(x)[0], 1, -1)),
                                    tf.reshape(x, (tf.shape(x)[0], -1, 1))) for x in list_of_activations]  # (k, N, 1)
    stacks = tf.stack(dots)  # # (k, N, 1)
    _, var_tensor = tf.nn.moments(stacks, axes=(0,))  # (N, 1)
    penalty = reduction(var_tensor)  # (1,) (scalar)
    return penalty


def multi_stack_var_and_reduce(fdds, reduction=tf.reduce_max, return_separately=False):
    """Iterate over all activations to be regularized, then apply Equation (3) to each."""
    sum_of_penalties = 0 if not return_separately else []
    for activ_n in zip(*fdds):
        penalty = stack_var_and_reduce(activ_n, reduction)
        sum_of_penalties += penalty if not return_separately else [penalty]
    return sum_of_penalties


def listify(x):
    """If x is already a list, do nothing. Otherwise, wrap x in a list."""
    if isinstance(x, list):
        return x
    else:
        return [x]

