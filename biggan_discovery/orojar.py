import torch


def rademacher(shape, gpu=True):
    """
    Creates a random tensor of size [shape] under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)
    """
    x = torch.empty(shape)
    if gpu:
        x = x.cuda()
    x.random_(0, 2)
    x[x == 0] = -1
    return x


def first_directional_derivative(G, z, c, x, G_z, epsilon, w=None, Q=None):
    """
    Computes the first directional derivative of G w.r.t. its input at z in the direction x or w.
    """
    if w is None:  # Apply the OroJaR in Z-space
        return (G(z + x, c, Q=Q) - G_z) / epsilon

    else:  # Apply it in W-space
        return (G(z, c, w=w+x, Q=Q) - G_z) / epsilon


def listify(x):
    """If x is already a list, do nothing. Otherwise, wrap x in a list."""
    if isinstance(x, list):
        return x
    else:
        return [x]


def multi_layer_first_directional_derivative(G, z, c, x, G_z, epsilon, w=None, Q=None):
    """Estimates the first directional derivative of G w.r.t. its input at z in the direction x or w."""
    if w is None:
        _, G_to_x = G(z + x, c, return_bn=True, Q=Q)
    else:
        _, G_to_x = G(z, c, w=w + x, return_bn=True, Q=Q)

    G_to_x = listify(G_to_x)
    G_z = listify(G_z)
    fdd = [(G2x - G_z_base) / epsilon for G2x, G_z_base in zip(G_to_x, G_z)]
    return fdd


def stack_var_and_reduce(list_of_activations, reduction=torch.max):
    dots = [torch.bmm(x.view(x.size(0), 1, -1), x.view(x.size(0), -1, 1)) for x in list_of_activations]
    stacks = torch.stack(dots)  # (k, N, 1)
    var_tensor = torch.var(stacks, dim=0, unbiased=True)  # (N, 1)
    penalty = reduction(var_tensor)  # (1,) (scalar)
    return penalty


def multi_stack_var_and_reduce(fdds, reduction=torch.max, return_separately=False):
    sum_of_penalties = 0 if not return_separately else []
    for activ_n in zip(*fdds):
        penalty = stack_var_and_reduce(activ_n, reduction)
        sum_of_penalties += penalty if not return_separately else [penalty]
    return sum_of_penalties


def orojar(G, z, c, w=None, G_z=None, k=2, epsilon=0.1, reduction=torch.mean,
                    multiple_layers=True, return_separately=False, Q=None):
    """
    Version of the OroJaR that allows taking the Jacobin matrix w.r.t. the w input instead of z
    Note: w here refers to the coefficients for the learned directions in Q, it has nothing to do with W-space
    as in StyleGAN, etc.

    :param G: Function that maps z to either a tensor or a size-N list of tensors (activations)
    :param z: (N, dim_z) input to G
    :param c: Class label input to G (not regularized in this version of OroJaR)
    :param w: (N, ndirs) tensor that represents how far to move z in each of the ndirs directions stored in Q.
              If specified, Jacobin matrix is taken w.r.t. w instead of w.r.t. z.
    :param k: Number of Jacobin matrix directions to sample (must be >= 2)
    :param G_z: Pre-cached G(z) computation (i.e., a size-N list)
    :param epsilon: Amount to blur G before estimating Jacobin matrix (must be > 0)
    :param reduction: Many-to-one function to reduce each pixel's individual OroJaR into a final loss
    :param multiple_layers: If True, G is expected to return a list of tensors that are all regularized jointly
    :param return_separately: If True, returns OroJaR for each layer separately, rather than combining them
    :param Q: (ndirs, nz) matrix of directions (rows correspond to directions)

    :return: A differentiable scalar (the OroJaR), or a list of regularization of OroJaR if return_separately is True
    """
    if G_z is None:
        G_z = G(z, c, w=w, return_bn=multiple_layers, Q=Q)
        if multiple_layers:
            G_z = G_z[1]
    if w is not None:
        xs = rademacher(torch.Size((k, *w.size()))) * epsilon
    else:
        xs = rademacher(torch.Size((k, *z.size()))) * epsilon
    first_orders = []
    for i in range(k):
        x = xs[i]
        if multiple_layers:
            central_first_order = multi_layer_first_directional_derivative(G, z, c, x, G_z, epsilon, w=w, Q=Q)
        else:
            central_first_order = first_directional_derivative(G, z, c, x, G_z, epsilon, w=w, Q=Q)
        first_orders.append(central_first_order)
    if multiple_layers:
        penalty = multi_stack_var_and_reduce(first_orders, reduction, return_separately)
    else:
        penalty = stack_var_and_reduce(first_orders, reduction)
    return penalty


