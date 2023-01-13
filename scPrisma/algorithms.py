
import numpy as np
from numpy import random

from scPrisma.data_gen import simulate_spatial_cyclic
from scPrisma.pre_processing import *
from scPrisma.visualizations import *
from numba import jit


@jit(nopython=True, parallel=True)
def numba_diagonal(A):
    d = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        d[i] = A[i, i]
    return d


@jit(nopython=True, parallel=True)
def reconstruct_e(E_prob):
    '''
    Greedy algorithm to reconstruct permutation matrix from Bi-Stochastic matrix
    Parameters
    ----------
    E_prob: np.array
        2D Bi-Stochastic matrix
    Returns: np.array
        2D Permutation matrix
    -------
    '''
    res = []
    for i in range(E_prob.shape[0]):
        tmp = -1
        pointer = -1
        for j in range(E_prob.shape[1]):
            if E_prob[i, j] > tmp:
                if not (j in res):
                    tmp = E_prob[i, j]
                    pointer = j
        res.append(pointer)
    res_array = np.zeros(E_prob.shape)
    for i, item in enumerate(res):
        res_array[i, item] = 1
    return res_array

@jit(nopython=True, parallel=True)
def boosted_reconstruct_e(E_prob):
    '''
    Greedy algorithm to reconstruct permutation matrix from Bi-Stochastic matrix
    Parameters
    ----------
    E_prob: np.array
        2D Bi-Stochastic matrix
    Returns: np.array
        2D Permutation matrix
    -------
    '''
    res = []
    res_array = np.zeros(E_prob.shape)
    for i in range(E_prob.shape[0]):
        tmp = -1
        pointer = -1
        for j in range(E_prob.shape[1]):
            if E_prob[i, j] > tmp and not (j in res):
                tmp = E_prob[i, j]
                pointer = j
        res.append(pointer)
        res_array[i, pointer] = 1
    return res_array


def ge_to_spectral_matrix(A, optimize_alpha=True):
    '''
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    optimize_alpha: bool
        Find the alpha value using optimization problem or by using the close formula


    Returns: np.array
        spectral matrix (concatenated eigenvectors multiplied by their appropriate eigenvalues)
    -------

    '''
    n = A.shape[0]
    p = A.shape[1]
    min_np = min(n, p)
    if optimize_alpha:
        u, s, vh = np.linalg.svd(A)
        for i in range(min_np):
            s[i] *= s[i]
        alpha = optimize_alpha_p(s, 15)
    else:
        alpha = np.exp(-2 / p)
    V = generate_spectral_matrix(n=n, alpha=alpha)
    # Removing the eigenvector which is related to the largest eigenvalue improve the results
    # This eigenvector is the 'offset' of the data
    V = V[1:, :]
    return V


def sga_reorder_rows_matrix(A, iterNum=300, batch_size=20):
    '''
    Reconstruction algorithm (without momentum)
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    iterNum: int
        Number of stochastic gradient ascent iterations
    batch_size: int
        batch size, number of genes sampled per batch

    Returns
    E: np.array
        Bi-Stochastic matrix
    E_recon: np.array
        Permutation matrix (which is calculated by greedy rounding of 'E' matrix).

    -------

    '''
    A = cell_normalization(A)
    n = A.shape[0]
    V = ge_to_spectral_matrix(A)
    E = sga_matrix(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size)
    E_recon = reconstruct_e(E)
    return E, E_recon


@jit(nopython=True, parallel=True)
def sga_matrix(A, E, V, lr, iterNum, batch_size):
    """
    Perform stochastic gradient descent optimization to find the optimal value of the bi-stochastic matrix E.

    Parameters
    ----------
    A : numpy array, shape (n, p)
        A gene expression matrix of inputs, where p is the number of genes and n is the number of cells.
    E : numpy array, shape (n, n)
        The current value of the bi-stochastic matrix E to be optimized, where n is the number of cells.
    V : numpy array, shape (n, n-1)
        Theoretical spectrum
    lr : float
        The initial learning rate for the gradient descent updates.
    iterNum : int
        The number of iterations to perform the optimization for.
    batch_size : int
        The number of examples to use in each batch for the gradient descent updates.

    Returns
    -------
    E : numpy array, shape (n, n)
        The optimized value of the bi-stochastic matrix E.
    """
    # Initialize loop counter and function value
    j = 0
    value = 0

    # Set the initial step size for the gradient descent updates
    epsilon_t = lr

    # Begin loop to perform gradient descent optimization
    while (j < iterNum):
        # Print iteration number and function value every 25 iterations
        if j % 25 == 0:
            print("Iteration number: " + str(j) + " function value= " + str(value))
            print("Iteration number: " + str(j))

        # Decrease the step size for the gradient descent updates
        epsilon_t *= 0.995

        # Select a batch of columns from the matrix A
        A_tmp = A[:, np.random.randint(A.shape[1], size=batch_size)]

        # Evaluate the function and its gradient at the current value of E
        value, grad = fAndG_matrix(A=A_tmp, E=E, V=V)

        # Perform a gradient descent update on E
        E = E + epsilon_t * grad

        # Apply the BBS function to E (Bi-Stochastic projection)
        E = BBS(E)

        # Print the iteration number and function value
        print("Iteration number: " + str(j) + " function value= " + str(value))

        # Increment loop counter
        j += 1

    # Return the optimized value of E
    return E


@jit(nopython=True, parallel=True)
def fAndG_matrix(A, E, V):
    '''
    Calculate the function value and the gradient of A matrix
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    E: np.array
        Bi-Stochastic matrix (should be constant)
    V: np.array:
        Theoretical spectrum

    Returns
    -------
    functionValue: int
        function value
    gradient: np.array
        gradient of E
    '''
    functionValue = np.trace((((((V.T).dot(E)).dot(A)).dot(A.T)).dot(E.T)).dot(V))
    gradient = (2 * ((((V).dot(V.T)).dot(E)).dot(A)).dot(A.T))

    return functionValue, gradient


@jit(nopython=True, parallel=True)
def g_matrix(A, E, VVT):
    '''
    Calculate the function value and the gradient of A matrix, using boosted formula
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    E: np.array
        Bi-Stochastic matrix (should be constant)
    VVT: np.array:
        Theoretical spectrum (V) multiplied by his transform (V.T)

    Returns
    -------
    gradient: np.array
        gradient of E
    '''
    gradient = (2 * (((VVT).dot(E)).dot(A)).dot(A.T))
    return gradient


@jit(nopython=True, parallel=True)
def sga_matrix_momentum(A, E, V, iterNum=400, batch_size=20, lr=0.1, gama=0.9, verbose=True):
    """
    Perform stochastic gradient ascent optimization with momentum to find the optimal value of the bi-stochastic matrix E.

    Parameters
    ----------
    A : numpy array, shape (n, p)
        A matrix of inputs, where p is the number of genes and n is the number of cells.
    E : numpy array, shape (n, n)
        The current value of the bi-stochastic matrix E to be optimized, where n is the number of cells.
    V : numpy array, shape (n, n-1)
        Theoretical spectrum.
    iterNum : int, optional
        The number of iterations to perform the optimization for. Default is 400.
    batch_size : int, optional
        The number of examples to use in each batch for the gradient descent updates. Default is 20.
    lr : float, optional
        The learning rate for the gradient ascent updates. Default is 0.1.
    gama : float, optional
        The momentum parameter. Default is 0.9.
    verbose : bool, optional
        A flag indicating whether to print the iteration number and function value every 25 iterations. Default is True.

    Returns
    -------
    E : numpy array, shape (n, n)
        The optimized value of the bi-stochastic matrix E.
    """
    # Initialize loop counter and function value
    j = 0
    value = 0

    # Pre-compute the product VVT for runtime optimization
    VVT = (V).dot(V.T)

    # Set the initial step size for the gradient ascent updates
    epsilon_t = lr

    # Initialize the momentum step
    step = np.zeros(E.shape)

    # Begin loop to perform gradient ascent optimization with momentum
    while (j < iterNum):
        # Print iteration number and function value every 25 iterations if verbose flag is set
        if (j % 25 == 0) & verbose:
            value, grad = fAndG_matrix(A=A, E=E, V=V)
            print("Iteration number: ")
            print(j)
            print(" function value= ")
            print(value)

        # Select a batch of columns from the matrix A
        A_tmp = A[:, np.random.randint(0, A.shape[1], batch_size)]

        # Evaluate the gradient at the current value of E
        grad = g_matrix(A=A_tmp, E=E, VVT=VVT)

        # Update the momentum step
        step = epsilon_t * grad + gama * step

        # Perform a gradient ascent update on E
        E = E + step

        # Apply the BBS function to E
        E = BBS(E)

        # Increment loop counter
        j += 1

    # Return the optimized value of E
    return E


def sga_matrix_momentum_indicator(A, E, V, IN, iterNum=400, batch_size=20, lr=0.1, gama=0.9):
    '''
    Reconstruction algorithm with optional use of prior knowledge
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    E: np.array
        Initial Bi-Stochastic matrix (should be constant)
    V: np.array:
        Theoretical spectrum
    IN: np.array:
        Indicator matrix which will be later entry-wise multiplied by the permutation matrix
    iterNum: int
        Number of stochastic gradient ascent iterations
    batch_size: int
        batch size, number of genes sampled per batch
    lr: int
        Learning rate
    gama: int
        Momentum parameter

    Returns
        E: np.array
            Bi-Stochastic matrix

    -------

    '''
    j = 0
    value = 0
    epsilon_t = lr
    step = np.zeros(E.shape)
    E = E * IN
    E = BBS(E) * IN
    while (j < iterNum):
        if j % 25 == 0:
            print("Iteration number: " + str(j) + " function value= " + str(value))
        A_tmp = A[:, np.random.randint(A.shape[1], size=batch_size)]
        value, grad = fAndG_matrix(A=A_tmp, E=E, V=V)
        grad = grad
        step = epsilon_t * grad + gama * step
        E = E + step
        E = BBS(E) * IN
        j += 1
    return E


def sga_m_reorder_rows_matrix(A, iterNum=300, batch_size=None, gama=0.5, lr=0.1):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    iterNum: int
        Number of stochastic gradient ascent iterations
    batch_size: int
        batch size, number of genes sampled per batch
    lr: int
        Learning rate
    gama: int
        Momentum parameter

    Returns
    E: np.array
        Bi-Stochastic matrix
    E_recon: np.array
        Permutation matrix (which is calculated by greedy rounding of 'E' matrix).
    -------

    '''
    if batch_size == None:
        batch_size = int((A.shape[0]) * 0.75)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    E = sga_matrix_momentum(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size, gama=gama, lr=lr)
    E_recon = reconstruct_e(E)
    return E, E_recon


def reconstruction_cyclic(A, iterNum=300, batch_size=None, gama=0.5, lr=0.1, verbose=True, final_loss=False):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    iterNum: int
        Number of stochastic gradient ascent iterations
    batch_size: int
        batch size, number of genes sampled per batch
    lr: int
        Learning rate
    gama: int
        Momentum parameter
    verbose: bool
        verbosity
    final_loss: bool
        For validation, retain False
    Returns
    E: np.array
        Bi-Stochastic matrix
    E_recon: np.array
        Permutation matrix (which is calculated by greedy rounding of 'E' matrix).
    -------

    '''
    A = np.array(A).astype('float64')
    if batch_size == None:
        batch_size = int((A.shape[0]) * 0.75)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    E = sga_matrix_momentum(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size, gama=gama, lr=lr,
                            verbose=verbose)
    E_recon = reconstruct_e(E)
    if final_loss:
        value, grad = fAndG_matrix(A=((1 / A.shape[0]) * A), E=E_recon, V=V.T)
        return E, E_recon, value
    return E, E_recon


def filter_non_cyclic_genes(A, regu=0.1, lr=0.1, iterNum=500) -> np.array:
    '''
    Filter out genes which are not smooth over the inferred circular topology. As a prior step for this algorithm, the reconstruction algorithm should be applied.

    Parameters
    ----------
    A: np.array
        Gene expression matrix
    regu: float
        Regularization coefficient, large regularization would lead to more non-cyclic genes which will be filtered out
    lr: int
        Learning rate
    iterNum: int
        Number of stochastic gradient ascent iterations

    Returns
    -------
    D: np.array
        diagonal filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    U = ge_to_spectral_matrix(A)
    A = gene_normalization(A)
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2, U=U.T, regu=regu, lr=lr, iterNum=iterNum)
    return D


def filter_cyclic_genes(A, regu=0.1, iterNum=500, lr=0.1, verbose=True):
    '''
    Filter out genes which are smooth over the inferred circular topology. As a prior step for this algorithm, the reconstruction algorithm should be applied.

    Parameters
    ----------
    A: np.array
        Gene expression matrix
    regu: float
        Regularization coefficient, large regularization would lead to more cyclic genes which will be filtered out
    lr: int
        Learning rate
    iterNum: int
        Number of stochastic gradient ascent iterations

    Returns
    -------
    D: np.array
        diagonal filtering matrix

    '''
    A = np.array(A).astype('float64')
    V = cell_normalization(A)
    p = V.shape[1]
    U = ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2, ascent=-1, U=U.T, regu=regu, iterNum=iterNum, lr=lr,
                                      verbose=verbose)
    return D


def filter_cyclic_genes_line(A, regu=0.1, iterNum=500, lr=0.1, verbosity=25):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    A = np.array(A).astype('float64')
    V = cell_normalization(A)
    p = V.shape[1]
    U = ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_descent_filter_matrix_line(A, D=np.identity((p)), U=U.T, regu=regu, max_evals=iterNum,
                                            verbosity=verbosity)
    return D


def filter_linear_padded_genes_line(A, regu=0.1, iterNum=500, lr=0.1, verbosity=25):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    Padded_array = np.zeros((int(A.shape[0] / 2), A.shape[1]))
    V = np.concatenate([A, Padded_array])
    V = cell_normalization(V)
    p = V.shape[1]
    U = ge_to_spectral_matrix(V)
    V = gene_normalization(V)
    D = gradient_ascent_filter_matrix(V, D=np.identity((p)) / 2, ascent=-1, U=U.T, regu=regu, iterNum=iterNum, lr=lr)
    return D


def filter_linear_genes_line(A, regu=0.1, iterNum=500, lr=0.1, verbosity=25, method='numeric'):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    p = A.shape[1]
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    A = gene_normalization(A)
    D = gradient_descent_filter_matrix_line(A, D=np.identity((p)), U=eigenvectors[:, 1:], regu=regu, max_evals=iterNum,
                                            verbosity=verbosity)
    return D


def filter_non_cyclic_genes_line(A, regu=0.1, iterNum=500, lr=0.1, verbosity=25):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    A = np.array(A).astype('float64')
    V = cell_normalization(A)
    p = V.shape[1]
    U = ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_descent_filter_matrix_line(A, D=np.identity((p)), U=U.T, regu=regu, max_evals=iterNum,
                                            verbosity=verbosity)
    np.identity(D.shape[1])
    return (np.identity(D.shape[1]) - D)


@jit(nopython=True, parallel=True)
def gradient_ascent_filter_matrix(A, D, U, ascent=1, lr=0.1, regu=0.1, iterNum=400, verbose=True):
    '''
    :param A: gene expression matrix
    :param D: diagonal filter matrix (initial value)
    :param U: Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix
    :param ascent: 1 - gradient ascent , -1 - gradient decent
    :param lr: learning rate
    :param regu: regularization parameter
    :param iterNum:  iteration number
    :return: diagonal filter matrix
    '''
    j = 0
    val = 0
    epsilon_t = lr
    ATUUTA = (2 * ((((A.T).dot(U)).dot(U.T)).dot(A)))  # .dot(D)) - regu*np.sign(D)#((1 / t_0) * ((A.T).dot(A)).dot(D))
    while (j < iterNum):
        if j % 25 == 1:
            if verbose:
                val = np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu * np.linalg.norm(D, 1)
                print("Iteration number: ")
                print(j)
                print("function value= ")
                print(val)
        epsilon_t *= 0.995
        T = numba_diagonal(D)  # .diagonal()
        grad = ATUUTA * T - regu * np.sign(D)
        D = D + ascent * epsilon_t * grad
        D = diag_projection(D)
        # print("Iteration number: " + str(j) + " grad value= " + str(grad))
        j += 1
    return D


# def loss_filter_genes(A,U,D,regu):
#    return np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu*np.linalg.norm(D,1)

@jit(nopython=True, parallel=True)
def loss_filter_genes(ATU, D, regu):
    D_diag = numba_diagonal(D)
    return np.trace((ATU.T * D_diag * D_diag).dot(ATU)) - regu * np.linalg.norm(D, 1)


@jit(nopython=True, parallel=True)
def gradient_descent_filter_matrix_line(A, D, U, regu=0.1, gamma=1e-04, max_evals=250, verbosity=float('inf')):
    '''
    :param A: ene expression matrix
    :param D: diagonal filter matrix (initial value)
    :param U: Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix
    :param lr: learning rate
    :param regu: regularization parameter
    :param max_evals:  iteration number
    :return: diagonal filter matrix
    '''
    ATUUTA = (2 * ((((A.T).dot(U)).dot(U.T)).dot(A)))
    w = D
    evals = 0
    ATU = (A.T).dot(U)
    loss = loss_filter_genes(ATU=ATU, D=w, regu=regu)
    w = diag_projection(w)

    grad = ATUUTA * w - regu * np.sign(w)
    G = numba_diagonal(grad)  # .diagonal()
    grad = np.diag(G)
    alpha = 1 / np.linalg.norm(grad)
    while evals < max_evals and np.linalg.norm(grad) > 1e-07:
        evals += 1
        if evals % verbosity == 0:
            print((evals))
            print('th Iteration    Loss :: ')
            print((loss))
            print(' gradient :: ')
            print((np.linalg.norm(grad)))
        gTg = np.linalg.norm(grad)
        gTg = gTg * gTg
        new_w = w - alpha * grad
        new_loss = loss_filter_genes(ATU, new_w, regu)
        new_w = diag_projection(new_w)
        new_grad = ATUUTA * new_w - regu * np.sign(new_w)
        G = numba_diagonal(new_grad)  # .diagonal()
        new_grad = np.diag(G)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_loss = loss_filter_genes(ATU, new_w, regu)
            new_w = diag_projection(new_w)
            new_grad = ATUUTA * new_w - regu * np.sign(new_w)
            G = numba_diagonal(new_grad)  # .diagonal()
            new_grad = np.diag(G)

        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w


@jit(nopython=True, parallel=True)
def fAndG_filter_matrix(A, D, U, alpha):
    functionValue = (np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - (alpha * np.sum(np.abs(D))))
    gradient = ((2 * ((((A.T).dot(U)).dot(U.T)).dot(A)).dot(D)) - (alpha * np.sign(D)))
    return gradient, functionValue


@jit(nopython=True, parallel=True)
def fAndG_fixed_filter(A, D, U, regu):
    t_0 = np.linalg.norm((A).dot(D), 'fro')
    functionValue = np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu * t_0
    gradient = (2 * ((((A.T).dot(U)).dot(U.T)).dot(A)).dot(D)) - regu * np.sign(
        D)  # ((1 / t_0) * ((A.T).dot(A)).dot(D))

    return functionValue, gradient


@jit(nopython=True, parallel=True)
def gradient_ascent_filter(A, D, eigenvectors_list, eigenvalues_list, epsilon=0.1, regu=0.1, iterNum=400):
    '''
    :param A: Gene expression matrix
    :param D: diagonal filtering matrix
    :param eigenvectors_list:
    :param eigenvalues_list:
    :param epsilon:
    :param regu:
    :param iterNum:
    :return:
    '''
    # print(eigenvectors_list.shape)
    j = 0
    epsilon_t = epsilon
    while (j < iterNum):
        value = 0
        if j % 25 == 0:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        grad = np.zeros(D.shape)
        for i, v in enumerate(eigenvectors_list):
            tmp_value, tmp_grad = fAndG_regu(A=A, E=D, alpha=regu, x=v * eigenvalues_list[i])
            grad += tmp_grad
            value += tmp_value
        D = D + epsilon_t * grad
        D = diag_projection(D)
        print("Iteration number: " + str(j) + " function value= " + str(value))
        j += 1
    return D


@jit(nopython=True)
def diag_projection(D):
    T = numba_diagonal(D)  # .diagonal()
    T = numba_vec_clip(T, len(T), 0, 1)
    return np.diag(T)


@jit(nopython=True, parallel=True)
def fAndG_regu(A, E, alpha, x):
    t_0 = (A.T).dot(x)
    t_1 = np.linalg.norm(E, 'fro')
    functionValue = ((x).dot((A).dot((E).dot((E.T).dot(t_0)))) - (alpha * t_1))
    gradient = ((2 * np.multiply.outer(t_0, ((x).dot(A)).dot(E))) - ((alpha / t_1) * E))
    return functionValue, gradient


def calculate_roc_auc(y_target, y_true):
    return roc_auc_score(y_true, y_target)


def filter_full(A, regu=0.1, iterNum=300):
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    # F = gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum)
    return F


def enhancement_cyclic(A, regu=0.1, iterNum=300, verbosity=25):
    ''' Enhancement of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum, verbosity=verbosity)
    return F


def filtering_cyclic(A, regu=0.1, iterNum=300, verbosity=25, error=10e-7, optimize_alpha=True, line_search=True):
    ''' Filtering of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A, optimize_alpha=optimize_alpha)
    print("starting filtering")
    if line_search:
        F = gradient_descent_full_line(A, F=np.ones(A.shape), V=V.T, regu=regu, max_evals=iterNum, verbosity=verbosity,
                                       error=error)
    else:
        F = gradient_descent_full(A, np.ones(A.shape), V=V.T, regu=regu, epsilon=0.1, iterNum=iterNum)
    return F


def filtering_cyclic_boosted(A, regu=0.1, iterNum=300, verbosity=25, error=10e-7):
    ''' Filtering of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    print("starting filtering")
    A = gradient_descent_full_line_boosted(A, V=V.T, regu=regu, max_evals=iterNum, verbosity=verbosity, error=error)
    return A


# def filter_cyclic_full(A, regu=0.1, iterNum=300):
#    A = cell_normalization(A)
#    V = ge_to_spectral_matrix(A)
#    F = gradient_descent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum)
#    return F

def filter_cyclic_full_line(A, regu=0.1, iterNum=300, verbosity=25, error=10e-7):
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    print("starting filtering")
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=V.T, regu=regu, max_evals=iterNum, verbosity=verbosity,
                                   error=error)
    return F


def filter_non_cyclic_full_reverse(A, regu=0.1, iterNum=300):
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=V.T, regu=regu, max_evals=iterNum)
    F = np.ones(F.shape) - F
    return F


@jit(nopython=True, parallel=True)
def gradient_ascent_full(A, F, V, regu, epsilon=0.1, iterNum=400):
    j = 0
    epsilon_t = epsilon
    while (j < iterNum):
        value = 0
        if j % 50 == 1:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        tmp_value, grad = fAndG_full(A=A, B=F, V=V, alpha=regu)
        F = F + epsilon_t * grad
        F = numba_clip(F, F.shape[0], F.shape[1], 0, 1)
        j += 1
    return F


@jit(nopython=True, parallel=True)
def stochastic_gradient_ascent_full(A, F, V, regu, epsilon=0.1, iterNum=400, regu_norm='L1', verbosity=25):
    '''
    :param A: gene expression matrix
    :param F: filtering matrix
    :param V: theoretic spectrum of covariance matrix
    :param regu: regularization coefficient
    :param epsilon: step size (learning rate)
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    # print(A.shape)
    # print(F.shape)
    # print(V.shape)
    j = 0
    epsilon_t = epsilon
    VVT = V.dot(V.T)
    while (j < iterNum):
        if j % verbosity == 1:
            value, grad = fAndG_full_acc(A=A, B=F, V=V, VVT=VVT, alpha=regu, regu_norm=regu_norm)
            print("Iteration number: ")
            print((j))
            print("function value: ")
            print((value))
        epsilon_t *= 0.995
        grad = G_full(A=A, B=F, V=V, alpha=regu, regu_norm=regu_norm)
        F = F + epsilon_t * (grad + np.random.normal(0, 0.01, grad.shape))
        F = numba_clip(F, F.shape[0], F.shape[1], 0, 1)
        j += 1
    return F


@jit(nopython=True, parallel=True)
def gradient_descent_full(A, F, V, regu, epsilon=0.1, iterNum=400, regu_norm='L1'):
    print(A.shape)
    print(F.shape)
    print(V.shape)
    j = 0
    epsilon_t = epsilon
    while j < iterNum:
        if j % 100 == 1:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        tmp_value, grad = fAndG_full(A=A, B=F, V=V, alpha=regu, regu_norm=regu_norm)
        F = F - epsilon_t * grad
        F = numba_clip(F, F.shape[0], F.shape[1], 0, 1)
        j += 1
    return F


@jit(nopython=True, parallel=True)
def gradient_descent_full_line(A, F, V, regu, gamma=1e-04,
                               max_evals=250,
                               verbosity=float('inf'), error=1e-07, regu_norm='L1'):
    '''
    :param A:
    :param F:
    :param V:
    :param regu:
    :param gamma:
    :param max_evals:
    :param verbosity:
    :param error:
    :return:
    '''
    VVT = V.dot(V.T)
    w = F
    evals = 0
    loss, grad = fAndG_full_acc(A=A, B=F, V=V, VVT=VVT, alpha=regu, regu_norm=regu_norm)
    alpha = 1 / np.linalg.norm(grad)
    # alpha=0.1
    prev_w = np.zeros(w.shape)
    while evals < max_evals and np.linalg.norm(w - prev_w) > error:
        prev_w = np.copy(w)
        evals += 1
        if evals % verbosity == 0:
            print(str(evals))
            print('th Iteration    Loss :: ')
            print((loss))
            # + ' gradient :: ' +  str(np.linalg.norm(grad)))
        gTg = np.linalg.norm(grad)
        gTg = gTg * gTg
        new_w = w - alpha * grad
        new_w = numba_clip(new_w, new_w.shape[0], new_w.shape[1], 0, 1)
        # new_w = new_w.clip(min=0, max=1)
        new_loss, new_grad = fAndG_full_acc(A=A, B=new_w, V=V,
                                            VVT=VVT, alpha=regu, regu_norm=regu_norm)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_w = numba_clip(new_w, new_w.shape[0], new_w.shape[1], 0, 1)
            # new_w = new_w.clip(min=0, max=1)
            new_loss, new_grad = fAndG_full_acc(A=A, B=new_w, V=V, VVT=VVT,
                                                alpha=regu, regu_norm=regu_norm)
        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w


@jit(nopython=True, parallel=True)
def fAndG_filtering_boosted(A, V, alpha_matrix):
    functionValue = np.trace((((V.T).dot(A)).dot(A.T)).dot(V)) - np.linalg.norm(alpha_matrix * A, 1)
    gradient = (2 * ((V).dot(V.T)).dot(A)) - alpha_matrix * np.sign(A)
    return functionValue, gradient


@jit(nopython=True, parallel=True)
def gradient_descent_full_line_boosted(A, V, regu, gamma=1e-04, max_evals=250, verbosity=float('inf'), error=1e-07):
    '''
    :param A:
    :param F:
    :param V:
    :param regu:
    :param gamma:
    :param max_evals:
    :param verbosity:
    :param error:
    :return:
    '''

    alpha_matrix = (regu / (A + 10e-5)) / (A + 10e-5)
    tmp_A = copy.deepcopy(A)
    VVT = V.dot(V.T)
    w = A
    evals = 0
    loss, grad = fAndG_filtering_boosted(A, V, alpha_matrix)
    alpha = 1 / np.linalg.norm(grad)
    # alpha=0.1
    prev_w = np.zeros(w.shape)
    while evals < max_evals and np.linalg.norm(w - prev_w) > error:
        prev_w = copy.deepcopy(w)
        evals += 1
        if evals % verbosity == 0:
            print(str(evals) + 'th Iteration    Loss :: ' + str(loss) + ' gradient :: ' + str(np.linalg.norm(grad)))
        gTg = np.linalg.norm(grad)
        gTg = gTg * gTg
        new_w = w - alpha * grad
        new_w = new_w.clip(min=0, max=tmp_A)
        new_loss, new_grad = fAndG_filtering_boosted(A=new_w, V=V, alpha_matrix=alpha_matrix)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_w = new_w.clip(min=0, max=tmp_A)
            new_loss, new_grad = fAndG_filtering_boosted(A=new_w, V=V, alpha_matrix=alpha_matrix)
        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w


@jit(nopython=True, parallel=True)
def fAndG_full(A, B, V, alpha, regu_norm='L2'):
    '''
    :param A: Gene expression matrix
    :param B: filtering matrix
    :param V: spectral matrix
    :param alpha: correlation between neighbors
    :return:projection over theoretic spectrum and gradient according to 'B'
    '''

    if regu_norm == 'L1':
        T_0 = (A * B)
        t_1 = np.linalg.norm(B, 1)
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (alpha * t_1))
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - ((alpha) * np.sign(B)))
    else:
        T_0 = (A * B)
        t_1 = np.linalg.norm(A * B, 'fro')
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (alpha * t_1))
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - ((alpha / t_1) * B))
    return functionValue, gradient


@jit(nopython=True, parallel=True)
def fAndG_full_acc(A, B, V, VVT, alpha, regu_norm):
    '''
    :param A: Gene expression matrix
    :param B: filtering matrix
    :param V: spectral matrix
    :param VVT: V.dot(V.T)
    :param alpha: correlation between neighbors
    :param regu_norm: regularization norm (L1/L2)
    :return:projection over theoretic spectrum and gradient according to 'B'
    '''
    if regu_norm == 'L1':
        T_0 = (A * B)
        t_1 = np.linalg.norm(B, 1)
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (alpha * t_1))
        gradient = ((2 * ((VVT).dot(T_0) * A)) - ((alpha) * np.sign(B)))
    else:
        T_0 = (A * B)
        t_1 = np.linalg.norm(A * B, 'fro')
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (alpha * t_1))
        gradient = ((2 * ((VVT).dot(T_0) * A)) - ((alpha / t_1) * B))
    return functionValue, gradient


@jit(nopython=True, parallel=True)
def G_full(A, B, V, alpha, regu_norm='L1'):
    '''
    :param A: Gene expression matrix
    :param B: filtering matrix
    :param V: spectral matrix
    :param alpha: correlation between neighbors
    :return:projection over theoretic spectrum and gradient according to 'B'
    '''
    if regu_norm == 'L1':
        T_0 = (A * B)
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - ((alpha) * np.sign(B)))
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (alpha * np.linalg.norm(B, 1)))
    else:
        T_0 = (A * B)
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - (alpha) * 2 * B)
    return gradient


def sga_m_linear_reorder_rows_matrix(A, iterNum=1000, batch_size=400):
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    # u, s, vh = np.linalg.svd(A)
    np1 = min(n, p)
    # for i in range(np1):
    #    s[i]*=s[i]
    # alpha = optimize_alpha_p(s,15)
    # V, eigenvalues = get_numeric_eigen_values(get_alpha(A.shape[0], 10), n)
    eigenvalues, V = linalg.eig(A.dot(A.T))
    V, eigenvalues = get_psuedo_data(A.shape[1], A.shape[0], 10)
    for i in range(A.shape[0]):
        V[:, i] = V[:, i] * (eigenvalues[i])
    E = sga_matrix_momentum(A, E=np.ones((n, n)) / n, V=V[:, 1:], iterNum=iterNum, batch_size=batch_size)
    E_recon = reconstruct_e(E)
    return E, E_recon


def filter_non_cyclic_genes_vector(A, alpha=0.99, regu=2, iterNum=500):
    A = np.array(A).astype('float64')
    A = gene_normalization(A)
    n = A.shape[0]
    p = A.shape[1]

    eigenvectors, eigenvalues = generate_eigenvectors_circulant()

    D = gradient_ascent_filter(A, D=np.identity((p)), eigenvectors_list=eigenvectors[1:],
                               eigenvalues_list=eigenvalues[1:], regu=regu, iterNum=iterNum)
    return D


def filter_linear_full(A: np.ndarray, method: str, regu: float = 0.1, iterNum: int = 300, lr: float = 0.1,
                       regu_norm: str = 'L1') -> np.ndarray:
    """
    Apply linear filtering on the full data.

    Parameters
    ----------
    A : np.ndarray
        gene expression matrix
    method : str
        method to get eigenvectors
    regu : float, optional
        regularization coefficient, by default 0.1
    iterNum : int, optional
        iteration number, by default 300
    lr : float, optional
        learning rate, by default 0.1
    regu_norm : str, optional
        Regularization norm, by default 'L1'

    Returns
    -------
    np.ndarray
        filtered matrix
    """
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(ngenes=A.shape[1], optimized_alpha=False, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = gradient_descent_full(A, F=np.ones(A.shape), V=eigenvectors[:, 1:], regu=regu,
                              iterNum=iterNum, epsilon=lr, regu_norm=regu_norm)
    return F


def enhancement_linear(A: np.ndarray, regu: float = 0.1, iterNum: int = 300, method: str = 'numeric') -> np.ndarray:
    """
    Enhancement of linear signal

    Parameters
    ----------
    A : np.ndarray
        Gene expression matrix (reordered according to linear ordering)
    regu : float, optional
        regularization coefficient, by default 0.1
    iterNum : int, optional
        iteration number, by default 300

    Returns
    -------
    np.ndarray
        filtering matrix
    """
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=eigenvectors[:, 1:], regu=regu, iterNum=iterNum)
    return F


def filtering_linear(A, method, regu=0.1, iterNum=300, verbosity=25,
                     error=10e-7, optimized_alpha=True, regu_norm='L1'):
    ''' Filtering of linear signal
    :param A: Gene expression matrix (reordered according to linear ordering)
    :param regu: regularization coefficient
    :param iterNumenhance_linear_full: iteration number
    :return: filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    print(A.shape)
    alpha = get_alpha(ngenes=A.shape[1], optimized_alpha=optimized_alpha, eigenvals=real_eigenvalues)
    print(alpha)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=eigenvectors[:, 1:],
                                   regu=regu, max_evals=iterNum, verbosity=verbosity,
                                   error=error, regu_norm=regu_norm)
    return F


#    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=eigenvectors[:,1:],
#                                   regu=regu, max_evals=iterNum,verbosity=verbosity ,
#                                   error=error , regu_norm=regu_norm)

#

def enhance_linear_genes(A, method, regu=2, iterNum=500, lr=0.1):
    A = np.array(A).astype('float64')
    A = gene_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    n = A.shape[0]
    p = A.shape[1]
    eigenvectors = get_linear_eig_data(n, alpha, method=method,
                                       normalize_vectors=True)
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2,
                                      U=eigenvectors[:, 1:], regu=regu,
                                      iterNum=iterNum, lr=lr)
    return D


def filter_linear_genes(A, method='numeric', regu=2, iterNum=500, lr=0.1):
    A = np.array(A).astype('float64')
    A = gene_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    n = A.shape[0]
    p = A.shape[1]
    eigenvectors = get_linear_eig_data(n, alpha, method=method,
                                       normalize_vectors=True)
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2,
                                      U=eigenvectors[:, 1:], regu=regu,
                                      iterNum=iterNum, lr=lr, ascent=-1)
    return D


@jit(nopython=True)
def numba_min_clip(A: np.ndarray, a_min: int) -> np.ndarray:
    """
    Implementing np.clip for a minimum value using numba

    Parameters
    ----------
    A : np.ndarray
        Array
    a_min : int
        minimum value to clip

    Returns
    -------
    np.ndarray
        clipped array
    """
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] < a_min:
                A[i, j] = a_min
    return A


@jit(nopython=True)
def numba_clip(A: np.ndarray, n: int, m: int, a_min: int, a_max: int) -> np.ndarray:
    """
    Implementing np.clip using numba

    Parameters
    ----------
    A : np.ndarray
        Array
    n : int
        number of rows
    m : int
        number of columns
    a_min : int
        minimum value to clip
    a_max : int
        maximum value to clip

    Returns
    -------
    np.ndarray
        clipped array
    """
    for i in range(n):
        for j in range(m):
            if A[i, j] < a_min:
                A[i, j] = a_min
            elif A[i, j] > a_max:
                A[i, j] = a_max
    return A


@jit(nopython=True)
def numba_vec_clip(v: list, n: int, a_min: int, a_max: int):
    """
    Implementing np.clip using numba for vectors

    Parameters
    ----------
    v : list
        vector
    n : int
        number of entries
    a_min : int
        minimum value to clip
    a_max : int
        maximum value to clip

    Returns
    -------
    np.ndarray
        clipped vector
    """
    for i in range(n):
        if v[i] < a_min:
            v[i] = a_min
        elif v[i] > a_max:
            v[i] = a_max
    return v


@jit(nopython=True, parallel=True)
def BBS(E: np.ndarray, iterNum: int = 1000, early_exit: int = 15) -> np.ndarray:
    """
    Bregmanian Bi-Stochastication algorithm as described in
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/KAIS_BBS_final.pdf

    Parameters
    ----------
    E : np.ndarray
        permutation matrix
    iterNum : int, optional
        iteration number, by default 1000
    early_exit : int, optional
        early exit number, by default 15

    Returns
    -------
    np.ndarray
        Bi-Stochastic matrix
    """
    n = E.shape[0]
    prev_E = np.empty(E.shape)
    I = np.identity(n)
    ones_m = np.ones((n, n))
    for i in range(iterNum):
        if i % early_exit == 1:
            prev_E = np.copy(E)
        ones_E = ones_m.dot(E)
        E = E + (1 / n) * (I - E + (1 / n) * (ones_E)).dot(ones_m) - (1 / n) * ones_E
        E = numba_min_clip(E, E.shape[0], E.shape[0], 0)
        if i % early_exit == 1:
            if np.linalg.norm(E - prev_E) < ((10e-6) * n):
                break
    return E



def reorder_indicator(A, IN, iterNum=300, batch_size=20, gama=0, lr=0.1):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    :param A: gene expression matrix
    :param iterNum:  iteration number
    :param batch_size: batch size
    :param gama: momentum parameter
    :return: permutation matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    E = np.ones((n, n)) / n
    E = E * IN
    E = BBS(E)  # *IN
    # plt.imshow(E)
    # plt.show()
    E = sga_matrix_momentum_indicator(A, E, V=V.T, IN=IN, iterNum=iterNum, batch_size=batch_size, gama=gama, lr=lr)
    E_recon = reconstruct_e(E)
    return E, E_recon


def enhance_general_topology(A, V, regu=0.5, iterNum=300):
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V, regu=regu, iterNum=iterNum)
    return F


def gene_inference_general_topology(A, V, regu=0.5, iterNum=100, lr=0.1):
    A = np.array(A).astype('float64')
    A = gene_normalization(A)
    p = A.shape[1]
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2, U=V, regu=regu, lr=lr, iterNum=iterNum)
    return D


def gene_inference_tuner(A, ngenes=10, w=0.05):
    '''
    Auto tune of the regularization parameter for the gene inference task
    :param A: gene expression matrix
    :param ngenes: number of simulated genes
    :param w: simulated genes window length
    :return: regularization parameter
    '''
    A_copy = A.copy()
    n = A.shape[0]
    B = simulate_spatial_cyclic(ngenes=ngenes, ncells=n, w=w)
    C = np.random.normal(0, 0.1, (n, ngenes))
    C = np.clip(C, 0, np.inf)
    A = np.concatenate((A_copy, B, C), axis=1)
    y_true = np.zeros(ngenes * 2)
    y_true[:ngenes] = 1
    regu = tuner_gd(A, y_true, ngenes=10)
    return regu


def tuner_gd(A, y_true, precision=1e-3, rate=1e-2, max_iters=100, ngenes=10):
    ''' Gradient descent for the autotuner
    :param A: gene expression matrix
    :param y_true:
    :param precision: precision os gd
    :param rate: learning rate
    :param max_iters: maximum number of iterations
    :param ngenes: number of genes to simulate
    :return: regularization parameter
    '''
    iters = 0
    previous_step_size = 1
    cur_x = 0
    while previous_step_size > precision and iters < max_iters:
        rate *= 0.99
        prev_x = cur_x  # Store current x value in prev_x
        cur_x = cur_x - rate * genes_df(A, y_true, prev_x, ngenes=ngenes)  # Grad descent
        previous_step_size = abs(cur_x - prev_x)  # Change in x
        iters = iters + 1  # iteration count
        print("Iteration", iters, "\nX value is", cur_x)  # Print iterations
    print("The local minimum occurs at", cur_x)
    return cur_x


def gene_inference_loss(A, y_true, regu, ngenes=10):
    D = filter_cyclic_genes(A, regu=regu, iterNum=100, verbose=False)
    D = np.identity(D.shape[0]) - D
    res = D.diagonal()
    res = res[A.shape[1] - ngenes * 2:]
    loss = hinge_loss(res, y_true)
    return loss


def genes_df(A, y_true, regu, delta=1e-4, ngenes=10):
    p1 = gene_inference_loss(A, y_true, regu, ngenes)
    p2 = gene_inference_loss(A, y_true, regu + delta, ngenes)
    return (p2 - p1) / delta


def hinge_loss(y_predicted, labels):
    '''
    Parameters
    ----------
    y_predicted
    labels

    Returns
    -------

    '''
    y_predicted = (y_predicted * 2) - 1
    labels = (labels * 2) - 1
    tothinge = 0
    num = len(y_predicted)
    for i in range(num):
        tothinge = tothinge + max(0.0, (1 - y_predicted[i] * labels[i]))
    return tothinge


def e_to_range(E):
    order =[]
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if E[i,j]==1:
                order.append(j)
    return np.array(order)
