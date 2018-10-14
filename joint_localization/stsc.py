"""
Zelnik-Manor, L., & Perona, P. (2005). Self-tuning spectral clustering.
In Advances in neural information processing systems (pp. 1601-1608).
Original Paper: https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
"""
from itertools import groupby

import numpy as np
from functools import reduce

from scipy.linalg import inv, sqrtm, eigh
from scipy.optimize import minimize


def generate_Givens_rotation(i, j, theta, size):
    g = np.eye(size)
    c = np.cos(theta)
    s = np.sin(theta)
    g[i, i] = c
    g[j, j] = c
    if i > j:
        g[j, i] = -s
        g[i, j] = s
    elif i < j:
        g[j, i] = s
        g[i, j] = -s
    else:
        raise ValueError('i and j must be different')
    return g


def generate_Givens_rotation_gradient(i, j, theta, size):
    g = np.zeros((size, size))
    c = np.cos(theta)
    s = np.sin(theta)
    g[i, i] = -s
    g[j, j] = -s
    if i > j:
        g[j, i] = -c
        g[i, j] = c
    elif i < j:
        g[j, i] = c
        g[i, j] = -c
    else:
        raise ValueError('i and j must be different')
    return g


def generate_U_list(ij_list, theta_list, size):
    return [generate_Givens_rotation(ij[0], ij[1], theta, size)
            for ij, theta in zip(ij_list, theta_list)]


def generate_V_list(ij_list, theta_list, size):
    return [generate_Givens_rotation_gradient(ij[0], ij[1], theta, size)
            for ij, theta in zip(ij_list, theta_list)]


def get_U_ab(a, b, U_list, K):
    I = np.eye(U_list[0].shape[0])
    if a == b:
        if a < K and a != 0:
            return U_list[a]
        else:
            return I
    elif a > b:
        return I
    else:
        return reduce(np.dot, U_list[a:b], I)


def get_A_matrix(X, U_list, V_list, k, K):
    Ul = get_U_ab(0, k, U_list, K)
    V = V_list[k]
    Ur = get_U_ab(k + 1, K, U_list, K)
    return X.dot(Ul).dot(V).dot(Ur)


def get_rotation_matrix(X, C):
    ij_list = [(i, j) for i in range(C) for j in range(C) if i < j]
    K = len(ij_list)

    def cost_and_grad(theta_list):
        U_list = generate_U_list(ij_list, theta_list, C)
        V_list = generate_V_list(ij_list, theta_list, C)
        R = reduce(np.dot, U_list, np.eye(C))
        Z = X.dot(R)
        mi = np.argmax(Z, axis=1)
        M = np.choose(mi, Z.T).reshape(-1, 1)
        cost = np.sum((Z / M) ** 2)
        grad = np.zeros(K)
        for k in range(K):
            A = get_A_matrix(X, U_list, V_list, k, K)
            tmp = (Z / (M ** 2)) * A
            tmp -= ((Z ** 2) / (M ** 3)) * (np.choose(mi, A.T).reshape(-1, 1))
            tmp = 2 * np.sum(tmp)
            grad[k] = tmp

        return cost, grad

    theta_list_init = np.array([0.0] * int(C * (C - 1) / 2))
    opt = minimize(cost_and_grad,
                   x0=theta_list_init,
                   method='CG',
                   jac=True,
                   options={'disp': False})
    return opt.fun, reduce(np.dot, generate_U_list(ij_list, opt.x, C), np.eye(C))


def reformat_result(cluster_labels, n):
    zipped_data = zip(cluster_labels, range(n))
    zipped_data = sorted(zipped_data, key=lambda x: x[0])
    grouped_feature_id = [[j[1] for j in i[1]] for i in groupby(zipped_data, lambda x: x[0])]
    return grouped_feature_id


def affinity_to_lap_to_eig(affinity):
    tril = np.tril(affinity, k=-1)
    a = tril + tril.T
    d = np.diag(a.sum(axis=0))
    dd = inv(sqrtm(d))
    l = dd.dot(a).dot(dd)
    w, v = eigh(l)
    return w, v


def get_min_max(w, min_n_cluster, max_n_cluster):
    if min_n_cluster is None:
        min_n_cluster = 2
    if max_n_cluster is None:
        max_n_cluster = np.sum(w > 0)
        if max_n_cluster < 2:
            max_n_cluster = 2
    if min_n_cluster > max_n_cluster:
        raise ValueError('min_n_cluster should be smaller than max_n_cluster')
    return min_n_cluster, max_n_cluster


def self_tuning_spectral_clustering(affinity, min_n_cluster=None, max_n_cluster=None):
    w, v = affinity_to_lap_to_eig(affinity)
    min_n_cluster, max_n_cluster = get_min_max(w, min_n_cluster, max_n_cluster)
    re = []
    for c in range(min_n_cluster, max_n_cluster + 1):
        x = v[:, -c:]
        cost, r = get_rotation_matrix(x, c)
        re.append((cost, x.dot(r)))
        #print('n_cluster: %d \t cost: %f' % (c, cost))
    COST, Z = sorted(re, key=lambda x: x[0])[0]
    return reformat_result(np.argmax(Z, axis=1), Z.shape[0])
