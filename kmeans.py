"""
    This is a file you will have to fill in.
    It contains helper functions required by K-means method via iterative improvement
"""
import numpy as np
from random import sample

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids
    Hint: use random.sample (it is already imported for you!)
    :param k: number of cluster centroids
    :param inputs: a 2D Numpy array, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    return np.array(sample(list(inputs), k))


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance
    :param inputs: inputs of data, a 2D Numpy array
    :param centroids: a Numpy array of k current centroids
    :return: a Numpy array of centroid indices, one for each row of the inputs
    """
    # TODO
    indices = []
    for input in inputs:
        l = None
        m = None
        for n, cent in enumerate(centroids):
            if l == None or dist(input, cent) < l:
                l = dist(input, cent)
                m = n
        indices.append(m)
    return np.array(indices)



def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster
    :param inputs: inputs of data, a 2D Numpy array
    :param indices: a Numpy array of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    mean = [np.mean(inputs[indices==n], axis=0) for n in range(k)]
    return np.array(mean)


def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    :param inputs: inputs of data, a 2D Numpy array
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: the tolerance we determine convergence with when compared to the ratio as stated on handout
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    pcenter = init_centroids(k, inputs)
    i = assign_step(inputs, pcenter)
    count = 0
    while (count < max_iter):
        ncenter = update_step(inputs, i, k)
        if np.linalg.norm(pcenter-ncenter)/np.linalg.norm(pcenter) < tol:
            return ncenter
        else:
            pcenter = ncenter
            i = assign_step(inputs, pcenter)
            count += 1
    return pcenter

def dist(x, y):
    return np.sqrt(sum([(yi-xi)**2 for xi, yi in zip(x,y)]))