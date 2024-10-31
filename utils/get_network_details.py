import numpy as np
import networkx as nx
from numpy import linalg as LA


def plot_degree_distribution(G,threshold,ax):
    G_unweighted = nx.Graph() 
    for u, v, data in G.edges(data=True):
        if data['weight'] >= threshold:
            G_unweighted.add_edge(u, v)
    degrees = [degree for node, degree in G_unweighted.degree()]
    degree_counts = np.bincount(degrees)
    degree_range = np.arange(len(degree_counts))
    ax.bar(degree_range, degree_counts, color='blue')
    return  G_unweighted, ax


def get_unweighted(G,threshold):
    G_unweighted = nx.Graph()  
    for u, v, data in G.edges(data=True):
        if data['weight'] >= threshold:
            G_unweighted.add_edge(u, v)
    return G_unweighted


def compute_gamma(degree_sequence, k_min=1):
    """
    Compute the power-law exponent (gamma) using Maximum Likelihood Estimation (MLE).
    Parameters:
    - degree_sequence: List or array of node degrees.
    - k_min: The minimum degree for which the power-law behavior is assumed to hold.
    Returns:
    - gamma: The estimated value of the power-law exponent.
    """
    filtered_degrees = [k for k in degree_sequence if k >= k_min]
    n = len(filtered_degrees)
    if n == 0:
        return None
    sum_log = np.sum(np.log(np.array(filtered_degrees) / k_min))
    # Estimate gamma using the MLE formula
    gamma = 1 + n / sum_log
    return gamma

def compute_network_distribution_property(G_weighted,G,adj_matrix):
    #Heterogeneity: Plot the heterogeneity vs num of layer
    #Gamma: Plot
    #Gamma without hub: Plot 
    #Variance without the hub: Plot
    degs = np.array([d for _, d in G.degree()])
    k1 = degs.mean()
    k2 = np.mean(degs** 2)
    if k1 == 0:
        return None, None
    heterogenity = (k2 -  k1**2)/k1
    gamma = compute_gamma(degs, k_min=1) 
    eigenvalues, _ = LA.eig(adj_matrix)

    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian_matrix = degree_matrix - adj_matrix
    lap_eigenvalues, _ = LA.eig(laplacian_matrix) 
    return G_weighted.degree(weight='weight'),G.degree(),heterogenity, gamma, eigenvalues.tolist(), lap_eigenvalues.tolist()