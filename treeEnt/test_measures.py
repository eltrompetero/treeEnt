# Testing
# Author: Eddie Lee, edlee@csh.ac.at

from .measures import *
from coniii.models import Ising
from .NSB_toolbox import meanAndStdevEntropyNem as NSB_entropy



def test_constraint(G):
    """For validating custom local constraint and constraint calculation with
    networkx code.
    """
    # calculate local constraint for every node
    lc = {}
    for node in G.nodes():
        for n_node in G.neighbors(node):
            indirect = sum([1/len(G.adj[nn_node])
                            for nn_node in nx.common_neighbors(G, node, n_node)])
            lc[(node,n_node)] = (1 + indirect)**2 / len(G.adj[node])**2
            assert np.isclose(lc[(node,n_node)], nx.local_constraint(G, node, n_node))

    # calculate total constraint
    nx_c = nx.constraint(G)
    c = {}
    for node in G.nodes():
        terms = [lc[(node,n_node)] for n_node in G.neighbors(node)]
        if len(terms):
            c[node] = sum(terms)
            assert np.isclose(c[node], nx_c[node])
        else:
            c[node] = np.nan
    return c, nx_c

def test_weighted_constraint(G=None):
    np.random.seed(0)

    if G is None:
        G = nx.random_graphs.fast_gnp_random_graph(10, .2, 0)

    for edge in G.edges:
        G.add_edge(*edge, weight=np.random.rand())

    my_con = TreeEntropy.constraint(G, weights='weight')
    nx_con = nx.constraint(G, weight='weight')

    for k in my_con.keys():
        if not np.isnan(my_con[k]):
            assert np.isclose(my_con[k], nx_con[k]), k

def test_hairy_triangle_and_point(J_scale=.5):
    """Setup a model instance with triangle and outgoing nodes and a single
    independent point.

    Parameters
    ----------
    J_scale : float, .5

    Returns
    -------
    tuple
    float
    tuple
    """
    n = 7
    
    model = Ising(n)
    model.setup_sampler()
    multipliers = np.array([0,0,0,0,0,0,0,
                              1,1,1,0,0,0,
                                1,0,1,0,0,
                                  0,0,1,0,
                                    0,0,0,
                                      0,0,
                                        0]) * J_scale * 1.
    model.set_multipliers(multipliers)
    
    adj = np.zeros((n,n), dtype=int)
    adj[np.where(squareform(model.multipliers[n:])!=0)] = 1

    entropyEstimator = TreeEntropy(adj, model,
                                   sample_size=10_000,
                                   cond_sample_size=1_000,
                                   iprint=False)
    entropyEstimator.estimate_entropy()

    S_naive = entropyEstimator.naive_entropy(10_000)
    counts = np.unique(entropyEstimator.model.sample, axis=0, return_counts=True)[1]

    return (entropyEstimator.entropy(),
            S_naive,
            NSB_entropy(counts, bits=True, K=2**model.n))

def test_two_hairy_triangles(J_scale=.5):  
    """Setup a model instance with triangle and outgoing nodes."""
    n = 12
    
    Jmat = np.zeros((n, n))
    Jmat[0,1] = 1
    Jmat[0,2] = 1
    Jmat[0,3] = 1
    Jmat[1,2] = 1
    Jmat[1,4] = 1
    Jmat[2,5] = 1
    Jmat[6:,6:] = Jmat[:6,:6]

    model = Ising(n)
    model.setup_sampler()
    multipliers = np.array([0]*n + Jmat[np.triu_indices_from(Jmat, k=1)].tolist()) * J_scale * 1.
    model.set_multipliers(multipliers)
    
    adj = np.zeros((n,n), dtype=int)
    adj[np.where(Jmat)] = 1
    
    entropyEstimator = TreeEntropy(adj, model,
                                   sample_size=10_000,
                                   cond_sample_size=1_000,
                                   iprint=False,
                                   mx_cluster_size=3)

    entropyEstimator.estimate_entropy()
    entropyEstimator.entropy()

    S_naive = entropyEstimator.naive_entropy(10_000)
    counts = np.unique(entropyEstimator.model.sample, axis=0, return_counts=True)[1]

    return (entropyEstimator.entropy(),
            S_naive,
            NSB_entropy(counts, bits=True, K=2**model.n))

def test_n_hairy_triangles(n_tri, J_scale=.5, chain=False, **treeEnt_kw):  
    """Setup a model instance with triangle and outgoing nodes."""
    assert n_tri>=1
    n = n_tri * 6
    
    # set up first triangle and then replicate it
    Jmat = np.zeros((n, n))
    Jmat[0,1] = 1
    Jmat[0,2] = 1
    Jmat[0,3] = 1
    Jmat[1,2] = 1
    Jmat[1,4] = 1
    Jmat[2,5] = 1
    for i in range(1, n_tri):
        Jmat[6*i:6*(i+1),6*i:6*(i+1)] = Jmat[:6,:6]
        if chain:
            # connect adjacent triangles into a chain
            Jmat[6*i-1,6*i] = Jmat[6*i,6*i-1] = 1 
    
    model = Ising(n)
    model.setup_sampler()
    multipliers = np.array([0]*n + Jmat[np.triu_indices_from(Jmat, k=1)].tolist()) * J_scale * 1.
    model.set_multipliers(multipliers)
    
    adj = np.zeros((n,n), dtype=int)
    adj[np.where(Jmat)] = 1
    
    entropyEstimator = TreeEntropy(adj, model,
                                   iprint=False,
                                   mx_cluster_size=3,
                                   **treeEnt_kw)
    
    entropyEstimator.estimate_entropy()
    entropyEstimator.entropy()

    S_naive = entropyEstimator.naive_entropy(10_000)
    counts = np.unique(entropyEstimator.model.sample, axis=0, return_counts=True)[1]

    return (entropyEstimator.entropy(),
            S_naive,
            NSB_entropy(counts, bits=True, K=2**model.n))

def exact_hairy_triangle_entropy(J_scale):
    """Exact entropy calculation of the hairy triangle using enumeration of all
    possible configurations.
    
    Parameters
    ----------
    J_scale : float
        Scale on couplings, i.e. inverse temperature.
        
    Returns
    -------
    float
        Entropy in bits.
    """
    n = 6

    # set up first triangle and then replicate it
    Jmat = np.zeros((n, n))
    Jmat[0,1] = 1
    Jmat[0,2] = 1
    Jmat[0,3] = 1
    Jmat[1,2] = 1
    Jmat[1,4] = 1
    Jmat[2,5] = 1

    model = Ising(np.array([0]*n + Jmat[np.triu_indices_from(Jmat, k=1)].tolist()) * J_scale * 1.)
    p = model.calc_p()

    return -p.dot(np.log2(p))

if __name__=='__main__':
    for seed in range(20):
        G = nx.fast_gnp_random_graph(60, .1, seed=seed)
        test_constraint(G)

