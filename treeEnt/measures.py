# ====================================================================================== #
# TreeEnt package.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import numpy as np
import networkx as nx
from toolbox.EntropyEstimates import meanAndStdevEntropyNem as _NSB_entropy
from warnings import warn
from threadpoolctl import threadpool_limits
from multiprocess import Pool


class TreeEntropy():
    def __init__(self, adj, model,
                 sample_size=10_000,
                 cond_sample_size=10_000,
                 mx_cluster_size=14,
                 iprint=True):
        """Use factorization of tree coarse-graining of graph to approximate the
        entropy of a sparse Ising model.

        Parameters
        ----------
        adj : ndarray
            Adjacency graph ignoring diagonal.
        model : coniii.Model
        sample_size : int, 10_000
            MC sample size used to estimate probability distribution over which to
            weight conditional entropies.
        cond_sample_size : int, 10_000
            MC sample size once having conditioned on upper branch to sample from
            leaf.
        mx_cluster_size : int, 14
        iprint : bool, True
        """
        
        assert (adj[np.diag_indices_from(adj)]==0).all()

        # obtain connectivity graph
        self.adj = adj
        self.model = model
        self.sample_size = sample_size
        self.cond_sample_size = cond_sample_size
        self.iprint = iprint
        
        self.setup_graph(mx_cluster_size)
        if self.iprint:
            print("After splitting, subgraphs are of sizes", [len(g) for g in self.subgraphs])
            print()
    
    def setup_graph(self, mx_cluster_size):
        """Wrapper for setting up self.G.
        """
        self.G = nx.Graph(self.adj)
        split_output = self.split_graph(self.G, True, mx_cluster_size)
        self.subgraphs, (self.split_nodes, self.nonsplit_nodes) = split_output

        if any([len(g) > mx_cluster_size for g in self.subgraphs]):
            warn("Some subgraphs are beyond desired threshold size.")
        self.contracted_G, self.node_sets = self.contract_G()
    
    def contract_G(self, G=None, subgraphs=None):
        """Create a "contracted" graph that groups subsets of nodes into a
        coarse-grained graph.

        Parameters
        ----------
        G : nx.Graph, None
            Default is self.G.
        subgraphs : list of lists of ints, None

        Returns
        -------
        list of nx.Graph
        dict
        """

        G = G if not G is None else self.G
        contracted_G = G.copy()
        subgraphs = subgraphs if not subgraphs is None else self.subgraphs

        node_sets = {}  # keep track of which nodes are contracted

        # figure out subgraph connectivity
        for nodes_to_contract in subgraphs:
            nodes_to_contract = list(nodes_to_contract)
            i = nodes_to_contract[0]
            node_sets[i] = nodes_to_contract
            for j in nodes_to_contract[1:]:
                nx.contracted_nodes(contracted_G, i, j, self_loops=False, copy=False)

        try:
            if nx.find_cycle(contracted_G):
                raise Exception("Cycles found in contracted graph.")
        except nx.NetworkXNoCycle:
            pass

        contracted_G = [contracted_G.subgraph(g).copy() for g in nx.connected_components(contracted_G)]
        return contracted_G, node_sets

    @classmethod
    def split_graph(cls, G, return_as_split=False, mx_cluster_size=14):
        """Remove structural holes til graph consists of sufficiently small
        components.
        
        Parameters
        ----------
        G : nx.Graph
        return_as_split : bool, False
            If True, also return graphs groups into split and nonsplit.
        mx_cluster_size : int, 14
        
        Returns
        -------
        list of lists
            Each list is a subgraph into which G can be split by grouping structural
            holes.
        tuple (optional)
            List of nodes divided as splitter and non-splitter nodes.
        """
        
        G_ = G.copy()
        
        removed_node = []
        n_comps = []
        comp_sizes = []

        # try removing node with smallest constraint at a time til the remaining
        # graph is below desired size
        # smaller constraint means structural hole
        mx_size = 1000
        mx_subgraph_size = max([len(g) for g in nx.connected_components(G)])
        largest_subgraph_growing = False
        while mx_size > mx_cluster_size:
            # remove one node at a time from the graph using min constraint heuristic
            cons = cls.constraint(G_)
            toremove = min(cons, key=cons.get)
            G_.remove_node(toremove)
            removed_node.append(toremove)

            comp_sizes.append([])
            counter = 0
            for g in nx.connected_components(G_):
                comp_sizes[-1].append(len(g))
                counter += 1
            n_comps.append(counter)
            mx_size = max(comp_sizes[-1])
            
            # compute subgraphs induced by this removal step to check how graph has
            # changed
            # consider induced subgraphs in the remaining subgraph G_
            nonsplit_nodes = []
            for g in nx.connected_components(G_):
                nonsplit_nodes.append(g)

            # subgraph of divider nodes (those which had been removed)
            split_nodes = []
            for g in nx.connected_components(G.subgraph(removed_node)):
                split_nodes.append(g)

            subgraphs = split_nodes + nonsplit_nodes

            # check if largest subgraph is growing or shrinking in size
            mx_subgraph_size_ = max([len(g) for g in subgraphs])
            if mx_subgraph_size_>mx_subgraph_size:
                largest_subgraph_growing = True
            else:
                mx_subgraph_size = mx_subgraph_size_

        # consider induced subgraphs in the remaining subgraph G_
        nonsplit_nodes = []
        for g in nx.connected_components(G_):
            nonsplit_nodes.append(g)

        # subgraph of divider nodes (those which had been removed)
        split_nodes = []
        for g in nx.connected_components(G.subgraph(removed_node)):
            split_nodes.append(g)
        
        subgraphs = split_nodes + nonsplit_nodes
        subgraphs = [list(i) for i in subgraphs]
        if return_as_split:
            return subgraphs, (split_nodes, nonsplit_nodes)
        return subgraphs
    
    def conditional_entropy(self, G, X, ix_hold, ix_free,
                            fast=False,
                            force=True):
        """For a given graph G, iterate through all possible spin orientations of
        spins in ix_hold while letting the spins in ix_free fluctuate. 
        
        Parameters
        ----------
        G : nx.Graph
        X : ndarray
            Starting sample to estimate probability distribution of held nodes.
        ix_hold : list of int
            Spins to hold fixed. Fixed states taken from X.
        ix_free : list of lists of int
            Different sets of spins from which to sample probability distribution
            from. Remember that fixing the spins in ix_hold can leave several
            independent leaves over which to estimate entropy. These give the
            conditional entropy.
        fast: bool, False
            If True, use naive entropy estimate.
        force : bool, True 
        
        Returns
        -------
        float
            Entropy of held (fixed) spins in bits.
        ndarray
            Entropy of free spins in bits once having conditioned on the fixed spins.
        """
        
        assert set(np.unique(X)) <= frozenset((-1,1))
        ix_hold = sorted(ix_hold)
        ix_free = [sorted(i) for i in ix_free]

        # first, obtain a probability distribution for the hold subset (spins on
        # which we are conditioning); this will constitute weights for each MC sample
        states_hold, p_hold = np.unique(X[:,ix_hold], axis=0, return_counts=True)
        p_hold = p_hold / p_hold.sum()
        n_hold = len(ix_hold)
        S_hold = NSB_estimate(X[:,ix_hold]) if not fast else self.naive_estimate(X[:,ix_hold])
        
        if not force:
            assert p_hold.size < 1e4, f"Too many possible states (K={p_hold.size}). This will take too long."
        elif p_hold.size >= 1e4:
            warn(f"Many possible states (K={p_hold.size}). This will take a while.")

        # second, iterate thru prob distrib from above and sample the free subsets and 
        # their weighted contrib to the entropy
        self.model.sampler.update_parameters(self.model.multipliers)
        def loop_wrapper(args):
            # takes the probability of this particular config, the states of the fixed spins
            p, this_sub = args

            # reset random seed for sampling
            np.random.seed()

            # for sampler, must specify a state of all spins in full graph, but only the ones that are
            # fixed matter
            this = np.ones(self.model.n)
            this[ix_hold] = this_sub

            # sample prob distribution of free spins given held spins
            assert self.model.multipliers.size==((self.model.n+1)*self.model.n//2)
            
            # sample from Ising model while holding this_sub spins fixed at this state
            fixed_subset = list(zip(ix_hold, this[ix_hold]))
            self.model.sampler.generate_cond_sample(self.sample_size, fixed_subset,
                                                      burn_in=self.model.n*1000,
                                                      n_iters=self.model.n*100,
                                                      parallel=False)
            hold_sample = self.model.sampler.sample
            
            # p_free[((cond_states+1)//2).dot(2**np.arange(n))] += p * cond_p
            # iterate over all subsets that remain free upon this conditioning
            S_free = np.zeros((len(ix_free), 3))
            for i, ix in enumerate(ix_free):
                #n_free = len(ix)
                #p_free = np.zeros(2**n_free)
                S_est = (NSB_estimate(hold_sample[:,ix]) if not fast
                                                              else self.naive_estimate(hold_sample[:,ix]))

                # calculate contribution to conditioned entropy, rror, and variance term without weights
                S_free[i] += S_est[0] * p, S_est[1] * p, S_est[0]**2 * p
            return S_free 
        
        S_free = np.zeros((len(ix_free), 3))
        with threadpool_limits(limits=1, user_api='blas'):
            with Pool() as pool:
                for this_S_free in pool.map(loop_wrapper, zip(p_hold, states_hold)):
                    S_free += this_S_free

        return S_hold, S_free

    def _estimate_entropy_component(self, sample, contracted_G,
                                    **conditional_entropy_kw):
        """Estimate entropy of single connected component of graph. See
        self.estimate_entropy().

        Parameters
        ----------
        sample : ndarray
        contracted_G : nx.Graph

        Returns
        -------
        float
            Estimated entropy of center. Uses NSB if center is more than 10 nodes.
        dict
            Estimated entropy of each leaf in clustered graph.
        """
        
        contracted_G = contracted_G.copy()
        
        S_root_ = None  # default value in case contracted_G is of size 1
        S_leaves = {}
        counter = 0
        while len(contracted_G) > 2:
            # first, find leaves
            leaves = [i[0] for i in contracted_G.degree() if i[1]==1]

            # second, find roots of leaves
            roots = []
            for leaf in leaves:
                roots.append(list(contracted_G.neighbors(leaf))[0])
            # and organize by unique roots
            uroots = np.unique(roots).tolist()
            uleaves = []
            for r in uroots:
                uleaves.append([leaf for i, leaf in enumerate(leaves) if roots[i]==r])
            roots = uroots
            leaves = uleaves
            assert len(roots)>0, contracted_G.nodes
            assert len(roots)==len(leaves)

            # third, iterate over all possible states of spins in roots
            # and obtain the conditional entropy of leaves
            for i in range(len(roots)):
                ix_hold = self.node_sets[roots[i]]
                ix_free = [self.node_sets[leaf] for leaf in leaves[i]]

                # read out entropy for each leaf
                S_root_, S_leaves_ = self.conditional_entropy(self.G, sample, ix_hold, ix_free,
                                                              **conditional_entropy_kw)
                for leaf, S_ in zip(leaves[i], S_leaves_):
                    S_leaves[leaf] = S_

            # trim leaves from contracted graph for next recursive step
            for leaves_ in leaves:
                for leaf in leaves_:
                    contracted_G.remove_node(leaf)
            
            if self.iprint: print(f"Done with loop {counter}. {len(contracted_G)-2} remaining.")
            counter += 1

        if len(contracted_G)==2:
            # set root to the the smaller of the two node sets
            if (len(self.node_sets[list(contracted_G.nodes)[0]]) >
                len(self.node_sets[list(contracted_G.nodes)[1]])):
                roots = [list(contracted_G.nodes)[1]]
                leaves = [[list(contracted_G.nodes)[0]]]
            else:
                roots = [list(contracted_G.nodes)[0]]
                leaves = [[list(contracted_G.nodes)[1]]]
            
            # iterate over all possible states of spins in roots
            # and obtain the conditional entropy of leaves
            for i in range(len(roots)):
                ix_hold = self.node_sets[roots[i]]
                ix_free = [self.node_sets[leaf] for leaf in leaves[i]]

                # read out entropy for each leaf
                S_root_, S_leaves_ = self.conditional_entropy(self.G, sample, ix_hold, ix_free,
                                                              **conditional_entropy_kw)
                for leaf, S_ in zip(leaves[i], S_leaves_):
                    S_leaves[leaf] = S_

            # trim leaves from contracted graph
            for leaves_ in leaves:
                for leaf in leaves_:
                    contracted_G.remove_node(leaf)

            if self.iprint: print(f"Done with loop {counter}. {len(contracted_G)-2} remaining.")
            S_root = S_root_ 
        elif len(contracted_G)==1 and S_root_ is None:
            ix_hold = list(contracted_G.nodes)
            S_root = NSB_estimate(sample[:,ix_hold])
            #counts = np.unique(sample[:,ix_hold], axis=0, return_counts=True)[1]
            #p = counts / counts.sum()
            #S_root = -(p*np.log(p)).sum()
        else:
            S_root = S_root_ 

        return S_root, S_leaves
    
    def naive_estimate(self, X):
        """Naive entropy estimate from counting.

        Parameters
        ----------
        X : ndarray

        Returns
        -------
        float
            Entropy in bits.
        float
            Error in bits (currently not implemented).
        """
        assert X.ndim==2
        counts = np.unique(X, axis=0, return_counts=True)[1]
        p = counts / counts.sum()
        return -p.dot(np.log2(p)), 0.

    def estimate_entropy(self, 
                         **conditional_entropy_kw):
        """Estimate entropy of whole graph.
        
        Algorithm as follows for each connected component as is listed in self.contracted_G:
        - Calculate entropy starting with leaves and their corresponding roots
          after each calculation prune the leaves such that the roots become the new
          leaves
        - Continue til there are no more leaves to prune, then the entropy of that
          center is the only entropy that we need to estimate without any conditioning

        Parameters
        ----------
        Returns
        -------
        list of float
            Estimated entropy of center, or root. One element for each
            component. NSB estimate unless not available.
        list of dict
            Estimated entropy of each leaf in clustered graph.
        """
        
        # Generate large sample from Ising model
        n = self.model.n
        self.model.generate_sample(n*100, n*1000,
                                          multipliers=self.model.multipliers,
                                          sample_size=self.sample_size)
        sample = self.model.sample
        
        # iterate thru each connected component and compute its entropy
        self.S_root = []
        self.S_leaves = []
        for i, g in enumerate(self.contracted_G):
            if self.iprint: print(f"Starting on component {i+1} out of {len(self.contracted_G)}.")
            output = self._estimate_entropy_component(sample, g, **conditional_entropy_kw)
            self.S_root.append(output[0])
            self.S_leaves.append(output[1])
            if self.iprint: print()

        return [self.S_root[i][0] for i in range(len(self.S_root))], self.S_leaves

    def entropy(self):
        """Estimated entropy in bits using pre-computed values.

        To run full sampling algorithm to cache value, must use .estimate_entropy().
        
        Returns
        -------
        float
            Estimated entropy in bits.
        float
            Estimated NSB error. Doesn't account for errors in the conditional
            probability weights.
        """

        assert 'S_root' in self.__dict__.keys()
        
        S = 0
        nsb_err = 0
        std_err = 0
        for s1 in self.S_root:
            S += s1[0]
            nsb_err += s1[1]**2
        S += sum([sum(list(zip(*list(i.values())))[0]) for i in self.S_leaves if len(i)])
        # sum over the NSB estimate errors for each leaf then sum over leaves
        nsb_err += sum([sum(np.array(list(zip(*list(i.values())))[1])**2) for i in self.S_leaves if len(i)])

        # sum over std estimated from conditioned sets of spins; we have already saved the
        # squared and weighted values of the entropies from the leaves so we must
        # subtract the squared means
        std_err = sum([sum([j[2]-j[0]**2 for j in i.values()])
                            for i in self.S_leaves if len(i)]) / self.sample_size
        
        if std_err<0 and np.isclose(std_err, 0):
            std_err = 0

        return S, np.sqrt(nsb_err + std_err)

    def naive_entropy(self, sample_size=1_000_000, force_resample=False):
        """Estimate naive entropy.
        
        Parameters
        ----------
        sample_size : int, 1_000_000
        force_resample : bool, False

        Returns
        -------
        float
            Entropy in bits.
        """
        
        model = self.model
        n = self.model.n

        if model.sample.shape[0]<sample_size or force_resample:
            model.generate_sample(sample_size=sample_size, n_iters=n*100, burn_in=n*1000)

        p = np.unique(model.sample[:sample_size], axis=0, return_counts=True)[1]
        p = p/p.sum()

        return -p.dot(np.log2(p))

    @staticmethod
    def constraint(G):
        """Burt's structural constraint for each node in G.
        
        Parameters
        ----------
        G : nx.Graph
        
        Returns
        -------
        dict
        """
        # calculate local constraint for every node
        lc = {}
        for node in G.nodes():
            for n_node in G.neighbors(node):
                indirect = sum([1/len(G.adj[nn_node])
                                for nn_node in nx.common_neighbors(G, node, n_node)])
                lc[(node,n_node)] = (1 + indirect)**2 / len(G.adj[node])**2

        # calculate total constraint
        c = {}
        for node in G.nodes():
            terms = [lc[(node,n_node)] for n_node in G.neighbors(node)]
            if len(terms):
                c[node] = sum(terms)
            else:
                c[node] = np.nan
        return c
#end TreeEntropy



def NSB_estimate(X):
    """Wrapper for NSB estimator.

    Parameters
    ----------
    X : ndarray
        Shape is (no. of data points, no. of spins).

    Returns
    -------
    float
        Entropy estimate in bits.
    float
        Error.
    """
    assert X.ndim==2
    counts = np.unique(X, axis=0, return_counts=True)[1]

    # assuming sample space is binary
    # this causes an exception case of two observed states so skip K argument in that case
    if counts.size==2:
        return _NSB_entropy(counts, bits=True)
    return _NSB_entropy(counts, bits=True, K=2**X.shape[1])
