### We reuse the code from TC-GNN ()
#!/usr/bin/env python3
import torch
import numpy as np
import time
from scipy.sparse import *
torch.manual_seed(0)
class DTC_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, path, verbose=False):
        super(DTC_dataset, self).__init__()
        self.nodes = set()
        self.num_nodes = 0
        self.edge_index = None
        self.verbose_flag = verbose
        self.init_sparse(path)
    
    def init_sparse(self, path):
        if not path.endswith('.npz'):
            raise ValueError("graph file must be a .npz file")
        start = time.perf_counter()
        graph_obj = np.load(path)
        src_li = graph_obj['src_li']
        dst_li = graph_obj['dst_li']
        self.num_nodes = graph_obj['num_nodes']
        self.num_edges = len(src_li)
        self.edge_index = np.stack([src_li, dst_li])
        dur = time.perf_counter() - start
        if self.verbose_flag:
            print("# Loading (npz)(s): {:.3f}".format(dur))
            print('# nodes: {}'.format(self.num_nodes))
        # Build CSR.
        val = [1] * self.num_edges
        start = time.perf_counter()
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        scipy_csr = scipy_coo.tocsr()
        build_csr = time.perf_counter() - start
        if self.verbose_flag:
            print("# Build CSR (s): {:.3f}".format(build_csr))
        self.column_index = torch.IntTensor(scipy_csr.indices)
        self.row_pointers = torch.IntTensor(scipy_csr.indptr)
