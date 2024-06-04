# import dgl
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import argparse
import numpy as np
import torch as th
from tvm.script import tir as T
from tvm.sparse import FormatRewriteRule, lower_sparse_buffer, lower_sparse_iter
import tvm.sparse

from dataset import *
import os.path as osp

@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    num_tiles: T.int32,
    nnz: T.int32,
    cwm: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K1 = T.dense_fixed(num_tiles)
    K2 = T.dense_fixed(cwm)
    K3 = T.dense_fixed(32)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float32")
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float32")
    with T.sp_iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
        with T.init():
            C[i, k1, k2, k3] = T.float32(0)
        C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]

class TorchOpTimer(object):
    def __enter__(self):
        self.start_event = th.cuda.Event(enable_timing=True)
        self.end_event = th.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        th.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event) / 1e3

def bench_hyb(
    indptr,
    indices,
    m,
    n,
    nnz,
    x,
    y_golden,
    feat_size=128,
    cwm=2,
):
    if feat_size < 64:
        cwm = 1
    mod = tvm.IRModule.from_expr(csrmm)
    # specialize
    params = mod["main"].params
    print(params)
    param_map = {
        params[5]: m,  # m
        params[6]: m,  # n
        params[7]: feat_size // cwm // 32,  # num_tiles,
        params[8]: nnz,  # nnz
        params[9]: cwm,  # cwm
    }
    print(param_map)
    mod["main"] = mod["main"].specialize(param_map)

    # schedule
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    outer_blk = sch.get_block("csrmm0")
    inner_blk = sch.get_block("csrmm1")
    (i,) = sch.get_loops(outer_blk)
    j, foo, foi, fi = sch.get_loops(inner_blk)
    sch.reorder(foo, fi, j, foi)
    sch.bind(fi, "threadIdx.x")
    sch.bind(foo, "blockIdx.y")
    sch.unroll(foi)
    io, ii = sch.split(i, [None, 8])
    sch.bind(io, "blockIdx.x")
    sch.bind(ii, "threadIdx.y")
    init_blk = sch.decompose_reduction(inner_blk, fi)
    ax0, ax1 = sch.get_loops(init_blk)[-2:]
    sch.bind(ax0, "threadIdx.x")
    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    f = tvm.build(mod["main"], target="cuda")
    # prepare nd array
    indptr_nd = tvm.nd.array(indptr.numpy().astype("int32"), device=tvm.cuda(0))
    b_nd = tvm.nd.array(
        x.numpy().reshape(-1).astype("float32"),
        device=tvm.cuda(0),
    )
    indices_nd = tvm.nd.array(indices.numpy().astype("int32"), device=tvm.cuda(0))
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float32"), device=tvm.cuda(0))
    a_nd = tvm.nd.array(np.ones((nnz,)).astype("float32"), device=tvm.cuda(0))
    args = [a_nd, b_nd, c_nd, indptr_nd, indices_nd]
    f(*args)
    # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)
    evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=1000)
    time = (evaluator(*args).mean * 1000).item()
    print("tir naive time: {:.5f} ms".format(time))
    throughput = (nnz * feat_size * 2) / time / 1e6
    with open("./SparseTIR_exe_time_and_throughput.csv", 'a') as f:
        f.write(dset + "," + str(feat_size) +"," + str(time) +","+ str(throughput)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, default="YeastH", help="dataset name")
    args = parser.parse_args()
    dataset = args.dataset
    dset = dataset
    path = osp.join("/mnt/raid/fanruibo/g_dataset/", dataset + ".npz") #4090
    dataset = DTC_dataset(path)
    name = args.dataset
    num_nodes = dataset.num_nodes.item()
    num_edges = dataset.num_edges
    print("M, E: ", num_nodes, " " , num_edges)
    column_index =  dataset.column_index
    row_pointers = dataset.row_pointers
    print(column_index)
    print(type(num_nodes))
    for feat_size in [128, 256, 512]:
        print("feat_size =", feat_size)
        x = th.rand((num_nodes, feat_size))
        bench_hyb(
            row_pointers,
            column_index,
            num_nodes,
            num_nodes,
            num_edges,
            x,
            x,
            feat_size=feat_size,
            cwm=2,
        )