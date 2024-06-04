# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from tvm.script import tir as T

@T.prim_func
def ell(
    a: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(1)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
    A = T.match_sparse_buffer(a, (O, I, J), "float32")
    T.evaluate(0)


import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import argparse
import numpy as np
import torch as th
from tvm.script import tir as T
from tvm.sparse import (
    FormatRewriteRule,
    lower_sparse_buffer,
    lower_sparse_iter,
    column_part_hyb,
    format_decompose,
)
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
            C[i, k1, k2, k3] = 0.0
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


def csr2ell_inv_index_map(o, i, j):
    return i, j


def csr2ell_index_map(i, j):
    return 0, i, j


cached_bucketing_format = None


def bench_hyb(
    indptr,
    indices,
    m,
    n,
    nnz,
    x,
    y_golden,
    feat_size=128,
    bucket_sizes=[],
    coersening_factor=2,
    num_col_parts=1,
    use_implicit_unroll=False,
):
    num_buckets = len(bucket_sizes)
    coersening_factor = min(coersening_factor, feat_size // 32)
    global cached_bucketing_format
    if cached_bucketing_format is None:
        indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
        indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
        cached_bucketing_format = column_part_hyb(
            m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
        )
    row_indices, col_indices, mask = cached_bucketing_format

    # rewrite csrmm
    nnz_cols_symbol = ell.params[-1]
    rewrites = []
    for part_id in range(num_col_parts):
        for bucket_id, bucket_size in enumerate(bucket_sizes):
            rewrites.append(
                FormatRewriteRule(
                    str(part_id) + "_" + str(bucket_id),
                    ell.specialize({nnz_cols_symbol: bucket_size}),
                    ["A"],
                    ["I", "J"],
                    ["O", "I", "J"],
                    {"I": ["O", "I"], "J": ["J"]},
                    csr2ell_index_map,
                    csr2ell_inv_index_map,
                )
            )
    mod = tvm.IRModule.from_expr(csrmm)
    mod = format_decompose(mod, rewrites)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    # specialize
    params = mod["main"].params
    param_map = {
        params[5]: m,  # m
        params[6]: n,  # n
        params[7]: feat_size // coersening_factor // 32,  # num_tiles,
        params[8]: nnz,  # nnz
        params[9]: coersening_factor,  # coersening_factor
    }
    print(param_map)
    for part_id in range(num_col_parts):
        for bucket_id in range(num_buckets):
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 4]] = m
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 5]] = n
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 6]] = row_indices[
                part_id
            ][bucket_id].shape[0]

    mod["main"] = mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)

    # schedule
    sch = tvm.tir.Schedule(mod)
    for sp_iter_name in [
        "csrmm_{}_{}".format(i, j) for j in range(num_buckets) for i in range(num_col_parts)
    ]:
        sp_iteration = sch.get_sparse_iteration(sp_iter_name)
        o, i, j, k1, k2, k3 = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [o, i])

    mod = sch.mod
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    for part_id in range(num_col_parts):
        for bucket_id, bucket_size in enumerate(bucket_sizes):
            is_atomic = num_col_parts > 1 or bucket_id + 1 == num_buckets
            blk = sch.get_block("csrmm_{}_{}0".format(part_id, bucket_id))
            i, j, foo, foi, fi = sch.get_loops(blk)
            sch.reorder(foo, fi, j, foi)
            if is_atomic:
                sch.annotate(blk, "atomic", True)
                write_blk = sch.reverse_cache_write(blk, 0, "local")
                sch.reverse_compute_at(write_blk, fi, True)
                # sch.unroll(sch.get_loops(write_blk)[-2])
            sch.bind(fi, "threadIdx.x")
            sch.bind(foo, "blockIdx.y")
            sch.unroll(foi)
            if use_implicit_unroll:
                sch.annotate(foi, "pragma_unroll_explicit", 0)
            sch.unroll(j)
            if use_implicit_unroll:
                sch.annotate(j, "pragma_unroll_explicit", 0)
            io, ioi, ii = sch.split(i, [None, bucket_sizes[-1] // bucket_size, 8])
            sch.bind(io, "blockIdx.x")
            sch.bind(ii, "threadIdx.y")
            init_blk = sch.decompose_reduction(blk, fi)
            ax0, ax1 = sch.get_loops(init_blk)[-2:]
            sch.bind(ax0, "threadIdx.x")
            sch.unroll(ax1)
            if use_implicit_unroll:
                sch.annotate(ax1, "pragma_unroll_explicit", 0)

    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    f = tvm.build(mod, target="cuda")

    # prepare nd array
    b_nd = tvm.nd.array(
        x.numpy().reshape(-1).astype("float32"),
        device=tvm.cuda(0),
    )
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float32"), device=tvm.cuda(0))
    # prepare args
    args = [b_nd, c_nd]

    for part_id in range(num_col_parts):
        for bucket_id, _ in enumerate(bucket_sizes):
            weight = tvm.nd.array(
                mask[part_id][bucket_id].numpy().reshape(-1).astype("float32"), device=tvm.cuda(0)
            )
            rows = tvm.nd.array(
                row_indices[part_id][bucket_id].numpy().astype("int32"), device=tvm.cuda(0)
            )
            cols = tvm.nd.array(
                col_indices[part_id][bucket_id].numpy().reshape(-1).astype("int32"),
                device=tvm.cuda(0),
            )
            args += [weight, rows, cols]

    # test accuracy
    f(*args)
    # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)

    # evaluate time
    evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=1000)
    print("tir hyb time: {:.5f}ms".format(evaluator(*args).mean * 1000))
    time = (evaluator(*args).mean * 1000).item()
    throughput = (nnz * feat_size * 2) / time / 1e6
    with open("./tcgnn_exe_time_and_throughput_sparsetir_normal_hyb.csv", 'a') as f:
        f.write(dset + "," + str(feat_size) +"," + str(time) +","+ str(throughput)+"\n")    


col_part_config = {
    "protein": 8,
    "YeastH": 1,
    "reddit": 8,
    "OVCAR-8H": 1,
    "Yeast": 1,
    "DD": 1,
    "ddi": 1,
    "web-BerkStan": 1,
}
bucketing_config = {
    "protein": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "YeastH": [1, 2, 4],
    "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    "OVCAR-8H": [1, 2, 4],
    "Yeast": [1, 2, 4],
    "DD": [1, 2, 4],
    "ddi": [1, 2, 4],
    "web-BerkStan": [1, 2, 4]
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, default="YeastH", help="dataset name")
    parser.add_argument("--implicit-unroll", "-i", action="store_true", help="use implicit unroll")

    args = parser.parse_args()
    dataset = args.dataset
    dset = dataset
    path = osp.join("/mnt/raid/fanruibo/g_dataset/", dataset + ".npz") #4090
    dataset = DTC_dataset(path)
    substring_to_remove = ".reorder"
    name = args.dataset.replace(substring_to_remove, "")
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
            bucket_sizes=bucketing_config[name],
            coersening_factor=2,
            num_col_parts=col_part_config[name],
            use_implicit_unroll=args.implicit_unroll,
        )
