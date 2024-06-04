import os.path as osp
import argparse
import torch

BLK_H = 16
BLK_W = 8
import DTCSpMM
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='YeastH', help="dataset")
args = parser.parse_args()
print(args)

## Load matrix from files.
dataset = args.dataset
# Set your own path to the dataset.
path = osp.join("/mnt/raid/fanruibo/g_dataset/", dataset + ".npz") #4090
matrix = DTC_dataset(path)
num_rows = matrix.num_nodes
num_nnz = matrix.num_edges
print("M, E: ", num_rows, " " , num_nnz)
column_index =  matrix.column_index 
row_pointers = matrix.row_pointers

#########################################
## Compute TC-GNN related graph MetaData.
#########################################
num_row_windows = (num_rows + BLK_H - 1) // BLK_H
edgeToColumn = torch.zeros(num_nnz, dtype=torch.int)
edgeToRow = torch.zeros(num_nnz, dtype=torch.int)
blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
column_index_ori  = column_index.cuda()
row_pointers_ori = row_pointers.cuda()

blockPartition_cuda  = blockPartition.cuda()
edgeToColumn_cuda = edgeToColumn.cuda()
edgeToRow_cuda  = edgeToRow.cuda()

#ORIGIN TCGNN
DTCSpMM.preprocess(column_index, row_pointers, num_rows,  \
                BLK_H,	BLK_W, blockPartition, edgeToColumn, edgeToRow)
blockPartition_ori  = blockPartition.cuda()
edgeToColumn_ori  = edgeToColumn.cuda()
edgeToRow_ori  = edgeToRow.cuda()

# Run tests.
for feat_size in [128, 256, 512]:
    print("feat_size =", feat_size)
    X = torch.ones((num_rows, feat_size)).cuda()
    # Run test.
    with open("./TCGNNSpMM_exe_time_and_throughput.csv", 'a') as f:
        f.write(dataset + "," + str(feat_size)  + ",")
    out = DTCSpMM.run_TCGNNSpMM(X, row_pointers_ori, column_index_ori,\
                blockPartition_ori, edgeToColumn_ori, edgeToRow_ori)
    with open("./TCGNNSpMM_exe_time_and_throughput.csv", 'a') as f:
        f.write("\n")