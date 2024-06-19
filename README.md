# DTC-SpMM: Accelerating General Sparse Matrix Multiplication with Tensor Cores

This project contains the codes for **DTC-SpMM**, a recent work aimed at enhancing the performance of general-purpose Sparse Matrix-Matrix Multiplication (SpMM) on GPUs equipped with Tensor Cores. This work has been accepted for presentation at **ASPLOS'24**.

- **If you find this work useful, please cite this project and our [paper](https://dl.acm.org/doi/abs/10.1145/3620666.3651378)**.

  ```tex
  @inproceedings{fan2024dtc,
    title={DTC-SpMM: Bridging the Gap in Accelerating General Sparse Matrix Multiplication with Tensor Cores},
    author={Fan, Ruibo and Wang, Wei and Chu, Xiaowen},
    booktitle={Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3},
    pages={253--267},
    year={2024}
  }
  ```

  

## 1. Prepare your environment

```shell
# RTX 4090 (prefer) or 3090 with CUDA 12.1 installed.
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME="/usr/local/cuda-12.1/"
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Create and activate virtual env
conda create -n DTCSpMM python=3.9
conda activate DTCSpMM

# install PyTorch (must)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# install cmake, numpy and scipy (must)
conda install cmake
pip install numpy
pip install scipy

# install cugraph (Optional for TCA-reordering)
pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com

# install datasketch (Optional for TCA-reordering)
pip install datasketch

# install cupy (Optional for TCA-reordering)
pip install cupy-cuda12x

# install cudf (Optional for TCA-reordering)
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12
```

## 2. Clone DTC-SpMM

```shell
git clone --recursive git@github.com:fan1997/DTC-SpMM-ASPLOS24.git
cd DTC-SpMM-ASPLOS24 && source init_dtc.sh
```

## 3. Prepare Sputnik (dependency)

```shell
cd third_party/
source ./build_sputnik.sh
```

## 4. Build DTC-SpMM

```shell
cd ${DTC_HOME}/DTC-SpMM && source build.sh
```

## 5. Download datasets

**Opt1.**

```shell
git lfs clone https://github.com/fan1997/dtc_datasets.git
cd dtc_datasets
tar -zxvf reordered_matrices.tar.gz
tar -zxvf origin_matrices.tar.gz
```

**Opt2.**

[Baidu Netdisk]: https://pan.baidu.com/s/1jJzMX82KVQfV8taZWGq28g?pwd=2024

**Opt3.**

[Onedrive]: https://hkustgz-my.sharepoint.com/personal/rfan404_connect_hkust-gz_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Frfan404%5Fconnect%5Fhkust%2Dgz%5Fedu%5Fcn%2FDocuments%2Fdtc%5Fdataset&amp;ga=1

or you can use wget to download: 

```shell
wget "https://hkustgz-my.sharepoint.com/:u:/g/personal/rfan404_connect_hkust-gz_edu_cn/EcC5u1XfHwxKsfh7Xj8sWpMBoRGSXe6tlhkPnsHsuEekuQ?download=1" -O origin_matrices.tar.gz
wget wget "https://hkustgz-my.sharepoint.com/:u:/g/personal/rfan404_connect_hkust-gz_edu_cn/Efl7AV-oH5ZKj6dkjn6wtm8BZoRHAjrGcC1zPCkEQ-Omug?download=1" -O reordered_matrices.tar.gz
```



## 6. Run tests

``` SHELL
# Run DTCSpMM
cd ${DTC_HOME}/scripts/DTCSpMM 
# modify the dataset path in run_DTC_SpMM.py to your own path.
source run_DTC_SpMM.sh

# Run cuSPARSE
cd ${DTC_HOME}/scripts/cusparse 
# modify the dataset path in run_cuSPARSE.py to your own path.
source run_cuSPARSE_SpMM.sh

# Run Sputnik
cd ${DTC_HOME}/scripts/Sputnik 
# modify the dataset path in run_Sputnik.py to your own path.
source run_Sputnik.sh

# Run SparseTIR, you need to install SparseTIR (https://sampl.cs.washington.edu/SparseTIR/install.html)
cd ${DTC_HOME}/scripts/SparseTIR 
# modify the dataset path in run_sparsetir.py to your own path.
source run_SparseTIR.sh

# Run TCGNN-SpMM
cd ${DTC_HOME}/scripts/TCGNN 
# modify the dataset path in run_TCGNN_SpMM.py to your own path.
source run_TCGNN_SpMM.sh
```

## 7. Use TCA-reordering

```shell
cd TCA-reordering

# install minhashcuda
git clone https://github.com/src-d/minhashcuda.git
mv minhashcuda/* ./ && rm -r minhashcuda
cmake -DCMAKE_BUILD_TYPE=Release . && make && python setup.py install

# Run an example to reorder reddit dataset
python TCA_reorder.py --dataset reddit --thres 16
```

## Related work

- We extend our heartfelt gratitude to the outstanding **TC-GNN** project (https://github.com/YukeWang96/TC-GNN_ATC23.git). The development of DTC-SpMM was based on the foundational work established by TC-GNN.
- We draw insights from **AsPT** (http://gitlab.hpcrl.cse.ohio-state.edu/chong/ppopp19_ae)
- We draw insights from **Merge-SpMM** ([GitHub - owensgroup/merge-spmm: Code for paper "Design Principles for Sparse Matrix Multiplication on the GPU" accepted to Euro-Par 2018](https://github.com/owensgroup/merge-spmm))