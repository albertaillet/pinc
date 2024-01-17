## PINC Reproducability

The repository contains the code for the reproducibility study of the NeurIPS 2023 paper "p-Poisson surface reconstruction in curl-free flow from point clouds" by Yesom Park, Taekyung Lee, Jooyoung Hahn, and Myungjoo Kang.'

### Installation

```bash
pip install -e .
```

### Data

The datasets used in the paper are the Surface Reconstruction Benchmark (SRB) and Thingi10K. It is unclear from the paper and the provided how the datasets were obtained.

#### Surface Reconstruction Benchmark (SRB)
We found in the README of [DiGS: Divergence guided shape implicit neural representation for unoriented point clouds](https://github.com/Chumbyte/DiGS) that SRB can be downloaded from using the provided script [`download_srb.sh`](https://github.com/Chumbyte/DiGS/blob/main/data/scripts/download_srb.sh) that downloads the data from the [Deep Geometric Prior for Surface Reconstruction](https://github.com/fwilliams/deep-geometric-prior) paper. This data is located on [Google Drive](https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view) is a 1.1GB zip file. The subfolders `scans` and `ground_truth` contain the data used in the paper.

#### Thingi10K

TODO: Find the data



