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

<<<<<<< HEAD
## Notes
=======
### Development
Pre-commit hooks should be used to keep the codebase formatting consitent. After running installing the dependencies, run the following command to install the pre-commit hooks.

```bash
pre-commit install
```
>>>>>>> 1e05e3e (added precommit hooks)

#### Mistake in 50th nearest neighbor calculation

There is probably a mistake in the original code when it comes to the calucaltion of the 50th nearest neightbor.
The following code from [line 250 to 258]() in `run.py` is used, but the `self.data` variable
does not only contain the location data, but also the normal data. This is probably a minor mistake.

```python
sigma_set = []
ptree = cKDTree(self.data)

for p in np.array_split(self.data, 100, axis=0):
    d = ptree.query(p, 50 + 1)
    sigma_set.append(np.array(d[0][:, -1]))

sigmas = np.concatenate(sigma_set)
```