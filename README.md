## PINC Reproducability

The repository contains the code for the reproducibility study of the NeurIPS 2023 paper "p-Poisson surface reconstruction in curl-free flow from point clouds" by Yesom Park, Taekyung Lee, Jooyoung Hahn, and Myungjoo Kang.

### Installation

```bash
pip install -e .
```

### Development
Pre-commit hooks are used to keep the codebase formatting consitent. Run the following to set it up

```bash
pip install -e ".[dev]"  # Install dev dependencies
pre-commit install
```

### Data

The datasets used in the paper are the Surface Reconstruction Benchmark (SRB) and Thingi10K. It is unclear from the paper and the provided how the datasets were obtained.

#### Surface Reconstruction Benchmark (SRB)

The SRB dataset was found in the README of [DiGS: Divergence guided shape implicit neural representation for unoriented point clouds](https://github.com/Chumbyte/DiGS) that SRB can be downloaded from using the provided script [`download_srb.sh`](https://github.com/Chumbyte/DiGS/blob/main/data/scripts/download_srb.sh) that downloads the data from the [Deep Geometric Prior for Surface Reconstruction](https://github.com/fwilliams/deep-geometric-prior) paper. This data is located on [Google Drive](https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view) is a 1.1GB zip file. The subfolders `scans` and `ground_truth` contain the data used in the paper.

## Notes

#### Mistake in 50th nearest neighbor calculation

There is probably a mistake in the original code when it comes to the calucaltion of the 50th nearest neightbor.
The following code from [line 251 to 258](https://github.com/Yebbi/PINC/blob/main/reconstruction/run.py#L251-L258) in `run.py` is used, but the `self.data` variable
does not only contain the location data, but also the normal data. This is probably a minor mistake.

```python
sigma_set = []
ptree = cKDTree(self.data)

for p in np.array_split(self.data, 100, axis=0):
    d = ptree.query(p, 50 + 1)
    sigma_set.append(np.array(d[0][:, -1]))

sigmas = np.concatenate(sigma_set)
```

#### Number of sampled global points

The paper does not mention how many points are sampled outside the points cloud. This was found in the file
`sample.py` on [line 33](https://github.com/Yebbi/PINC/blob/main/model/sample.py#L33) and is set to one eighth of the number of points sampled from the point cloud.

```python
sample_global = (torch.rand(batch_size, sample_size//8, dim, device=pc_input.device, requires_grad=True) * (self.global_sigma * 2)) - self.global_sigma
```

#### What Minkowski p-norm to use.

In the paper it is not clear what Minkowski p-norm to use. The paper first mentions the l1-norm, but then uses the l2-norm in the equations. The [evaluation code from DiGs](https://github.com/Chumbyte/DiGS/blob/main/surface_reconstruction/compute_metrics_srb.py) uses the l2-norm, but it is yet to be confirmed if this is the correct norm to use.


#### Geometric initialization is slightly different

```python
def create_params(w_mean: Array, w_std: Array, b_const: float) -> tuple[Array, Array]:
    return w_mean + w_std * normal(key, (out_dim, in_dim)), b_const * jnp.ones(out_dim)

# Geometric initialization according to "SAL: Sign Agnostic Learning of Shapes From Raw Data"
# https://github.com/matanatz/SAL/blob/master/code/model/network.py#L112
# PINC implementation:
# https://github.com/Yebbi/PINC/blob/main/model/network.py#L46
# IGR implementation:
# https://github.com/amosgropp/IGR/blob/master/code/model/network.py#L48
if last_layer:
    # Inconsitency between SAL and PINC (factor of 2) and p in denominator
    # in SAL: 2*sqrt(pi) / sqrt(in_dim*p)
    # in PINC:  sqrt(pi) / sqrt(in_dim)
    return create_params(w_mean=jnp.sqrt(jnp.pi / in_dim), w_std=jnp.array(1e-5), b_const=-0.1)
return create_params(w_mean=jnp.array(0), w_std=jnp.sqrt(2 / in_dim), b_const=0.0)
```

#### Division by sqrt(2) in the skip connection

The layer with the skip connection in the model is divided by sqrt(2) in the original code. This is not mentioned in the paper.

```python
if layer in self.skip_in:
    output = torch.cat([output, input], -1) / np.sqrt(2)
```
