# HoloNav - Algorithmic Patient Alignment

Algorithmic approaches for performing patient alignment when using the HoloLens as a surgical navigation device.

_Part of the 2021-2022 [research project](https://github.com/TU-Delft-CSE/Research-Project) at [TU Delft](https://github.com/TU-Delft-CSE)._ 

---

## Algorithms

The repository contains different algorithms for performing rought point cloud alignment:

- Fast point feature histograms
- Principal component analysis
- Manual point selection

After the rough point cloud registration is performed using one of these algorithms, the registration is refined using the Iterative Closest Point algorithm.

## Repository structure

Each algorithm is contained in its own package. Every package contains a method which performs rough and precise point cloud registration on the provided point clouds. This method provides an easy to use endpoint for point cloud registration.

- The `pca_icp_alignment` method performs PCA rough registration followed by ICP precise registration
- The `fpfh_icp_alignment` method performs FPFH rough registration followed by ICP precise registration
- The `mps_icp_alignment` method performs manual point selection rough registration followed by ICP registration

To use a given algorithm, just import the algorithm from the package. For example:

```python
from pca.pca_icp_alignment import pca_icp_alignment
```

## Input

Every algorithm needs at least the following input parameters:

- `source`: The source point cloud (the point cloud that should be transformed)
- `target_depth_sensor`: The depth-sensor version of the target point cloud (the point cloud that the source should be transformed to)
- `target_pointer`: The pointer version of the target point cloud (the point cloud that the source should be transformed to)

Any additional supported parameters are present in the documentation.