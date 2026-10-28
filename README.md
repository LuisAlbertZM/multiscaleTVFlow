# multiscaleTVFlow
This repository contains code to perform multi-scale roto-translational total variation (MS-RT-TV) flow as described in [[1]](#1). The results of [[1]](#1) can be reproduced with the `denoising.ipynb` notebook in the `experiments` directory.

This work extends roto-translational total variation (RT-TV) flow Smets et al. [[2]](#2) and Chambolle & Pock [[3]](#3).

## Installation
The core functionality of this repository requires (in brackets the versions that we used):
* `python` (3.14.4)
* `torch` (2.13.0+cu132)
* `numpy` (2.5.2)
* `scipy` (1.18.0)
* `tqdm` (4.70.0)

To reproduce the experiments, one additionally needs (in brackets the versions that we used):
* `jupyter` (1.1.1)
* `matplotlib` (3.11.1)
* `bm3d` (4.0.3)

The code can be installed as a package into your python environment by executing
```
pip install -e .
```
in the root directory of this repository. 

## Functionality
The main functionality is as follows:
| Method | Description |
| ------ | ----------- |
| `msrtv.SIM2.tvflow.denoise`    | MS-RT-TV flow [[1]](#1) |
| `msrtv.SE2.tvflow.denoise`    | RT-TV flow [[2]](#2), [[3]](#3) |
| `msrtv.SIM2.threshold.denoise`    | soft-thresholding [[4]](#4) |
| `msrtv.SIM2.simoncellitransform.wavelet_transform`    | continuous wavelet transform (CWT) |
| `msrtv.SIM2.simoncellitransform.steerable_wavelet_stack`    | Simoncelli pyramid for CWT |
| `msrtv.SE2.cakewavelettransform.wavelet_transform`    | orientation score transform |
| `msrtv.SE2.cakewavelettransform.cakewavelet_stack`    | Cakewavelets for orientation score |

## Cite
If you use this code in your own work, please cite our paper:

<a id="1">[1]</a> "Multi-Scale Roto-Translational Total Variation Flow."
```
@misc{MSRTV,
  title =        {{Multi-Scale Roto-Translational Total Variation Flow}}
}
```

We extend roto-translational total variation (RT-TV) flow by Smets et al. and Chambolle & Pock; this repository contains our own reimplementations:

<a id="2">[2]</a> Smets, B.M.N., Portegies, J.W., St-Onge, E., Duits, R. "Total Variation and Mean Curvature PDEs on the Homogeneous Space of Positions and Orientations." Journal of Mathematical Imaging and Vision (2021). https://doi.org/10.1007/s10851-020-00991-4
```
@article{Smets2021TV,
  author =       {Smets, Bart M.N. and Portegies, Jacobus W. and St-Onge, Etienne and Duits, Remco},
  title =        {{Total Variation and Mean Curvature PDEs on the Homogeneous Space of Positions and Orientations}},
  journal =      {Journal of Mathematical Imaging and Vision},
  publisher =    {Springer},
  year =         {2021},
  volume =       {63},
  pages =        {237--262},
  doi =          {10.1007/s10851-020-00991-4}
}
```

<a id="3">[3]</a> Chambolle, A., Pock, Th. "Total roto-translational variation." Numerische Mathematik (2019). https://doi.org/10.1007/s00211-019-01026-w
```
@article{Chambolle2019TRTV,
  author =       {Chambolle, Antonin and Pock, Thomas},
  title =        {{Total roto-translational variation}},
  journal =      {Numerische Mathematik},
  publisher =    {Springer},
  year =         {2019},
  volume =       {142},
  pages =        {611--666},
  doi =          {10.1007/s00211-019-01026-w}
}
```

We compare to denoising by soft-thresholding the wavelet coefficients, as proposed by Donoho:

<a id="4">[4]</a> Donoho, D.L. "De-Noising by Soft-Thresholding." IEEE Transactions on Information Theory (1995). https://doi.org/10.1109/18.382009
```
@article{Donoho1995DenoisingThresholding,
  author =       {Donoho, David L.},
  title =        {{De-Noising by Soft-Thresholding}},
  journal =      {IEEE Transactions on Information Theory},
  publisher =    {IEEE},
  year =         {1995},
  volume =       {41},
  number =       {3},
  pages =        {613--627},
  doi =          {10.1109/18.382009}
}
```
