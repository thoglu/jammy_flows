# jammy_flows

<img src="https://github.com/thoglu/jammy_flows/workflows/build/badge.svg"> <img src="https://github.com/thoglu/jammy_flows/workflows/tests/badge.svg">

This package implements (conditional) PDFs with **J**oint **A**utoregressive **M**anifold (**MY**) normalizing-flows. It grew out of work for the paper [Unifying supervised learning and VAEs - automating statistical inference in (astro-)particle physics with amortized conditional normalizing flows [arXiv:2008.05825]](https://arxiv.org/abs/2008.05825) and includes the paper's described methodology for coverage calculation of PDFs on tensor products of manifolds. For Euclidean manifolds, it includes an updated implementation of the [offical implementation](https://github.com/chenlin9/Gaussianization_Flows) of [Gaussianization flows [arXiv:2003.01941]](https://arxiv.org/abs/2003.01941), where now the inverse is differentiable (adding Newton iterations to the bisection) and made more stable using better approximations of the inverse Gaussian CDF. Several other state-of-the art flows are implemented sometimes using slight modifications or extensions.

The package has a simple syntax that lets the user define a PDF and get going with a single line of code that **should just work**. To define a 10-d PDF, with 4 Euclidean dimensions, followed by a 2-sphere, followed again by 4 Euclidean dimensions, one could for example write
```
import jammy_flows

pdf=jammy_flows.pdf("e4+s2+e4", "gggg+n+gggg")
```
The first argument describes the manifold structure, the second argument the flow layers for a particular manifold. Here **"g"** and **"n"** stand for particular normalizing flow layers that are pre-implemented (see **Features** below). The Euclidean parts in this example use 4 **"g"** layers each.
<img src="animation.gif" alt="drawing" width="800"/>

Have a look at the [script](examples/jammy_flows.py) that generates the above animation.

### Documentation

The docs can be found [here](https://thoglu.github.io/jammy_flows/index.html).

Also check out the [example notebook](examples/examples.ipynb).

## Features

### General

- [x] Autoregressive conditional structure is taken care of behind the scenes and connects manifolds
- [x] Coverage is straightforward. Everything (including spherical, interval and simplex flows) is based on a Gaussian base distribution ([arXiv:2008.0582](https://arxiv.org/abs/2008.05825)).
- [x] Bisection & Newton iterations for differentiable inverse (used for certain non-analytic inverse flow functions)
- [x] amortizable MLPs that can use low-rank approximations
- [x] amortizable PDFs - the total PDF can be the output of another neural network
- [x] unit tests that make sure backwards / and forward flow passes of all implemented flow-layers agree 
- [x] include log-lambda as an additional flow parameter to define parametrized Poisson-Processes
- [x] easily extendible: define new Euclidean / spherical flow layers by subclassing Euclidean or spherical base classes

### Euclidean flows:

- [x] Generic affine flow (Multivariate normal distribution) (**"t"**)
- [x] Gaussianization flow [arXiv:2003.01941](https://arxiv.org/abs/2003.01941) (**"g"**)
- [x] Hybrid of nonlinear scalings and rotations ("Polynomial Stretch flow") (**"p"**)

### Spherical flows:

### S1:
- [x] Moebius transformations  (described in [arXiv:2002.02428](https://arxiv.org/abs/2002.02428)) (**"m"**)
- [x] Circular rational-quadratic splines  (described in [arXiv:2002.02428](https://arxiv.org/abs/2002.02428)) (**"o"**)

### S2:
- [x] Autorregressive flow for N-Spheres ([arXiv:2002.02428](https://arxiv.org/abs/2002.02428)) (**"n"**)
- [x] Exponential map flow ([arXiv:0906.0874](https://arxiv.org/abs/0906.0874)/[arXiv:2002.02428](https://arxiv.org/abs/2002.02428)) (**"v"**)
- [x] Neural Manifold Ordinary Differential Equations [arXiv:2006.10254](https://arxiv.org/abs/2006.10254) (**"c"**)
 
### Interval Flows:

- [x] "Neural Spline Flows" (Rational-quadratic splines) [arXiv:1906.04032](https://arxiv.org/abs/1906.04032) (**"r"**)

### Simplex Flows:

- [x] Autoregressive simplex flow [arXiv:2008.05456](https://arxiv.org/abs/2008.05456) (**"w"**)

For a description of all flows and abbreviations, have a look in the docs [here](https://thoglu.github.io/jammy_flows/usage/api.html#module-jammy_flows.layers.layer_base).

## Requirements

- pytorch (>=1.7)
- numpy (>=1.18.5)
- scipy (>=1.5.4)
- matplotlib (>=3.3.3)
- torchdiffeq (>=0.2.1)

The package has been built and tested with these versions, but might work just fine with older ones.

## Installation

### specific version:

```
pip install git+https://github.com/thoglu/jammy_flows.git@*tag* 
```
e.g.
```
pip install git+https://github.com/thoglu/jammy_flows.git@1.0.0
```
to install release 1.0.0.

### master:

```
pip install git+https://github.com/thoglu/jammy_flows.git
```
## Contributions

If you want to implement your own layer or have bug / feature suggestions, just file an issue.