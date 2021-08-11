# HCORE

**C++ software library provides the Basic Linear Algebra Subroutines (BLAS)**
**operations and Linear Algebra PACKage (LAPACK) for matrices in tile low-rank format**

**Extreme Computing Research Center (ECRC)**

**King Abdullah University of Science and Technology (KAUST)**

* * *

- [About](#about)
- [Documentation](#documentation)
- [Getting Help](#getting-help)
- [Resources](#resources)
- [References](#references)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

* * *

### About

HCORE software library implements BLAS and LAPACK functionality in the form of tile routines,
which update one or a small number of individual tiles, generally sequentially on a single
compute unit. Notebly, an m-by-n matrix is a collection of individual mb-by-nb tiles.
In the context of C++, HCORE tiles are first class objects, which are entities that can be
individually allocated, destroyed, and passed to low-level tile routines (e.g., GEMM).
HCORE tile routines rely on the tile low-rank compression, which replaces the dense operations
with the equivalent low-rank operations, to reduce the memory footprint and/or the time-to-solution

The objective of HCORE is to provide a convenient, performance-oriented API for development in the
C++ language, that, for the most part, preserves established conventions, while, at the same time,
takes advantages of modern C++ features, e.g., classes, namespaces, templates, exceptions,
standard containers, etc.

HCORE is part of the HiCMA project (Hierarchical Computations on Manycore Architectures (HiCMA)),
which aims to redesign existing dense linear algebra libraries to exploit the data sparsity
of the matrix operator. The core idea of HiCMA is to develop fast linear algebra computations
operating on the underlying tile low-rank data format, while satisfying a specified numerical
accuracy and leveraging performance from massively parallel hardware architectures. 

* * *

### Documentation

* [INSTALL.md](INSTALL.md) for installation notes.
* [HCORE Doxygen](https://ecrc.github.io/hcore/)
* [HiCMA Users' Guide]()
* [HiCMA Developers' Guide]()

* * *

### Getting Help

For assistance or bug reports, use Github's issue tracker:
https://github.com/ecrc/hcorepp/issues to create a new issue.

* * *

### Resources

* Visit the [HiCMA website](https://cemse.kaust.edu.sa/hicma)
  for more information about the Hierarchical Computations on Manycore
  Architectures (HiCMA) research group research group.
* Visit the [ECRC website](https://cemse.kaust.edu.sa/ecrc)
  to find out more about the Extreme Computing Research Center (ECRC).
* Visit the [HiCMA repository](https://github.com/ecrc/hicmapp)
  for more information about the HiCMA project.
* Visit the [BLAS++ repository](https://bitbucket.org/icl/blaspp)
  for more information about the C++ API for BLAS.
* Visit the [LAPACK++ repository](https://bitbucket.org/icl/lapackpp)
  for more information about the C++ API for LAPACK.
* Visit the [TestSweeper repository](https://bitbucket.org/icl/testsweeper)
  for more information about the C++ testing framework for parameter sweeps.

* * *

### References

[1] K. Akbudak, H. Ltaief, A. Mikhalev, and D. E. Keyes,
*Tile Low Rank Cholesky Factorization for Climate/Weather Modeling Applications
on Manycore Architectures*, **International Supercomputing Conference (ISC17)**,
June 18-22, 2017, Frankfurt, Germany.

[2] K. Akbudak, H. Ltaief, A. Mikhalev, A. Charara, and D. E. Keyes,
*Exploiting Data Sparsity for Large-Scale Matrix Computations*,
**Euro-Par 2018**, August 27-31, 2018, Turin, Italy.

[3] Q. Cao, Y. Pei, T. Herauldt, K. Akbudak, A. Mikhalev, G. Bosilca, H. Ltaief,
D. E. Keyes, and J. Dongarra, *Performance Analysis of Tile Low-Rank Cholesky
Factorization Using PaRSEC Instrumentation Tools*,
**2019 IEEE/ACM International Workshop on Programming and Performance
Visualization Tools (ProTools)**, Denver, CO, USA, 2019, pp. 25-32.

[4] Q. Cao, Y. Pei, K. Akbudak, A. Mikhalev, G. Bosilca, H. Ltaief, D. E. Keyes,
and J. Dongarra, *Extreme-Scale Task-Based Cholesky Factorization Toward Climate
and Weather Prediction Applications*, **The Platform for Advanced Scientific
Computing (PASC 2020)**.

[5] N. Al-Harthi, R. Alomairy, K. Akbudak, R. Chen, H. Ltaief, H. Bagci,
and D. E. Keyes, *Solving Acoustic Boundary Integral Equations Using High
Performance Tile Low-Rank LU Factorization*, **International Supercomputing
Conference (ISC 2020)**.

* * *

### Contributing

The HiCMA project welcomes contributions from new developers. Contributions can
be offered through the standard Github pull request model. We strongly encourage
you to coordinate large contributions with the HiCMA development team early in
the process. See [CONTRIBUTING.md](CONTRIBUTING.md) for contributing notes.

* * *

### Acknowledgments

For computer time, this research uses the Shaheen-2 supercomputer hosted at the
Supercomputing Laboratory at KAUST.

* * *

### License

Copyright (c) 2017-2021, King Abdullah University of Science and Technology
(KAUST). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
