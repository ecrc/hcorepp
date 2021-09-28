# Copyright (c) 2017-2021,
# King Abdullah University of Science and Technology (KAUST).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

# CXX compiler must match the one used to compiler HCORE
# set it in your environment or here, e.g., CXX = g++

# set CXXFLAGS and LIBS
$(warning WARNING: The flags in the Makefile are hard-coded.)

# here is a hard-coded example using OpenBLAS
CXXFLAGS = -I/usr/local/hcore/include -std=c++11
LIBS     = -L/usr/local/hcore/lib$(LIB_SUFFIX) -lhcore -lopenblas

# rules
example_gemm_ccc: example_gemm_ccc.o
	$(CXX) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	-rm -f example_gemm_ccc *.o

# debugging
echo:
	@echo "CXX      $(CXX)"
	@echo "CXXFLAGS $(CXXFLAGS)"
	@echo "LIBS     $(LIBS)"
