# Distributed FEM

This repo contains a simple educational example of parallel FEM implementation using OpenMPI library for a 2D scalar second-order equation on a unit square. It is conceptually similar to the PETSc library.

Provided code demonstrates the following:
- domain decomposition;
- distributed mesh generation;
- parallel linear algebra structures:
  - distributed sparse matrix, stored in a CSR format;
  - distributed vector with interface synchronization across cluster nodes;
- distributed FEM linear system assembly;
- parallel implementation of the conjugate gradient method;
- Jacobi diagonal preconditioner;
- VTK files generation for obtained solution.

# Commands

## compile

```bash
mpicxx main.cpp fem/distributed_synced_ops.cpp fem/local.cpp fem/mpi_iface_sync.cpp fem/per_block_global.cpp mesh/geometry.cpp mesh/geometry_serializer.cpp mesh/mpi_build_mesh.cpp solver/pcg.cpp vtk/writer.cpp -lm -lOpenCL -o d_fem
```

## run

Without OpenCL usage:

```bash
mpirun --host node01:4,node02:6,node03:6 -np 16 d_fem
```

With OpenCL usage for sparse matrix-vector product:

```bash
mpirun --host node01:4,node02:6,node03:6 -np 16 d_fem --ocl
```

# Notes

Library is created for educational purposes, it is not polished and can contain bugs.

**Written without code generation by AI tools.**

*Copyright (c) 2026 Roman Drebotiy*

*Licensed under the Apache License 2.0 (see LICENSE file)*