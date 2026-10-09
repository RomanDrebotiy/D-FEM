#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <string>

#include "mesh/mpi_build_mesh.h"
#include "mesh/geometry.h"
#include "fem/per_block_global.h"
#include "solver/pcg.h"
#include "vtk/writer.h"

int main(int argc, char **argv)
{   
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--ocl") {
            USE_OPENCL = true;
            break;
        }
    }

    int rank, size;
    char hostname[256];

    bool multithreaded = true;

    if (multithreaded) {
        int provided;

        MPI_Init_thread(
            &argc,
            &argv,
            MPI_THREAD_MULTIPLE,
            &provided
        );

        if (provided < MPI_THREAD_MULTIPLE) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

    } else {
        MPI_Init(&argc, &argv);
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, sizeof(hostname));

    printf("Started MPI rank %d of %d on %s\n", rank, size, hostname);

    int num_seg_per_block = 1000;

    int num_dofs_per_dim = number_blocks_per_dim(size) * num_seg_per_block + 1;
    int num_dofs = num_dofs_per_dim * num_dofs_per_dim;
    if (rank == 0)
        printf("Total d.o.f. count: %d\n", num_dofs);

    double start = MPI_Wtime();
    double start_total = start;
    double elapsed = 0;

    BlockMesh bm = distribute_and_build_mesh(rank, size, num_seg_per_block);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        elapsed = MPI_Wtime() - start;
        printf("TIME >> Mesh distributed in %.6f\n", elapsed);
    }

    start = MPI_Wtime();
    GlobalSystem gs = assemble(bm);

    if (USE_OPENCL) {
        if (rank == 0) {
            printf("Using OpenCL\n");
        }
        init_opencl(gs.matr);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        elapsed = MPI_Wtime() - start;
        printf("TIME >> System assembled in %.6f\n", elapsed);
        printf("TIME >> PCG satrted\n");
    }

    start = MPI_Wtime();
    GlobalVector sol = pcg(gs, 0.05, rank, size, 100);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        elapsed = MPI_Wtime() - start;
        printf("TIME >> PCG finished in %.6f\n", elapsed);
        elapsed = MPI_Wtime() - start_total;
        printf("TIME >> Full FEM solver finished in %.6f\n", elapsed);
    }

    start = MPI_Wtime();
    write_vtk(bm, sol, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        elapsed = MPI_Wtime() - start;
        printf("TIME >> VTU generated in %.6f\n", elapsed);
        elapsed = MPI_Wtime() - start_total;
        printf("TIME >> Full pipeline finished in %.6f\n", elapsed);
    }

    MPI_Finalize();
    
    return 0;
}
