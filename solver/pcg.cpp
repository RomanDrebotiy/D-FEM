#include <stdio.h>
#include "pcg.h"
#include "../fem/per_block_global.h"
#include "../fem/distributed_synced_ops.h"
#include "../fem/mpi_iface_sync.h"

// conjugate gradient method with Jacobi diagonal preconditioner
GlobalVector pcg(GlobalSystem gs, double eps, int rank, int size, int max_iter) {
    int n = gs.get_size();
    GlobalVector x(n, gs.vec),
                 x1(n, gs.vec),
                 r(n, gs.vec),
                 err(n, gs.vec),
                 z(n, gs.vec),
                 Ap(n, gs.vec),
                 pgamma(n, gs.vec),
                 Apgamma(n, gs.vec),
                 pdelta(n, gs.vec);
    sync_vector(gs.vec, rank, size);
    subv_dest(gs.vec, gmv_synced(gs.matr, x, rank, size), r);
    GlobalVector diag = gs.matr.get_diagonal(gs.vec);
    sync_vector(diag, rank, size);
    diag.inverse_nonzero_with_eps(0.001);
    prodv_compwise_dest(diag, r, z);
    GlobalVector p = z;

    for (int it = 0; it < max_iter; it++) {
        printf("CG iter %d on rank %d\n", it, rank);
        gmv_synced_dest(gs.matr, p, rank, size, Ap);
        double rz = dot_synced(r, z, rank, size);
        double gamma = rz / dot_synced(Ap, p, rank, size);
        vm_num_dest(p, gamma, pgamma);
        addv_dest(x, pgamma, x1);
        subv_dest(x1, x, err);
        double err_norm = norm_synced(err, rank, size);
        double sol_norm = norm_synced(x1, rank, size);
        double rel_err = err_norm / sol_norm;
        printf("CG iter %d on rank %d relative error level: %f\n", it, rank, rel_err);
        if (rel_err < eps) {
            return x1;
        }
        x = x1;
        vm_num_dest(Ap, gamma, Apgamma);
        subv_dest(r, Apgamma, r);
        prodv_compwise_dest(diag, r, z);
        double delta = dot_synced(r, z, rank, size) / rz;
        vm_num_dest(p, delta, pdelta);
        addv_dest(z, pdelta, p);
    }

    throw -1;
}