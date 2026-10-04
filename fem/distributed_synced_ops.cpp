#include "distributed_synced_ops.h"

#include "per_block_global.h"
#include "mpi_iface_sync.h"
#include <cmath>
#include <mpi.h>

GlobalVector gmv_synced(GlobalMatrix& m, GlobalVector& v, int rank, int size) {
    GlobalVector res(v.get_size());
    gmv_synced_dest(m, v, rank, size, res);
    return res;
}

void gmv_synced_dest(GlobalMatrix& m, GlobalVector& v, int rank, int size, GlobalVector& res) {
    gmv_dest(m, v, res);
    sync_vector(res, rank, size);
}

// (!) only works for the case where one is synced and other not.
// Otherwise if both v and w are already synced, we do not need to do another sync, since it will provide bad results
GlobalVector prodv_compwise_synced(GlobalVector& v, GlobalVector& w, int rank, int size) {
    GlobalVector res(v.get_size());
    prodv_compwise_synced_dest(v, w, rank, size, res);
    return res;
}

// (!) only works for the case where one is synced and other not.
// Otherwise if both v and w are already synced, we do not need to do another sync, since it will provide bad results
void prodv_compwise_synced_dest(GlobalVector& v, GlobalVector& w, int rank, int size, GlobalVector& res) {
    prodv_compwise_dest(v, w, res);
    sync_vector(res, rank, size);
}

// works correctly for already synced vectors
double dot_synced(GlobalVector& v, GlobalVector& w, int rank, int size) {
    if (v.get_size() != w.get_size()) {
        throw -1;
    }

    SyncMeta sm = get_sync_meta(rank, size);
    std::array<double, 4> corner_multipliers;
    std::array<double, 4> iface_multipliers = {1.0, 1.0, 1.0, 1.0};

    for (int i = 0; i < 4; i++) {
        corner_multipliers[i] = 1.0 / (sm.corners.ranks_to_sync[i].size() + 1);
    }

    for (size_t i = 0; i < sm.ifaces.ifaces_active.size(); i++) {
        iface_multipliers[sm.ifaces.ifaces_active[i]] = 0.5;
    }

    double res = 0;

    for (int i = 0; i < 4; i++) {
        res += v.vals[i] * w.vals[i] * corner_multipliers[i];
        int i_start = v.iface_boundaries[i];
        int i_end = v.iface_boundaries[i + 1];
        for (int j = i_start; j < i_end; j++) {
            res += v.vals[j] * w.vals[j] * iface_multipliers[i];
        }
    }

    for (size_t i = v.iface_boundaries[4]; i < v.get_size(); i++) {
        res += v.vals[i] * w.vals[i];
    }

    MPI_Allreduce(
        MPI_IN_PLACE, 
        &res, 
        1, 
        MPI_DOUBLE, 
        MPI_SUM, 
        MPI_COMM_WORLD
    );

    return res;
}

double norm_synced(GlobalVector& v, int rank, int size) {
    return std::sqrt(
        dot_synced(v, v, rank, size)
    );
}