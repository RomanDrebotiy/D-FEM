
#include "mpi_iface_sync.h"
#include "per_block_global.h"
#include "../mesh/mpi_build_mesh.h"
#include "../mesh/geometry.h"
#include <utility>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <mpi.h>


int get_tag_for_iface(int rank, int sibling_rank, int size) {
    int r1 = rank;
    int r2 = sibling_rank;
    if (r1 < r2) {
        std::swap(r1, r2);
    }
    int digits = std::log10(size) + 1;
    int p = std::pow(10, digits);
    return r1 * p + r2;
}

int get_tag_for_corner(int rank, std::vector<int> sibling_ranks, int size) {
    std::vector<int> ranks(sibling_ranks);
    ranks.push_back(rank);
    std::sort(ranks.begin(), ranks.end());
    int digits = std::log10(size) + 1;
    int p = std::pow(10, digits);
    int s = size * (p + 1); // start from this to not overlap with iface tags
    int base = 1;
    for (size_t i = 0; i < ranks.size(); i++) {
        s += ranks[i] * base;
        base *= p;
    }
    return s;
}

int get_accum_rank_for_iface(int rank, int sibling_rank) {
    return std::min(rank, sibling_rank);
}

int get_accum_rank_for_corner(int rank, std::vector<int> sibling_ranks) {
    std::vector<int> ranks(sibling_ranks);
    ranks.push_back(rank);
    std::sort(ranks.begin(), ranks.end());
    return *std::min_element(ranks.begin(), ranks.end());
}

SyncMeta get_sync_meta(int rank, int size) {
    int num_blocks = number_blocks_per_dim(size);
    std::pair<int, int> block_ij = map_rank_to_block(rank, num_blocks);
    int i = block_ij.first;
    int j = block_ij.second;

    // ifaces
    SyncMeta sm;
    if (i > 0) {
        sm.ifaces.ifaces_active.push_back(0);
        sm.ifaces.ranks_to_sync.push_back(
            map_block_to_rank(i - 1, j, num_blocks)
        );
    }
    if (j < num_blocks - 1) {
        sm.ifaces.ifaces_active.push_back(1);
        sm.ifaces.ranks_to_sync.push_back(
            map_block_to_rank(i, j + 1, num_blocks)
        );
    }
    if (i < num_blocks - 1) {
        sm.ifaces.ifaces_active.push_back(2);
        sm.ifaces.ranks_to_sync.push_back(
            map_block_to_rank(i + 1, j, num_blocks)
        );
    }
    if (j > 0) {
        sm.ifaces.ifaces_active.push_back(3);
        sm.ifaces.ranks_to_sync.push_back(
            map_block_to_rank(i, j - 1, num_blocks)
        );
    }

    // corners
    int di = -1;
    int dj = -1;
    for (int k = 0; k < 4; k++) {
        int i_shifted = i + di;
        int j_shifted = j + dj;
        bool i_inside = (i_shifted >= 0 && i_shifted <= num_blocks - 1);
        bool j_inside = (j_shifted >= 0 && j_shifted <= num_blocks - 1);
        if (i_inside) {
            sm.corners.ranks_to_sync[k].push_back(
                map_block_to_rank(i_shifted, j, num_blocks)
            );
        }
        if (j_inside) {
            sm.corners.ranks_to_sync[k].push_back(
                map_block_to_rank(i, j_shifted, num_blocks)
            );
        }
        if (i_inside && j_inside) {
            sm.corners.ranks_to_sync[k].push_back(
                map_block_to_rank(i_shifted, j_shifted, num_blocks)
            );
        }
        // rotate by 90 degree counterclockwise
        int c = di;
        di = dj;
        dj = -c;
    }

    return sm;
}

void sync_vector(GlobalVector& v, int rank, int size) {
    int n = number_blocks_per_dim(size);
    SyncMeta sm = get_sync_meta(rank, size);
    std::vector<MPI_Request> requests;
    // sync interfaces (inner nodes)
    requests.reserve(2 * n * (n + 1));
    for (size_t i = 0; i < sm.ifaces.ranks_to_sync.size(); i++) {
        int iface_idx = sm.ifaces.ifaces_active[i];
        int sibling_rank = sm.ifaces.ranks_to_sync[i];
        int i_start = v.iface_boundaries[iface_idx];
        int i_end = v.iface_boundaries[iface_idx + 1];
        int tag = get_tag_for_iface(rank, sibling_rank, size);
        int accum_rank = get_accum_rank_for_iface(rank, sibling_rank);
        int count = i_end - i_start;

        // TODO: rework this to use allreduce as for corners. For now keeping it as a demo
        if (rank == accum_rank) {
            std::vector<double> recv(count);

            MPI_Recv(
                recv.data(),
                count,
                MPI_DOUBLE,
                sibling_rank,
                tag,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            for (int k = 0; k < count; k++) {
                v.vals[i_start + k] += recv[k];
            }

            requests.push_back(MPI_REQUEST_NULL);
            MPI_Isend(
                &v.vals[i_start],       // start of data
                count,                  // number of elements
                MPI_DOUBLE,             // element type
                sibling_rank,           // destination
                tag,                    // tag
                MPI_COMM_WORLD,
                &requests.back()
            );
        } else {
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Isend(
                &v.vals[i_start],
                count,
                MPI_DOUBLE,
                sibling_rank,
                tag,
                MPI_COMM_WORLD,
                &requests.back()
            );

            MPI_Recv(
                &v.vals[i_start],
                count,
                MPI_DOUBLE,
                sibling_rank,
                tag,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        }
    }

    MPI_Waitall(
        requests.size(),
        requests.data(),
        MPI_STATUSES_IGNORE
    );

    requests.clear();
    int ranks_num = 4; //sm.corners.ranks_to_sync.size() // always = 4 (!)

    std::vector<MPI_Comm> sub_comms;

    for (int i = 0; i < ranks_num; i++) {
        std::vector<int> sibling_ranks = sm.corners.ranks_to_sync[i];
        int tag = MPI_UNDEFINED;
        if (sibling_ranks.size() != 0) {
            tag = get_tag_for_corner(rank, sibling_ranks, size);
        }
        
        MPI_Comm subcomm;
        MPI_Comm_split(MPI_COMM_WORLD, tag, rank, &subcomm);
        sub_comms.push_back(subcomm);
        if (sub_comms.back() != MPI_COMM_NULL) {
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Iallreduce(
                MPI_IN_PLACE, 
                &v.vals[i], 
                1, 
                MPI_DOUBLE, 
                MPI_SUM, 
                sub_comms.back(), 
                &requests.back()
            );
        }
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    for (size_t i = 0; i < sub_comms.size(); i++) {
        if (sub_comms[i] != MPI_COMM_NULL) {
            MPI_Comm_free(&sub_comms[i]);
        }
    }
}

void sync_matrix(GlobalMatrix& v, int rank, int size) {
    // not needed for now. Only vector sync can be used.
    throw -1;
}