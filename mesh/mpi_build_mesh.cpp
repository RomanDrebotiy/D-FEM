#include <mpi.h>
#include <stdio.h>
#include <utility>

#include "geometry.h"
#include "geometry_serializer.h"

int map_block_to_rank(int i, int j, int num_blocks) {
    return i * num_blocks + j;
}

std::pair<int, int> map_rank_to_block(int rank, int num_blocks) {
    return {rank / num_blocks, rank % num_blocks};
}

BlockMesh distribute_and_build_mesh(int rank, int size, int num_seg_per_block) {
    // blocks ^ 2 == size
    int num_blocks = number_blocks_per_dim(size);
    int num_internal_pts_per_block = (num_seg_per_block - 1) * (num_seg_per_block - 1);

    BlockMesh bm;
    int start_ind_for_internal_nodes = -1;

    if (rank == 0) {
        DistributedMesh dm({0, 0, 0.0, 0.0}, {0, 0, 1.0, 1.0}, num_blocks, num_seg_per_block);
        start_ind_for_internal_nodes = dm.get_next_idx();
        for (int dest_rank = 0; dest_rank < size; dest_rank++) {
            std::pair<int, int> block_ij = map_rank_to_block(dest_rank, num_blocks);
            BlockMesh bm_to_send = dm.blocks[block_ij.first][block_ij.second];
            if (dest_rank == 0) {
                bm = bm_to_send;
                continue;
            }
            std::vector<std::byte> buf = serialize_partial_block_mesh(bm_to_send);
            MPI_Send(&start_ind_for_internal_nodes, 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
            int buf_size = buf.size();
            MPI_Send(&buf_size, 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
            // this is small, so using just blocking calls
            int err = MPI_Send(buf.data(), buf.size(), MPI_BYTE, dest_rank, 0, MPI_COMM_WORLD);
            if (err != MPI_SUCCESS) {
                printf("MPI_Send failed\n");
                throw -1;
            }
        }
    } else {
        int msg_size;

        MPI_Recv(&start_ind_for_internal_nodes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
        MPI_Recv(&msg_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<std::byte> buf(msg_size);

        int err = MPI_Recv(buf.data(), buf.size(), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (err != MPI_SUCCESS) {
            printf("MPI_Recv failed\n");
            throw -1;
        }

        bm = deserialize_partial_block_mesh(buf);
    }

    bm.generate_mesh(start_ind_for_internal_nodes + num_internal_pts_per_block * (rank - 1));
    bm.assign_local_idx();

    // (!) We return full object copy here, which is bad... need to rework...
    // (!) But there is also an issue with dangling pointers here which is not always reproducible :)
    // (!) ... so we have 2 tasks for students...
    return bm;
}
