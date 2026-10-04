#pragma once

#include <vector>
#include <array>

int number_blocks_per_dim(int comm_size);

struct Node {
    int block_local_idx;
    int global_idx;
    double x;
    double y;
};

struct Triangle {
    Node* verts[3];
};

class MeshInterface {
public:
    std::vector<Node> nodes;
    bool active;

    MeshInterface();
    MeshInterface(const Node& start, const Node& end, int num_seg, int start_ind, bool is_active);
    std::vector<Node> get_nodes();
    int get_max_inner_idx();
    int get_start_idx();
    int get_end_idx();
};

class BlockMesh {
public:
    int num_seg_per_block;
    std::array<Node, 4> corners; // 00 10 11 01
    std::array<MeshInterface*, 4> ifaces; // bottom right top left. Separated those interfaces to store halo nodes.
    std::vector<Node> inner_nodes;
    std::vector<Triangle> triangles;
    std::array<bool, 4> boundary_markers;

    BlockMesh();
    BlockMesh(int num_seg_per_block, const std::array<Node, 4>& block_corners, const std::array<bool, 4>& bnd_markers);
    int generate_mesh(int start_ind);
    int assign_local_idx();
    int get_matr_size();
};

class DistributedMesh {
public:
    std::vector<std::vector<BlockMesh>> blocks;
    int max_global_iface_ind;

    /**
     *   +------------ tr
     *   |             |
     *   |             |
     *  bl ------------+
     */
    DistributedMesh(const Node& bl, const Node& tr, int num_blocks, int num_seg_per_block);
    bool is_on_bnd(int i, int j, int num_blocks);
    int get_next_idx();
};
