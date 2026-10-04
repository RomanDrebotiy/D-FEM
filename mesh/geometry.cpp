#include "geometry.h"

#include <cmath>
#include <vector>
#include <array>

int number_blocks_per_dim(int comm_size) {
    return static_cast<int>(std::floor(std::sqrt(comm_size)));
}

// ---- MeshInterface ----

MeshInterface::MeshInterface() : active(false) {

}

MeshInterface::MeshInterface(
    const Node& start, const Node& end, int num_seg, int start_ind, bool is_active
) : active(is_active) {
    nodes.push_back(start);
    for (int i = 0; i < num_seg - 1; i++) {
        nodes.push_back(
            Node{
                0,
                start_ind + i,
                start.x + (end.x - start.x) * (i + 1) / num_seg,
                start.y + (end.y - start.y) * (i + 1) / num_seg
            }
        );
    }
    nodes.push_back(end);
}

std::vector<Node> MeshInterface::get_nodes() {
    return nodes;
}

int MeshInterface::get_max_inner_idx() {
    return nodes[nodes.size() - 2].global_idx;
}

int MeshInterface::get_start_idx() {
    return nodes[0].global_idx;
}

int MeshInterface::get_end_idx() {
    return nodes[nodes.size() - 1].global_idx;
}

// ---- BlockMesh ----

BlockMesh::BlockMesh() {

}

BlockMesh::BlockMesh(
    int num_seg_per_block,
    const std::array<Node, 4>& block_corners, 
    const std::array<bool, 4>& bnd_markers
) : num_seg_per_block(num_seg_per_block), corners(block_corners), boundary_markers(bnd_markers) {
    
}

int BlockMesh::generate_mesh(int start_ind) {
    int num_pts = ifaces[0]->nodes.size() - 2;
    inner_nodes.reserve(num_pts * num_pts);
    triangles.reserve((num_pts + 1) * (num_pts + 1) * 2);
    double dx = (corners[1].x - corners[0].x) / (num_pts + 1);
    double dy = (corners[2].y - corners[1].y) / (num_pts + 1);
    for (int i = 0; i < num_pts; i++) {
        if (i == 0) { 
            for (int j = 0; j < num_pts; j++) {
                inner_nodes.push_back(
                    Node{
                        0, 
                        start_ind++,
                        corners[0].x + (j + 1) * dx,
                        corners[1].y + (i + 1) * dy
                    }
                );
                triangles.push_back(
                    Triangle{
                        &inner_nodes[inner_nodes.size() - 1],
                        &ifaces[0]->nodes[j],
                        &ifaces[0]->nodes[j + 1]
                    }
                );
                triangles.push_back(
                    Triangle{
                        &inner_nodes[inner_nodes.size() - 1],
                        j == 0 ? &ifaces[3]->nodes[i + 1] :  &inner_nodes[inner_nodes.size() - 2],
                        &ifaces[0]->nodes[j]
                    }
                );
            }

            triangles.push_back(
                Triangle{
                    &inner_nodes[inner_nodes.size() - 1],
                    &ifaces[0]->nodes[num_pts],
                    &ifaces[1]->nodes[1]
                }
            );

            triangles.push_back(
                Triangle{
                    &ifaces[1]->nodes[1],
                    &ifaces[0]->nodes[num_pts],
                    &ifaces[1]->nodes[0]
                }
            );
        } else {
            for (int j = 0; j < num_pts; j++) {
                inner_nodes.push_back(
                    Node{
                        0, 
                        start_ind++,
                        corners[0].x + (j + 1) * dx,
                        corners[1].y + (i + 1) * dy
                    }
                );
                triangles.push_back(
                    Triangle{
                        &inner_nodes[inner_nodes.size() - 1],
                        j == 0 ? &ifaces[3]->nodes[i] : &inner_nodes[inner_nodes.size() - 1 - num_pts - 1],
                        &inner_nodes[inner_nodes.size() - 1 - num_pts]
                    }
                );
                triangles.push_back(
                    Triangle{
                        &inner_nodes[inner_nodes.size() - 1],
                        j == 0 ? &ifaces[3]->nodes[i + 1] : &inner_nodes[inner_nodes.size() - 2],
                        j == 0 ? &ifaces[3]->nodes[i] : &inner_nodes[inner_nodes.size() - 1 - num_pts - 1]
                    }
                );
            }

            triangles.push_back(
                Triangle{
                    &inner_nodes[inner_nodes.size() - 1],
                    &inner_nodes[inner_nodes.size() - 1 - num_pts],
                    &ifaces[1]->nodes[i + 1]
                }
            );

            triangles.push_back(
                Triangle{
                    &ifaces[1]->nodes[i + 1],
                    &inner_nodes[inner_nodes.size() - 1 - num_pts],
                    &ifaces[1]->nodes[i]
                }
            );
        }
    }

    for (int j = 1; j <= num_pts + 1; j++) {
        if (j == 1) {
            triangles.push_back(
                Triangle{
                    &ifaces[2]->nodes[j],
                    &ifaces[3]->nodes[num_pts],
                    &inner_nodes[inner_nodes.size() - 1 - (num_pts - j)]
                }
            );
            triangles.push_back(
                Triangle{
                    &ifaces[2]->nodes[j],
                    &ifaces[2]->nodes[j - 1],
                    &ifaces[3]->nodes[num_pts]
                }
            );
        } else if (j == num_pts + 1) {
            triangles.push_back(
                Triangle{
                    &ifaces[2]->nodes[j],
                    &ifaces[2]->nodes[j - 1],
                    &inner_nodes[inner_nodes.size() - 1]
                }
            );
            triangles.push_back(
                Triangle{
                    &ifaces[2]->nodes[j],
                    &inner_nodes[inner_nodes.size() - 1],
                    &ifaces[1]->nodes[num_pts]
                }
            );
        } else {
            triangles.push_back(
                Triangle{
                    &ifaces[2]->nodes[j],
                    &inner_nodes[inner_nodes.size() - 2 - (num_pts - j)],
                    &inner_nodes[inner_nodes.size() - 1 - (num_pts - j)]
                }
            );
            triangles.push_back(
                Triangle{
                    &ifaces[2]->nodes[j],
                    &ifaces[2]->nodes[j - 1],
                    &inner_nodes[inner_nodes.size() - 2 - (num_pts - j)]
                }
            );
        }
    }

    return start_ind;
}

int BlockMesh::assign_local_idx() {
    int ind = 0;
    for (int i = 0; i < 4; i++) {
        corners[i].block_local_idx = ind++;
    }

    ifaces[0]->nodes.front().block_local_idx = corners[0].block_local_idx;
    ifaces[0]->nodes.back().block_local_idx = corners[1].block_local_idx;

    ifaces[1]->nodes.front().block_local_idx = corners[1].block_local_idx;
    ifaces[1]->nodes.back().block_local_idx = corners[2].block_local_idx;

    ifaces[2]->nodes.front().block_local_idx = corners[3].block_local_idx;
    ifaces[2]->nodes.back().block_local_idx = corners[2].block_local_idx;

    ifaces[3]->nodes.front().block_local_idx = corners[0].block_local_idx;
    ifaces[3]->nodes.back().block_local_idx = corners[3].block_local_idx;

    for (int i = 0; i < 4; i++) {
        for (size_t k = 1; k < ifaces[i]->nodes.size() - 1; k++) {
            ifaces[i]->nodes[k].block_local_idx = ind++;
        }
    }

    for (size_t i = 0; i < inner_nodes.size(); i++) {
        inner_nodes[i].block_local_idx = ind++;
    }

    return ind;
}

int BlockMesh::get_matr_size() {
    return inner_nodes.back().block_local_idx + 1;
}

// ---- DistributedMesh ----

DistributedMesh::DistributedMesh(const Node& bl, const Node& tr, int num_blocks, int num_seg_per_block) {
    blocks = std::vector<std::vector<BlockMesh>>(num_blocks, std::vector<BlockMesh>(num_blocks));
    double dx = (tr.x - bl.x) / num_blocks;
    double dy = (tr.y - bl.y) / num_blocks;
    for (int i = 0; i < num_blocks; i++) {
        for (int j = 0; j < num_blocks; j++) {
            double bx0 = bl.x + j * dx;
            double by0 = bl.y + i * dy;
            blocks[i][j] = BlockMesh(
                num_seg_per_block,
                std::array<Node, 4>{
                    Node{0, i * (num_blocks + 1) + j,           bx0,      by0     }, // 00
                    Node{0, i * (num_blocks + 1) + j + 1,       bx0 + dx, by0     }, // 10
                    Node{0, (i + 1) * (num_blocks + 1) + j + 1, bx0 + dx, by0 + dy}, // 11
                    Node{0, (i + 1) * (num_blocks + 1) + j,     bx0,      by0 + dy}  // 01
                },
                std::array<bool, 4>{
                    is_on_bnd(i,     j,     num_blocks),
                    is_on_bnd(i,     j + 1, num_blocks),
                    is_on_bnd(i + 1, j + 1, num_blocks),
                    is_on_bnd(i + 1, j,     num_blocks),
                }
            );
        }
    }
    
    int global_ind = (blocks.size() + 1) * (blocks.size() + 1);

    // ----- boundary ifaces ------
    // bottom
    for (int j = 0; j < num_blocks; j++) {
        BlockMesh *m = &blocks[0][j];
        m->ifaces[0] = new MeshInterface(
            m->corners[0], 
            m->corners[1], 
            num_seg_per_block, 
            global_ind, 
            false
        );
        global_ind = m->ifaces[0]->get_max_inner_idx() + 1;
    }

    // right
    for (int i = 0; i < num_blocks; i++) {
        BlockMesh *m = &blocks[i][num_blocks - 1];
        m->ifaces[1] = new MeshInterface(
            m->corners[1], 
            m->corners[2], 
            num_seg_per_block, 
            global_ind, 
            false
        );
        global_ind = m->ifaces[1]->get_max_inner_idx() + 1;
    }

    // top
    for (int j = 0; j < num_blocks; j++) {
        BlockMesh *m = &blocks[num_blocks - 1][j];
        m->ifaces[2] = new MeshInterface(
            m->corners[3], 
            m->corners[2], 
            num_seg_per_block, 
            global_ind, 
            false
        );
        global_ind = m->ifaces[2]->get_max_inner_idx() + 1;
    }

    // left
    for (int i = 0; i < num_blocks; i++) {
        BlockMesh *m = &blocks[i][0];
        m->ifaces[3] = new MeshInterface(
            m->corners[0], 
            m->corners[3], 
            num_seg_per_block, 
            global_ind, 
            false
        );
        global_ind = m->ifaces[3]->get_max_inner_idx() + 1;
    }

    // ----- inner ifaces ------

    // vertical
    for (int i = 0; i < num_blocks; i++) {
        for (int j = 0; j < num_blocks - 1; j++) {
            BlockMesh *m = &blocks[i][j];
            m->ifaces[1] = new MeshInterface(
                m->corners[1], 
                m->corners[2], 
                num_seg_per_block, 
                global_ind, 
                true
            );
            blocks[i][j + 1].ifaces[3] = m->ifaces[1];
            global_ind = m->ifaces[1]->get_max_inner_idx() + 1;
        }
    }

    // horizontal
    for (int i = 0; i < num_blocks - 1; i++) {
        for (int j = 0; j < num_blocks; j++) {
            BlockMesh *m = &blocks[i][j];
            m->ifaces[2] = new MeshInterface(
                m->corners[3], 
                m->corners[2], 
                num_seg_per_block, 
                global_ind, 
                true
            );
            blocks[i + 1][j].ifaces[0] = m->ifaces[2];
            global_ind = m->ifaces[2]->get_max_inner_idx() + 1;
        }
    }

    max_global_iface_ind = global_ind - 1;
}

bool DistributedMesh::is_on_bnd(int i, int j, int num_blocks) {
    return i == 0 || i == num_blocks - 1 || j == 0 || j == num_blocks - 1;
}

int DistributedMesh::get_next_idx() {
    return max_global_iface_ind + 1;
}
