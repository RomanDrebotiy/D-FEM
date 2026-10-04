#include "geometry.h"
#include "geometry_serializer.h"
#include <cstring>


std::vector<std::byte> serialize_partial_block_mesh(const BlockMesh& bm) {
    int corners_size = 4 * sizeof(Node);
    int iface_size = bm.ifaces[0]->nodes.size() * sizeof(Node) + sizeof(bool);
    int bnd_markers_size = 4 * sizeof(bool);
    int size = sizeof(int) + corners_size + bm.ifaces.size() * iface_size + bnd_markers_size;

    std::vector<std::byte> res(size);
    std::memcpy(res.data(), &bm.num_seg_per_block, sizeof(int));
    int offset = sizeof(int);
    std::memcpy(res.data() + offset, bm.corners.data(), corners_size);
    offset += corners_size;
    for (int i = 0; i < 4; i++) {
        std::memcpy(res.data() + offset, bm.ifaces[i]->nodes.data(), iface_size - sizeof(bool));
        offset += iface_size - sizeof(bool);
        std::memcpy(res.data() + offset, &bm.ifaces[i]->active, sizeof(bool));
        offset += sizeof(bool);
    }
    std::memcpy(res.data() + offset, bm.boundary_markers.data(), bnd_markers_size);

    return res;
}

BlockMesh deserialize_partial_block_mesh(const std::vector<std::byte>& res) {
    BlockMesh bm;
    std::memcpy(&bm.num_seg_per_block, res.data(), sizeof(int));
    int offset = sizeof(int);
    std::memcpy(bm.corners.data(), res.data() + offset, 4 * sizeof(Node));
    offset += 4 * sizeof(Node);
    for (int i = 0; i < 4; i++) { 
        bm.ifaces[i] = new MeshInterface();
        bm.ifaces[i]->nodes.resize(bm.num_seg_per_block + 1);
        std::memcpy(bm.ifaces[i]->nodes.data(), res.data() + offset, (bm.num_seg_per_block + 1) * sizeof(Node));
        offset += (bm.num_seg_per_block + 1) * sizeof(Node);
        std::memcpy(&bm.ifaces[i]->active, res.data() + offset, sizeof(bool));
        offset += sizeof(bool);
    }

    std::memcpy(bm.boundary_markers.data(), res.data() + offset, 4 * sizeof(bool));

    return bm;
}
