#pragma once

#include "geometry.h"


std::vector<std::byte> serialize_partial_block_mesh(const BlockMesh& bm);

BlockMesh deserialize_partial_block_mesh(const std::vector<std::byte>& res);
