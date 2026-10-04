#pragma once

#include "geometry.h"
#include <utility>

int map_block_to_rank(int i, int j, int num_blocks);

std::pair<int, int> map_rank_to_block(int rank, int num_blocks);

BlockMesh distribute_and_build_mesh(int rank, int size, int num_seg_per_block);
