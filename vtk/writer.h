#pragma once

#include "../mesh/geometry.h"
#include "../fem/per_block_global.h"

int write_vtk(BlockMesh& bm, GlobalVector& v, int rank, int size);