#pragma once

#include "../fem/per_block_global.h"

GlobalVector pcg(GlobalSystem gs, double eps, int rank, int size, int max_iter);