#pragma once

#include "per_block_global.h"

GlobalVector gmv_synced(GlobalMatrix& m, GlobalVector& v, int rank, int size);

void gmv_synced_dest(GlobalMatrix& m, GlobalVector& v, int rank, int size, GlobalVector& res);

GlobalVector prodv_compwise_synced(GlobalVector& v, GlobalVector& w, int rank, int size);

void prodv_compwise_synced_dest(GlobalVector& v, GlobalVector& w, int rank, int size, GlobalVector& res);

double dot_synced(GlobalVector& v, GlobalVector& w, int rank, int size);

double norm_synced(GlobalVector& v, int rank, int size);
