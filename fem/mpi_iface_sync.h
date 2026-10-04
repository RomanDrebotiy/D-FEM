#pragma once

#include "per_block_global.h"

struct InnerIfaceSyncMeta {
    std::vector<int> ifaces_active;
    std::vector<int> ranks_to_sync;
};

struct CornerSyncMeta {
    std::array<std::vector<int>, 4> ranks_to_sync;
};

struct SyncMeta {
    InnerIfaceSyncMeta ifaces;
    CornerSyncMeta corners;
};

SyncMeta get_sync_meta(int rank, int size);

void sync_vector(GlobalVector& v, int rank, int size);

void sync_matrix(GlobalMatrix& v, int rank, int size);