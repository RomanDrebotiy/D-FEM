#include "per_block_global.h"
#include "../mesh/geometry.h"
#include "local.h"
#include <utility>
#include <vector>
#include <array>
#include <algorithm>

// Here we manage per-block parts of global system without any MPI sync. Syncing we will do in other file.

void sort(std::array<std::pair<int, double>, 6>& arr) {
    std::sort(arr.begin(), arr.end());
}

std::vector<std::pair<int, double>> col_vals(std::array<std::pair<int, double>, 6>& arr) {
    std::vector<std::pair<int, double>> res;
    for (int i = 0; i < 6; i++) {
        if (arr[i].first != -1) {
            res.push_back(arr[i]);
        }
    }
    return res;
}

bool add_el(std::array<std::pair<int, double>, 6>& arr, int ind, double val) {
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i].first == -1) {
            arr[i].first = ind;
            arr[i].second = val;
            return true;
        }

        if (arr[i].first == ind) {
            arr[i].second += val;
            return false;
        }
    }
    return false;
}

struct TempGlobalMatr {
    // each entry corresponds to the row and stores unsorted pairs (col, val). Not greater than 6 for the regular mesh.
    std::vector<std::array<std::pair<int, double>, 6>> matr;

    TempGlobalMatr(int n) : matr(n) {
        for (auto& row : matr) {
            row.fill({-1, 0.0});
        }
    }
};

GlobalSystem assemble(BlockMesh& bm) {
    int size = bm.get_matr_size();
    GlobalSystem gs(size);
    TempGlobalMatr temp_matr(size);
    int nonzero = 0;
    for (size_t i = 0; i < bm.triangles.size(); i++) {
        Triangle t = bm.triangles[i];
        LocalMatrix lm = get_local_matrix(t);
        LocalVector lv = get_local_vector(t);
        for (int p = 0; p < 3; p++) {
            int gp = t.verts[p]->block_local_idx;
            for (int q = 0; q < 3; q++) {
                int gq = t.verts[q]->block_local_idx;
                bool is_new = add_el(temp_matr.matr[gp], gq, lm.el[p][q]);
                if (is_new) {
                    nonzero++;
                }
            }
            gs.vec.vals[gp] += lv.el[p];
        }
    }

    // Now generate CSR
    gs.matr.col_idx.resize(nonzero);
    gs.matr.vals.resize(nonzero);
    int nxt_row_ind = 0;
    gs.matr.row_idx[0] = nxt_row_ind;
    for (int i = 0; i < size; i++) {
        sort(temp_matr.matr[i]);
        std::vector<std::pair<int, double>> cv = col_vals(temp_matr.matr[i]);
        for (size_t j = 0; j < cv.size(); j++) {
            gs.matr.col_idx[nxt_row_ind + j] = cv[j].first;
            gs.matr.vals[nxt_row_ind + j] = cv[j].second;
        }
        nxt_row_ind += cv.size();
        gs.matr.row_idx[i + 1] = nxt_row_ind;
    }

    // and generate interface mappings for further MPI sync operations
    // We skip here corner values, since they will be synced separately to avoid duplication
    int iface_start = 4; // first are corner vars
    // for matrix we will not use it for now. We will keep matrix without sync since it is not needed and we can do syncing only
    // on vectors after multiplication
    bool skip_iface_maps = true;
    gs.vec.iface_boundaries[0] = iface_start;
    for (int i = 0; i < 4; i++) {
        int iface_end = iface_start + bm.ifaces[i]->nodes.size() - 2;
        if (!skip_iface_maps) {
            for (int row = iface_start; row < iface_end; row++) {
                for (int col = gs.matr.row_idx[row]; col < gs.matr.row_idx[row + 1]; col++) {
                    int real_col = gs.matr.col_idx[col];
                    if (real_col >= iface_start && real_col < iface_end) {
                        gs.matr.iface_idx.mapping[i].push_back(col);
                    }
                }
            }
        }
        gs.vec.iface_boundaries[i + 1] = iface_end;
        iface_start = iface_end;
    }
    
    return gs;
}

// A * x
GlobalVector gmv(GlobalMatrix& m, GlobalVector& v) {
    GlobalVector res(v.get_size());
    gmv_dest(m, v, res);
    return res;
}

void gmv_dest(GlobalMatrix& m, GlobalVector& v, GlobalVector& res) {
    for (int i = 0; i < m.get_size(); i++) {
        res.vals[i] = 0;
        for (int j = m.row_idx[i]; j < m.row_idx[i+1]; j++) {
            res.vals[i] += m.vals[j] * v.vals[m.col_idx[j]];
        }
    }
    res.iface_boundaries = v.iface_boundaries;
}

GlobalVector addv(GlobalVector& v, GlobalVector& w) {
    GlobalVector res(v.get_size());
    addv_dest(v, w, res);
    return res;
}

void addv_dest(GlobalVector& v, GlobalVector& w, GlobalVector& res) {
    for (size_t i = 0; i < res.get_size(); i++) {
        res.vals[i] = v.vals[i] + w.vals[i];
    }
    res.iface_boundaries = v.iface_boundaries;
}

GlobalVector subv(GlobalVector& v, GlobalVector& w) {
    GlobalVector res(v.get_size());
    subv_dest(v, w, res);
    return res;
}

void subv_dest(GlobalVector& v, const GlobalVector& w, GlobalVector& res) {
    for (size_t i = 0; i < res.get_size(); i++) {
        res.vals[i] = v.vals[i] - w.vals[i];
    }
    res.iface_boundaries = v.iface_boundaries;
}

GlobalVector prodv_compwise(GlobalVector& v, GlobalVector& w) {
    GlobalVector res(v.get_size());
    prodv_compwise_dest(v, w, res);
    return res;
}

void prodv_compwise_dest(GlobalVector& v, GlobalVector& w, GlobalVector& res) {
    for (size_t i = 0; i < res.get_size(); i++) {
        res.vals[i] = v.vals[i] * w.vals[i];
    }
    res.iface_boundaries = v.iface_boundaries;
}

void addv_tov(GlobalVector& v, GlobalVector& w) {
    for (size_t i = 0; i < v.get_size(); i++) {
        v.vals[i] += w.vals[i];
    }
}

GlobalVector vm_num(GlobalVector& v, double a) {
    GlobalVector res(v.get_size());
    vm_num_dest(v, a, res);
    return res;
}

void vm_num_dest(GlobalVector& v, double a, GlobalVector& res) {
    for (size_t i = 0; i < res.get_size(); i++) {
        res.vals[i] = v.vals[i] * a;
    }
    res.iface_boundaries = v.iface_boundaries;
}

void vm_num_tov(GlobalVector& v, double a) {
    for (size_t i = 0; i < v.get_size(); i++) {
        v.vals[i] *= a;
    }
}
