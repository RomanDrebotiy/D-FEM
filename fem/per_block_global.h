#pragma once

#include "../mesh/geometry.h"
#include <vector>
#include <cmath>

struct GlobalVector {
    std::vector<double> vals;
    std::array<int, 5> iface_boundaries;

    GlobalVector(int n) {
        vals.resize(n);
    }

    GlobalVector(int n, const GlobalVector& v_ref) {
        vals.resize(n);
        iface_boundaries = v_ref.iface_boundaries;
    }

    size_t get_size() {
        return vals.size();
    }

    void inverse_nonzero_with_eps(double eps) {
        for (size_t i = 0; i < get_size(); i++) {
            if (std::abs(vals[i]) > eps) {
                vals[i] = 1.0 / vals[i];
            } else {
                vals[i] = 1.0;
            }
        }
    }
};

struct IfaceToBlockLocalMatrIdx {
    std::array<std::vector<int>, 4> mapping;
};

struct GlobalMatrix {
    // CSR format
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<double> vals;
    IfaceToBlockLocalMatrIdx iface_idx;

    GlobalMatrix(int n) {
        row_idx.resize(n + 1);
    }

    int get_size() {
        return row_idx.size() - 1;
    }

    GlobalVector get_diagonal(const GlobalVector& ref) {
        GlobalVector res(get_size(), ref);
        for (int i = 0; i < get_size(); i++) {
            res.vals[i] = 0;
            for (int j = row_idx[i]; j < row_idx[i+1]; j++) {
                if (col_idx[j] == i) {
                    res.vals[i] = vals[j];
                }
            }
        }
        return res;
    }
};

struct GlobalSystem {
    GlobalMatrix matr;
    GlobalVector vec;

    GlobalSystem(int n) : matr(n), vec(n) {

    }

    int get_size() {
        return vec.get_size();
    }
};

GlobalSystem assemble(BlockMesh& bm);

GlobalVector gmv(GlobalMatrix& m, GlobalVector& v);

void gmv_dest(GlobalMatrix& m, GlobalVector& v, GlobalVector& res);

GlobalVector addv(GlobalVector& v, GlobalVector& w);

void addv_dest(GlobalVector& v, GlobalVector& w, GlobalVector& res);

GlobalVector subv(GlobalVector& v, GlobalVector& w);

void subv_dest(GlobalVector& v, const GlobalVector& w, GlobalVector& res);

GlobalVector prodv_compwise(GlobalVector& v, GlobalVector& w);

void prodv_compwise_dest(GlobalVector& v, GlobalVector& w, GlobalVector& res);

void addv_tov(GlobalVector& v, GlobalVector& w);

GlobalVector vm_num(GlobalVector& v, double a);

void vm_num_dest(GlobalVector& v, double a, GlobalVector& res);

void vm_num_tov(GlobalVector& v, double a);
