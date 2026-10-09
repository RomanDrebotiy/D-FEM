#include "per_block_global.h"
#include "../mesh/geometry.h"
#include "local.h"
#include <utility>
#include <vector>
#include <array>
#include <algorithm>

#include <mpi.h>
#include <CL/cl.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Here we manage per-block parts of global system without any MPI sync. Syncing we will do in other file.

bool USE_OPENCL = false;

static cl_context opencl_context = nullptr;
static cl_command_queue opencl_queue = nullptr;
static cl_program opencl_program = nullptr;
static cl_kernel opencl_kernel = nullptr;

static cl_mem opencl_row_idx = nullptr;
static cl_mem opencl_col_idx = nullptr;
static cl_mem opencl_vals = nullptr;

static cl_mem opencl_x_buffer = nullptr;
static cl_mem opencl_y_buffer = nullptr;

static int opencl_matrix_size = 0;

void init_opencl(GlobalMatrix& m) {
    cl_int err;
    opencl_matrix_size = m.get_size();
    cl_platform_id platform = nullptr;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OpenCL: failed to find platform");
    }

    cl_device_id device = nullptr;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OpenCL: failed to find GPU");
    }

    opencl_context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OpenCL: failed to create context");
    }

    opencl_queue = clCreateCommandQueue(opencl_context, device, 0, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OpenCL: failed to create command queue");
    }

    std::ifstream file("fem/spmv.cl");
    if (!file) {
        throw std::runtime_error("OpenCL: cannot open fem/spmv.cl");
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();
    const char* source_ptr = source.c_str();
    size_t source_size = source.size();

    opencl_program = clCreateProgramWithSource(opencl_context, 1, &source_ptr, &source_size, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OpenCL: failed to create program");
    }

    err = clBuildProgram(opencl_program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(opencl_program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        std::string log(log_size, '\0');
        clGetProgramBuildInfo(opencl_program, device, CL_PROGRAM_BUILD_LOG,  log_size, log.data(), nullptr);

        std::cerr << "OpenCL compilation error:\n" << log << std::endl;

        throw std::runtime_error("OpenCL: failed to build program");
    }

    opencl_kernel = clCreateKernel(opencl_program, "csr_spmv", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OpenCL: failed to create kernel");
    }

    opencl_row_idx = clCreateBuffer(opencl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m.row_idx.size() * sizeof(int), m.row_idx.data(), &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "OpenCL: failed to create row_idx buffer");
    }

    opencl_col_idx = clCreateBuffer(opencl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m.col_idx.size() * sizeof(int), m.col_idx.data(), &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OpenCL: failed to create col_idx buffer");
    }

    opencl_vals = clCreateBuffer(opencl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m.vals.size() * sizeof(double), m.vals.data(),  &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OpenCL: failed to create vals buffer");
    }

    opencl_x_buffer = clCreateBuffer(opencl_context, CL_MEM_READ_ONLY, m.get_size() * sizeof(double), nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create x buffer");
    }

    opencl_y_buffer = clCreateBuffer(opencl_context, CL_MEM_WRITE_ONLY, m.get_size() * sizeof(double), nullptr, &err);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create y buffer");
    }

    std::cout << "OpenCL initialized successfully\n";
}

void sort(std::array<std::pair<int, double>, 7>& arr) {
    std::sort(arr.begin(), arr.end());
}

std::vector<std::pair<int, double>> col_vals(std::array<std::pair<int, double>, 7>& arr) {
    std::vector<std::pair<int, double>> res;
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i].first != -1) {
            res.push_back(arr[i]);
        }
    }
    return res;
}

bool add_el(std::array<std::pair<int, double>, 7>& arr, int ind, double val) {
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
    // each entry corresponds to the row and stores unsorted pairs (col, val). Not greater than 7 for the regular mesh.
    std::vector<std::array<std::pair<int, double>, 7>> matr;

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

void gmv_dest_cpu(GlobalMatrix& m, GlobalVector& v, GlobalVector& res) {
    for (int i = 0; i < m.get_size(); i++) {
        res.vals[i] = 0;
        for (int j = m.row_idx[i]; j < m.row_idx[i+1]; j++) {
            res.vals[i] += m.vals[j] * v.vals[m.col_idx[j]];
        }
    }
    res.iface_boundaries = v.iface_boundaries;
}

void gmv_dest_opencl(GlobalVector& v, GlobalVector& res) {
    cl_int err;
    const size_t n = static_cast<size_t>(opencl_matrix_size);
    const size_t bytes = n * sizeof(double);

    err = clEnqueueWriteBuffer(opencl_queue, opencl_x_buffer, CL_TRUE, 0, bytes, v.vals.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to upload x vector");
    }

    err = clSetKernelArg(opencl_kernel, 0, sizeof(cl_mem), &opencl_row_idx);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set row_idx argument");
    }

    err = clSetKernelArg(opencl_kernel, 1, sizeof(cl_mem), &opencl_col_idx);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set col_idx argument");
    }

    err = clSetKernelArg(opencl_kernel, 2, sizeof(cl_mem), &opencl_vals);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set vals argument");
    }

    err = clSetKernelArg(opencl_kernel, 3, sizeof(cl_mem), &opencl_x_buffer);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set x argument");
    }

    err = clSetKernelArg(opencl_kernel, 4, sizeof(cl_mem), &opencl_y_buffer);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set y argument");
    }

    const size_t global_size = n;
    err = clEnqueueNDRangeKernel(opencl_queue, opencl_kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to launch SpMV kernel");
    }

    err = clEnqueueReadBuffer(opencl_queue, opencl_y_buffer, CL_TRUE, 0, bytes, res.vals.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to download y vector");
    }

    res.iface_boundaries = v.iface_boundaries;
}

void gmv_dest(GlobalMatrix& m, GlobalVector& v, GlobalVector& res) {
    double start = MPI_Wtime();

    if (USE_OPENCL) {
        gmv_dest_opencl(v, res);
    } else {
        gmv_dest_cpu(m, v, res);
    }

    double elapsed = MPI_Wtime() - start;
    printf("TIME >> GMV %.6f\n", elapsed);
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
