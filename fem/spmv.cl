__kernel void csr_spmv(
    __global const int* row_idx,
    __global const int* col_idx,
    __global const double* vals,
    __global const double* x,
    __global double* y
) {
    int row = get_global_id(0);

    double sum = 0.0;

    for (int j = row_idx[row]; j < row_idx[row + 1]; j++) {
        sum += vals[j] * x[col_idx[j]];
    }

    y[row] = sum;
}
