#include <iostream>
#include <iomanip>
#include <tuple>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cusolver_utils.h"
#include "cusolver_csrqr.cuh"
#include <helper_cuda.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include <time.h>

using namespace std;

#define tol_diff_fea 2
#define tol_diff 1


__global__ void initialise_c(double **dd_c_copy, double *d_c_copy, bool **dd_comb_bool, int *d_lp_id_dict, int n_lps, int N, int K, int n_c) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_lps * N) return;
    int lp_idx = tid / N;
    int elem_idx = tid % N;
    if (elem_idx >= K && elem_idx < K + n_c) {
        dd_c_copy[lp_idx][elem_idx] = (double) dd_comb_bool[d_lp_id_dict[lp_idx]][elem_idx - K];
    } else {
        dd_c_copy[lp_idx][elem_idx] = 0;
    }
}

__global__ void initialise_cb(int **dd_b, double **dd_c_copy, double **dd_cb, int n_active_lps, int M, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_active_lps * M) return;
    int lp_idx = tid / M;
    int elem_idx = tid % M;
    dd_cb[lp_idx][elem_idx] = dd_c_copy[lp_idx][dd_b[lp_idx][elem_idx]];
}

__global__ void initialise_is_b_cb(int **dd_b, int **dd_is_b, double **dd_c_copy, double **dd_cb, int n_active_lps, int M, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_active_lps * M) return;
    int lp_idx = tid / M;
    int elem_idx = tid % M;
    dd_is_b[lp_idx][dd_b[lp_idx][elem_idx]] = 1;
    dd_cb[lp_idx][elem_idx] = dd_c_copy[lp_idx][dd_b[lp_idx][elem_idx]];
}

__global__ void initialise_is_b_cb_2(int **dd_b, int *d_is_b, int **dd_is_b, double *d_c_copy, double **dd_cb, int n_active_lps, int M, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_active_lps * M) return;
    int lp_idx = tid / M;
    int elem_idx = tid % M;
    if (!elem_idx) dd_is_b[lp_idx] = d_is_b + N * lp_idx;
    d_is_b[lp_idx * N + dd_b[lp_idx][elem_idx]] = 1;
    dd_cb[lp_idx][elem_idx] = d_c_copy[lp_idx * N + dd_b[lp_idx][elem_idx]];
}

__global__ void check_invertible_kernel_dual_2(int *d_infoArray, int *d_init_lp_idxs, int *d_lp_id_dict, int *d_is_invertible, double **dd_B_copy, double **dd_A, int **dd_b, int **dd_non_b,
                                               int *d_b_init, int *d_non_b_init, int n_active_lps, int M, int N, int K)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j;
    if (tid >= n_active_lps) return;
    if (d_infoArray[tid]) {

        for (i = 0; i < M; i++) {
            dd_b[tid][i] = d_b_init[i];
            for (j = 0; j < M; j++) {
                dd_B_copy[tid][i * M + j] = dd_A[tid][d_b_init[i] * M + j];
            }
        }
        for (i = 0; i < K; i++) {
            dd_non_b[tid][i] = d_non_b_init[i];
        }
    }
}

__global__ void initialise_dd(double *d_A, double **dd_A, double *d_B, double **dd_B, double *d_B_copy, double **dd_B_copy,
            int *d_b, int **dd_b, int *d_is_b, int **dd_is_b, double *d_c, double **dd_c, double *d_c_copy, double **dd_c_copy,
            double *d_cb, double **dd_cb, double *d_xb, double **dd_xb, double *d_col, double **dd_col, int n_lps, int M, int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_lps) return;
    dd_A[tid] = d_A + tid * M * N;
    dd_B[tid] = d_B + tid * M * M;
    dd_B_copy[tid] = d_B_copy + tid * M * M;
    dd_b[tid] = d_b + tid * M;
    dd_is_b[tid] = d_is_b + tid * N;
    dd_c[tid] = d_c + tid * N;
    dd_c_copy[tid] = d_c_copy + tid * N;
    dd_cb[tid] = d_cb + tid * M;
    dd_xb[tid] = d_xb + tid * M;
    dd_col[tid] = d_col + tid * M;
}

__global__ void initialise_dd_2_1(double *d_A, double **dd_A, double *d_B, double **dd_B, double *d_B_copy, double **dd_B_copy,
            int *d_b, int **dd_b, int *d_non_b, int **dd_non_b, double *d_c, double **dd_c, double *d_c_copy, double **dd_c_copy,
            double *d_cb, double **dd_cb, double *d_xb, double **dd_xb, double *d_col, double **dd_col, double *d_cols, double **dd_cols,
            int *d_b_counter, int **dd_b_counter, int *d_non_b_counter, int **dd_non_b_counter,
            double *d_z_final, int *d_opt_test, int *d_is_invertible, int *d_lp_idxs, int n_lps, int M, int N, int K)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_lps) return;
    dd_A[tid] = d_A + tid * M * N;
    dd_B[tid] = d_B + tid * M * M;
    dd_B_copy[tid] = d_B_copy + tid * M * M;
    dd_b[tid] = d_b + tid * M;
    dd_non_b[tid] = d_non_b + tid * K;
    dd_c[tid] = d_c + tid * N;
    dd_c_copy[tid] = d_c_copy + tid * N;
    dd_cb[tid] = d_cb + tid * M;
    dd_xb[tid] = d_xb + tid * M;
    dd_col[tid] = d_col + tid * M;
    dd_cols[tid] = d_cols + tid * M * K;
    dd_b_counter[tid] = d_b_counter + tid * N;
    dd_non_b_counter[tid] = d_non_b_counter + tid * N;
    d_z_final[tid] = 100;
    d_opt_test[tid] = 1;
    d_is_invertible[tid] = 1;
    d_lp_idxs[tid] = tid;
}

__global__ void initialise_init_lp_idxs(int *d_init_lp_idxs, int n_lps) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_lps) return;
    d_init_lp_idxs[tid] = tid;
}

__global__ void initialise_init_lp_idxs_2(int *d_init_lp_idxs, int n_lps, int **dd_b_counter, int N_) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_lps) return;
    d_init_lp_idxs[tid] = tid;

    for (int i = 0; i < N_; i++) {
        dd_b_counter[tid][i] = 0;
    }
}

__global__ void perturbation_kernel(double *d_xb, double *d_xb_copy, int n_cycles_same_lps, int n_active_lps, int M) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_active_lps * M) return;

    int lp_idx = tid / M;
    int elem_idx = tid % M;

    d_xb[lp_idx * M + elem_idx] = d_xb_copy[elem_idx] + (elem_idx + 1) * 1e-6 * n_cycles_same_lps;
}

__global__ void dual_min_idx_kernel(double **dd_xb, int *d_min_idxs, double *d_min_vals, int *d_th_idxs, bool *d_go_on, double **dd_cb,
                                    double *d_z_final, int *d_init_lp_idxs, int *d_lp_id_dict, int **dd_b, int **dd_non_b, int **dd_b_counter, bool *d_is_feasible,
                                    int *d_infoArray, int n_cycles_same_lps, int M, int M_2, int N, int N_, double dual_feasible_tol, double dual_stop_tol)  // const 
{
    if (!d_go_on[blockIdx.x] || d_infoArray[blockIdx.x]) return;

    extern __shared__ char array[];
    double *vals = (double *)array;
    int *idxs = (int *)(M * sizeof(double) + array);

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    vals[threadIdx.x] = dd_xb[blockIdx.x][threadIdx.x];
    idxs[threadIdx.x] = threadIdx.x;

    __syncthreads();

    for (int i = M_2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i && threadIdx.x + i < M)
        {
            if (n_cycles_same_lps % 20 < 10)
            {
                if (vals[threadIdx.x] > vals[threadIdx.x + i]
                    || (vals[threadIdx.x] > vals[threadIdx.x + i] - 1e-10
                        && dd_b_counter[d_init_lp_idxs[blockIdx.x]][dd_b[blockIdx.x][idxs[threadIdx.x]]] > dd_b_counter[d_init_lp_idxs[blockIdx.x]][dd_b[blockIdx.x][idxs[threadIdx.x + i]]]))
                {
                    vals[threadIdx.x] = vals[threadIdx.x + i];
                    idxs[threadIdx.x] = idxs[threadIdx.x + i]; 
                }

            } else {
                if ((vals[threadIdx.x] > vals[threadIdx.x + i] + 1e-10
                        && dd_b_counter[d_init_lp_idxs[blockIdx.x]][dd_b[blockIdx.x][idxs[threadIdx.x]]] > dd_b_counter[d_init_lp_idxs[blockIdx.x]][dd_b[blockIdx.x][idxs[threadIdx.x + i]]] - 10)
                    || (vals[threadIdx.x] > vals[threadIdx.x + i]
                        && dd_b_counter[d_init_lp_idxs[blockIdx.x]][dd_b[blockIdx.x][idxs[threadIdx.x]]] > dd_b_counter[d_init_lp_idxs[blockIdx.x]][dd_b[blockIdx.x][idxs[threadIdx.x + i]]] - 5)
                    || (vals[threadIdx.x + i] < dual_stop_tol && vals[threadIdx.x] < dual_stop_tol
                        && dd_b_counter[d_init_lp_idxs[blockIdx.x]][dd_b[blockIdx.x][idxs[threadIdx.x]]] > dd_b_counter[d_init_lp_idxs[blockIdx.x]][dd_b[blockIdx.x][idxs[threadIdx.x + i]]] + 10))
                {
                    vals[threadIdx.x] = vals[threadIdx.x + i];
                    idxs[threadIdx.x] = idxs[threadIdx.x + i];
                }
            }
        }
        __syncthreads();
    }

    if (!threadIdx.x){
        d_min_vals[blockIdx.x] = vals[0];
        d_min_idxs[d_init_lp_idxs[blockIdx.x]] = idxs[0];
        dd_b_counter[d_init_lp_idxs[blockIdx.x]][dd_b[blockIdx.x][idxs[0]]] += 1;

        d_go_on[blockIdx.x] = false;
    }

    __syncthreads();
    if (!d_go_on[blockIdx.x] && dd_xb[blockIdx.x][threadIdx.x] < dual_stop_tol) {
        d_go_on[blockIdx.x] = true;
    }

    __syncthreads();

    if (d_go_on[blockIdx.x]) return;

    vals[threadIdx.x] = dd_cb[blockIdx.x][threadIdx.x] * dd_xb[blockIdx.x][threadIdx.x];

    __syncthreads();
    for (int i = M_2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i && threadIdx.x + i < M)
            vals[threadIdx.x] += vals[threadIdx.x + i];
        __syncthreads();
    }

    if (!threadIdx.x) {
        d_z_final[d_init_lp_idxs[blockIdx.x]] = vals[0];
        if (vals[0] > -dual_feasible_tol && vals[0] < dual_feasible_tol) {
            d_is_feasible[d_init_lp_idxs[blockIdx.x]] = true;
        }
    }

    __syncthreads();
    if (d_is_feasible[d_init_lp_idxs[blockIdx.x]] && idxs[threadIdx.x] >= N_) {
        d_is_feasible[d_init_lp_idxs[blockIdx.x]] = false;
    }
}

__global__ void prepare_phase2_kernel(int n_feasible_lps, int *d_idxs, int *d_b, int **dd_b, double *d_A, double **dd_A, 
                                      double *d_B_copy, double **dd_B_copy, double *d_c, double *d_c_copy, double **dd_c, double *d_cb, double **dd_cb, int M, int N, int N_)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_feasible_lps) return;
    dd_b[tid] = d_b + M * d_idxs[tid];
    dd_A[tid] = d_A + M * N * d_idxs[tid];
    dd_B_copy[tid] = d_B_copy + M * M * d_idxs[tid];
    d_c_copy[tid * N_ + N - M - 2] = -1;
    dd_c[tid] = d_c + N_ * tid;
    dd_cb[tid] = d_cb + M * tid;
}

__global__ void min_idx_kernel(double **dd_c, int **dd_is_b, int *d_min_idxs, double *d_min_vals, bool *d_go_on, double **dd_cb, double **dd_xb,
                                 double *d_z_final, bool *d_feasibilities, int *d_init_lp_idxs, int M, int N, int M_2, int N_2, double tol_minus_1)  // const 
{
    if (!d_go_on[blockIdx.x]) return;

    extern __shared__ char array[];

    double *vals = (double *)array;
    int *idxs = (int *)(N * sizeof(double) + array);

    if (dd_is_b[blockIdx.x][threadIdx.x]) {
        vals[threadIdx.x] = 100000.0;
    } else {
        vals[threadIdx.x] = dd_c[blockIdx.x][threadIdx.x];
    }

    idxs[threadIdx.x] = threadIdx.x;

    __syncthreads();
    
    for (int i = N_2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i && threadIdx.x + i < N)
            if (vals[threadIdx.x] > vals[threadIdx.x + i] || (vals[threadIdx.x] == vals[threadIdx.x + i] && idxs[threadIdx.x] > idxs[threadIdx.x + i]))
            {
                vals[threadIdx.x] = vals[threadIdx.x + i];
                idxs[threadIdx.x] = idxs[threadIdx.x + i];
            }
        __syncthreads();
    }
    if (!threadIdx.x){
        d_min_vals[blockIdx.x] = vals[threadIdx.x];
        d_min_idxs[blockIdx.x] = idxs[threadIdx.x];

        if (vals[threadIdx.x] > tol_minus_1) {
            d_go_on[blockIdx.x] = false;
        }
    }

    __syncthreads();

    if (d_go_on[blockIdx.x] || threadIdx.x >= M) return;

    vals[threadIdx.x] = dd_cb[blockIdx.x][threadIdx.x] * dd_xb[blockIdx.x][threadIdx.x];

    __syncthreads();
    for (int i = M_2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i && threadIdx.x + i < M)
            vals[threadIdx.x] += vals[threadIdx.x + i];
        __syncthreads();
    }
    if (!threadIdx.x) {
        d_z_final[d_init_lp_idxs[blockIdx.x]] = vals[0];
        if (vals[0] >= -1e-12 && vals[0] <= 1e-12) {
            d_feasibilities[d_init_lp_idxs[blockIdx.x]] = 1;
        } else {
            d_feasibilities[d_init_lp_idxs[blockIdx.x]] = 0;
        }
    }
}

__global__ void min_idx_kernel_optimal(double **dd_c, int **dd_is_b, int *d_min_idxs, double *d_min_vals, bool *d_go_on, double **dd_cb, double **dd_xb,
                               double *d_z_final, int **dd_b_counter, int *d_init_lp_idxs, int M, int N, int M_2, int N_2, double tol_minus_optimal)
{
    if (!d_go_on[blockIdx.x]) return;

    extern __shared__ char array[];

    double *vals = (double *)array;
    int *idxs = (int *)(N * sizeof(double) + array);

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (dd_is_b[blockIdx.x][threadIdx.x]) {
        vals[threadIdx.x] = 100000.0;
    } else {
        vals[threadIdx.x] = dd_c[blockIdx.x][threadIdx.x];
    }

    idxs[threadIdx.x] = threadIdx.x;

    __syncthreads();
    
    for (int i = N_2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i && threadIdx.x + i < N)
            if (vals[threadIdx.x] > vals[threadIdx.x + i] || (dd_b_counter[d_init_lp_idxs[blockIdx.x]][idxs[threadIdx.x]] > dd_b_counter[d_init_lp_idxs[blockIdx.x]][idxs[threadIdx.x + i]]
            && vals[threadIdx.x + i] < -1e-10))
            {
                vals[threadIdx.x] = vals[threadIdx.x + i];
                idxs[threadIdx.x] = idxs[threadIdx.x + i];
            }
        __syncthreads();
    }
    if (!threadIdx.x){
        // printf("%d, %d, %f\t\t", threadIdx.x, idxs[threadIdx.x], vals[threadIdx.x]);
        d_min_vals[blockIdx.x] = vals[threadIdx.x];
        d_min_idxs[blockIdx.x] = idxs[threadIdx.x];

        dd_b_counter[d_init_lp_idxs[blockIdx.x]][idxs[threadIdx.x]] += 1;

        if (vals[threadIdx.x] > tol_minus_optimal) {
            d_go_on[blockIdx.x] = false;
        }
    }

    __syncthreads();

    if (threadIdx.x >= M) return;

    int tid = threadIdx.x + blockIdx.x * M;

    vals[threadIdx.x] = dd_cb[blockIdx.x][threadIdx.x] * dd_xb[blockIdx.x][threadIdx.x];

    __syncthreads();
    for (int i = M_2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i && threadIdx.x + i < M)
            vals[threadIdx.x] += vals[threadIdx.x + i];
        __syncthreads();
    }
    if (!threadIdx.x) {
        if (d_go_on[blockIdx.x] == false) {
            d_z_final[d_init_lp_idxs[blockIdx.x]] = vals[0];
        }
    }
}

__global__ void copy_col_kernel(double **dd_A, double **dd_col, int *d_min_idxs, const int out_size, int M)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int mat_idx = tid / M;
    int row_idx = tid % M;
    if (tid >= out_size) return;

    dd_col[mat_idx][row_idx] = dd_A[mat_idx][d_min_idxs[mat_idx] * M + row_idx];
}

__global__ void copy_multi_cols_kernel(double **dd_A, double **dd_cols, int **dd_non_b, const int out_size, int M, int K)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int mat_idx = tid / K;
    int col_idx = tid % K;
    if (tid >= out_size) return;
    
    int idx = 0;
    for (int i = 0; i < M; i++) {
        dd_cols[mat_idx][M*col_idx + i] = dd_A[mat_idx][M*dd_non_b[mat_idx][col_idx] + i];
    }
}

__global__ void copy_B_kernel(double **dd_B, double **dd_B_copy, int n_active_lps, int MM)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_lps * MM) return;
    int lp_idx = tid / MM;
    int elem_idx = tid % MM;

    dd_B[lp_idx][elem_idx] = dd_B_copy[lp_idx][elem_idx];
}

__global__ void copy_cb_kernel_1(double **dd_col, double **dd_cb, int n_active_lps, int M)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_lps * M) return;
    int lp_idx = tid / M;
    int elem_idx = tid % M;

    dd_col[lp_idx][elem_idx] = dd_cb[lp_idx][elem_idx];
}

__global__ void copy_c_kernel_1(double **dd_c, double **dd_c_copy, int n_active_lps, int N)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_lps * N) return;
    int lp_idx = tid / N;
    int elem_idx = tid % N;

    dd_c[lp_idx][elem_idx] = dd_c_copy[lp_idx][elem_idx];
}

__global__ void move_kernel_1(int n_active_lps, int *d_lp_idxs, int*d_init_lp_idxs, int **dd_b, int **dd_is_b,
                                double **dd_A, double **dd_B_copy, double **dd_cb, double **dd_c, double **dd_c_copy,
                                int n_cycles, int M, int N) 
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_lps) return;

    int idx_chg = d_init_lp_idxs[d_lp_idxs[tid]] - d_init_lp_idxs[tid];

    __syncthreads();

    if (idx_chg == 0) {
        d_lp_idxs[tid] = tid;
        return;
    }

    dd_b[tid] += idx_chg * M;
    dd_is_b[tid] += idx_chg * N;

    dd_A[tid] += idx_chg * M * N;
    dd_B_copy[tid] += idx_chg * M * M;
    dd_cb[tid] += idx_chg * M;
    dd_c[tid] += idx_chg * N;
    dd_c_copy[tid] += idx_chg * N;

    __syncthreads();

    d_lp_idxs[tid] = tid;
    d_init_lp_idxs[tid] += idx_chg;
}

__global__ void move_kernel_dual(int n_active_lps, int *d_lp_idxs, int* d_init_lp_idxs, int **dd_b, int **dd_non_b,
                                double **dd_A, double **dd_B, double **dd_B_copy, double **dd_cb,
                                int M, int N, int K)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_lps) return;

    int idx_chg = d_init_lp_idxs[d_lp_idxs[tid]] - d_init_lp_idxs[tid];

    if (idx_chg == 0) {
        d_lp_idxs[tid] = tid;
        return;
    }
    
    dd_b[tid] += idx_chg * M;
    dd_non_b[tid] += idx_chg * K;
    dd_A[tid] += idx_chg * M * N;
    dd_B_copy[tid] += idx_chg * M * M;
    dd_cb[tid] += idx_chg * M;

    __syncthreads();

    d_lp_idxs[tid] = tid;
    d_init_lp_idxs[tid] += idx_chg;
}

__global__ void move_kernel_3(int n_active_lps, int *d_lp_idxs, int *d_idxs, int *d_init_lp_idxs, int **dd_b, int **dd_is_b,
                                double **dd_A, double **dd_B_copy, double **dd_cb,
                                int M, int N, int N_) 
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_lps) return;

    int idx_chg_1 = d_idxs[d_lp_idxs[tid]] - d_idxs[tid];
    int idx_chg_2 = d_init_lp_idxs[d_lp_idxs[tid]] - d_init_lp_idxs[tid];

    if (idx_chg_1 == 0 && idx_chg_2 == 0) {
        d_lp_idxs[tid] = tid;
        return;
    }

    d_idxs[tid] += idx_chg_1;
    d_init_lp_idxs[tid] += idx_chg_2;
    
    dd_b[tid] += idx_chg_1 * M;
    dd_A[tid] += idx_chg_1 * M * N;
    dd_B_copy[tid] += idx_chg_1 * M * M;

    dd_is_b[tid] += idx_chg_2 * N_;
    dd_cb[tid] += idx_chg_2 * M;

    __syncthreads();

    d_lp_idxs[tid] = tid;
}

__global__ void copy_basis_dual_2(int split_idx, int n_copy_lps, int *d_root_idxs, int *d_min_differ,
                                      int *d_b, int *d_non_b, double *d_A, double *d_B_copy, int M, int N, int K, int max_M_K)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= max_M_K * n_copy_lps) return;

    int lp_idx = tid / max_M_K + split_idx;  // For each loop, max_M_K threads compare the difference between current lp with max_M_K simultaneously
    int c_idx = tid % M;

    if (tid < n_copy_lps * M) {
        lp_idx = tid / M + split_idx;
        if (d_root_idxs[lp_idx] >= 0 && d_min_differ[lp_idx] <= tol_diff) {
            d_b[lp_idx * M + c_idx] = d_b[d_root_idxs[lp_idx] * M + c_idx];
            for (int i = 0; i < M; i++) {
                d_B_copy[lp_idx * M * M + c_idx * M + i] = d_A[lp_idx * M * N + d_b[lp_idx * M + c_idx] * M + i];
            }
        }
    }

    if (tid >= n_copy_lps * K) return;
    lp_idx = tid / K + split_idx;
    c_idx = tid % K;

    if (d_root_idxs[lp_idx] >= 0 && d_min_differ[lp_idx] <= tol_diff) {
        d_non_b[lp_idx * K + c_idx] = d_non_b[d_root_idxs[lp_idx] * K + c_idx];
    }
}

__global__ void copy_basis_dual_1(int split_idx, bool **dd_comb_bool, int *d_lp_id_dict, int *d_root_idxs, int *d_min_differ, double *d_z_final, int n_c, int nTPB)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int lp_idx = blockIdx.x + split_idx;  // For each loop, nTPB threads compare the difference between current lp with nTPB simultaneously
    int lp_id = d_lp_id_dict[lp_idx];
    int root_idx = split_idx - 1;  // First lp_idx to be compared with
    int c_idx = 0;
    int n_differ = 0;

    if (threadIdx.x == 0) {
        d_root_idxs[lp_idx] = -1;
        d_min_differ[lp_idx] = n_c;
    }

    __syncthreads();

    while (root_idx >= 0) {
        n_differ = 0;
        for (c_idx = 0; c_idx < n_c; c_idx++) {
            if (dd_comb_bool[d_lp_id_dict[root_idx]][c_idx] != dd_comb_bool[lp_id][c_idx]) {
                n_differ++;
            }
            if (n_differ >= 2) break;
        }
        
        if (n_differ < d_min_differ[lp_idx] && d_z_final[root_idx] > -1e-12 && d_z_final[root_idx] < 1e-12) {
            d_min_differ[lp_idx] = n_differ;
            d_root_idxs[lp_idx] = root_idx;
        }

        if (d_min_differ[lp_idx] <= tol_diff) {
            break;
        }

        root_idx -= nTPB;
    }
}

__global__ void reset_d_go_on_feasi(bool *d_go_on, bool *d_feasibilities, int *d_lp_idxs, int split_idx, int n_copy_lps, int n_active_lps, int n_lps) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_lps) return;
    d_lp_idxs[tid] = tid;

    if (tid < n_active_lps) {
        return;
    } else if (tid < split_idx) {
        d_go_on[tid] = false;
    } else if (tid < split_idx + n_copy_lps && d_feasibilities[tid]) {
        d_go_on[tid] = true;
    } else {
        d_go_on[tid] = false;
    }
}

__global__ void reset_d_go_on(bool *d_go_on, int *d_lp_idxs, int split_idx, int n_copy_lps, int n_active_lps, int n_lps) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_lps) return;
    d_lp_idxs[tid] = tid;

    if (tid < n_active_lps) {
        return;
    } else if (tid < split_idx) {
        d_go_on[tid] = false;
    } else if (tid < split_idx + n_copy_lps) {
        d_go_on[tid] = true;
    } else {
        d_go_on[tid] = false;
    }
}

__global__ void replicate_A_kernel_1(double **dd_A, int n_lps, int MN)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_lps * MN) return;

    int lp_idx = tid / MN;
    int elem_idx = tid % MN;
    
    dd_A[lp_idx][elem_idx] = dd_A[0][elem_idx];
}

__global__ void initialise_A_kernel_1(double **dd_A, bool **dd_comb_bool, int *d_lp_id_dict, int n_c, int n_x, int n_t, int M, int N, int n_lps)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_lps * n_c) return;

    int lp_idx = tid / n_c;
    int j = tid % n_c;

    if (dd_comb_bool[d_lp_id_dict[lp_idx]][j]) {
        dd_A[lp_idx][(n_x + n_t + j) * M + M - n_c + j] = 0;
    } else {
        for (int i=0; i < n_x; i++) {
            dd_A[lp_idx][(n_x + n_t + j) * M + M - n_c - n_x + i] = 0;
        }
    }
}

__global__ void replicate_b_kernel_1(int *d_b, int n_active_lps, int M)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_lps * M) return;

    int lp_idx = tid / M;
    int elem_idx = tid % M;
    
    d_b[lp_idx * M + elem_idx] = d_b[elem_idx];
}

__global__ void replicate_non_b_kernel_1(int *d_non_b, int n_active_lps, int K)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_lps * K) return;

    int lp_idx = tid / K;
    int elem_idx = tid % K;
    
    d_non_b[lp_idx * K + elem_idx] = d_non_b[elem_idx];
}

__global__ void replicate_xb_kernel_1(double *d_xb_copy, int n_lps, int M)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_lps * M) return;

    int lp_idx = tid / M;
    int elem_idx = tid % M;
    
    d_xb_copy[lp_idx * M + elem_idx] = d_xb_copy[elem_idx];
}

__global__ void replicate_c_kernel_1(double *d_c_copy, int *d_b_counter, int *d_non_b_counter, int n_eqs, int n_lps, int N)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_lps * N) return;

    int lp_idx = tid / N;
    int elem_idx = tid % N;
    
    if (elem_idx < N - n_eqs) {
        d_c_copy[lp_idx * N + elem_idx] = 0;
    } else {
        d_c_copy[lp_idx * N + elem_idx] = 1;
    }
    d_b_counter[lp_idx * N + elem_idx] = 0;  // initialise d_b_counter
    d_non_b_counter[lp_idx * N + elem_idx] = 0;  // initialise d_non_b_counter
}

__global__ void get_B_kernel_1(double *d_A, int **dd_b, double **dd_B_copy, int n_active_lps, int M, int MM)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_lps * MM) return;

    int lp_idx = tid / MM;
    int elem_idx = tid % MM;
    int col_idx = elem_idx / M;
    int row_idx = elem_idx % M;

    dd_B_copy[lp_idx][elem_idx] = d_A[dd_b[lp_idx][col_idx] * M + row_idx];  // NOTE: A is the same for all n_lps in feasibility problem
}

__global__ void get_B_kernel_2(double **dd_A, int **dd_b, double **dd_B_copy, int n_active_lps, int M, int MM)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_active_lps * MM) return;

    int lp_idx = tid / MM;
    int elem_idx = tid % MM;
    int col_idx = elem_idx / M;
    int row_idx = elem_idx % M;

    dd_B_copy[lp_idx][elem_idx] = dd_A[lp_idx][dd_b[lp_idx][col_idx] * M + row_idx];
}


__global__ void copy_basis_feasibility_1(int split_idx, bool **dd_comb_bool, int *d_lp_id_dict, int *d_root_idxs, int *d_min_differ, bool *d_feasibilities, double *d_z_final, int n_c, int nTPB)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int lp_idx = blockIdx.x + split_idx;
    int lp_id = d_lp_id_dict[lp_idx];
    // int root_idx = threadIdx.x; // First lp_idx to be compared with
    int root_idx = split_idx - threadIdx.x - 1; // First lp_idx to be compared with
    int c_idx = 0;
    int n_differ = 0;  // Difference between root lp and sub lp. NOTE: only consider the root lp rather than similar lp

    if (threadIdx.x == 0) {
        d_root_idxs[lp_idx] = -1;
        d_min_differ[lp_idx] = n_c;
    }

    __syncthreads();

    while (root_idx >= 0 && d_feasibilities[lp_idx]) {
        n_differ = 0;
        for (c_idx = 0; c_idx < n_c; c_idx++) {
            if (dd_comb_bool[d_lp_id_dict[root_idx]][c_idx] && !dd_comb_bool[lp_id][c_idx]) {
                n_differ = n_c;
                break;
            } else if (!dd_comb_bool[d_lp_id_dict[root_idx]][c_idx] && dd_comb_bool[lp_id][c_idx]) {
                n_differ++;
                if (n_differ >= tol_diff_fea + 1) {
                    n_differ = n_c;
                    break;  
                }
                
            }
        }

        if (n_differ < d_min_differ[lp_idx]) {
            if (!d_feasibilities[root_idx]) {
                d_feasibilities[lp_idx] = false;
                return;
            } else if (d_z_final[d_lp_id_dict[root_idx]] > -1e-12 && d_z_final[d_lp_id_dict[root_idx]] < 1e-12) {
                d_min_differ[lp_idx] = n_differ;
                d_root_idxs[lp_idx] = root_idx;
            } else {
                d_min_differ[lp_idx] = n_differ;
                d_root_idxs[lp_idx] = root_idx;
            }
        }

        if (d_min_differ[lp_idx] <= tol_diff_fea) {
            break;
        }

        root_idx -= nTPB;
    }
}

__global__ void copy_basis_feasibility_2(int split_idx, int n_copy_lps, int *d_root_idxs, int *d_min_differ, bool *d_feasibilities,
                                         int *d_b, int *d_is_b, double *d_A, double *d_B_copy, double *d_c_copy, double *d_cb, int M, int N)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N * n_copy_lps) return;

    int lp_idx = tid / M + split_idx;

    int c_idx = tid % M;
    
    if (tid < M * n_copy_lps && d_feasibilities[lp_idx] && d_min_differ[lp_idx] <= tol_diff_fea) {

        d_b[lp_idx * M + c_idx] = d_b[d_root_idxs[lp_idx] * M + c_idx];

        d_cb[lp_idx * M + c_idx] = d_c_copy[lp_idx * N + d_b[lp_idx * M + c_idx]];
        for (int i = 0; i < M; i++) {
            d_B_copy[lp_idx * M * M + c_idx * M + i] = d_A[lp_idx * M * N + d_b[lp_idx * M + c_idx] * M + i];
        }
    }

    lp_idx = tid / N + split_idx;
    if (!d_feasibilities[lp_idx] || d_min_differ[lp_idx] > tol_diff_fea) return;

    c_idx = tid % N;
    d_is_b[lp_idx * N + c_idx] = d_is_b[d_root_idxs[lp_idx] * N + c_idx];
}

__global__ void dual_pivot_kernel_1(double **dd_cols, int **dd_non_b, double **dd_xb, int *d_min_idxs, double **dd_c,
                               int *d_th_idxs, double *d_th_star, int **dd_non_b_counter, const int data_size, bool *d_go_on,
                               int *d_opt_test, int *d_infoArray, int *d_init_lp_idxs, int M, int N, int K, int K_2, int n_cycles_same_lps, int batch_idx, int output_flag, double tol_minus)
{
    if (!d_go_on[blockIdx.x] || !d_opt_test[blockIdx.x] || d_infoArray[blockIdx.x]) return;

    extern __shared__ char array[];

    double *vals = (double *)array;
    int *idxs = (int *)(K * sizeof(double) + array);
    const int tid = blockIdx.x * K + threadIdx.x;
    double val = 0;

    val = dd_cols[blockIdx.x][threadIdx.x * M + d_min_idxs[d_init_lp_idxs[blockIdx.x]]];

    __syncthreads();
    if (!d_opt_test[d_init_lp_idxs[blockIdx.x]]) return;
    
    if (val < tol_minus) {
        idxs[threadIdx.x] = dd_non_b[blockIdx.x][threadIdx.x];
        vals[threadIdx.x] = - dd_c[blockIdx.x][idxs[threadIdx.x]] / val;
    } else {
        idxs[threadIdx.x] = -1;
        vals[threadIdx.x] = 100000.0;
    }
    __syncthreads();

    for (int i = K_2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i && threadIdx.x + i < K)
            if (n_cycles_same_lps < 10) {
                if ((idxs[threadIdx.x + i] >= 0) &&
                    (idxs[threadIdx.x] < 0 || vals[threadIdx.x] > vals[threadIdx.x + i]))  // Ensure that the minimum value obtained is the first one
                {
                    vals[threadIdx.x] = vals[threadIdx.x + i];
                    idxs[threadIdx.x] = idxs[threadIdx.x + i];
                }
            } else if (n_cycles_same_lps < 100) {
                if ((idxs[threadIdx.x + i] >= 0) &&
                    (idxs[threadIdx.x] < 0 || vals[threadIdx.x] > vals[threadIdx.x + i] ||
                        (vals[threadIdx.x] >= vals[threadIdx.x + i] - 1e-12 &&
                        dd_non_b_counter[d_init_lp_idxs[blockIdx.x]][dd_non_b[blockIdx.x][idxs[threadIdx.x]]] > dd_non_b_counter[d_init_lp_idxs[blockIdx.x]][dd_non_b[blockIdx.x][idxs[threadIdx.x + i]]] + 2)))
                {
                    vals[threadIdx.x] = vals[threadIdx.x + i];
                    idxs[threadIdx.x] = idxs[threadIdx.x + i];
                }
            } else {
                if ((idxs[threadIdx.x + i] >= 0) &&
                    (idxs[threadIdx.x] < 0 ||
                        (vals[threadIdx.x] >= vals[threadIdx.x + i] - 1e-10 &&
                        dd_non_b_counter[d_init_lp_idxs[blockIdx.x]][dd_non_b[blockIdx.x][idxs[threadIdx.x]]] > dd_non_b_counter[d_init_lp_idxs[blockIdx.x]][dd_non_b[blockIdx.x][idxs[threadIdx.x + i]]] + 5)))
                {
                    vals[threadIdx.x] = vals[threadIdx.x + i];
                    idxs[threadIdx.x] = idxs[threadIdx.x + i];
                }
            }
            
        __syncthreads();
    }

    if (dd_non_b[blockIdx.x][threadIdx.x] == idxs[0]) {
        d_th_idxs[d_init_lp_idxs[blockIdx.x]] = threadIdx.x;
        dd_non_b_counter[d_init_lp_idxs[blockIdx.x]][idxs[0]] += 1;

        d_th_star[blockIdx.x] = dd_xb[blockIdx.x][d_min_idxs[d_init_lp_idxs[blockIdx.x]]] / val;
    }
}

__global__ void dual_pivot_kernel_2(int **dd_b, int **dd_non_b, int *d_th_idxs, int *d_min_idxs, double *d_c_copy, double **dd_c, double **dd_cb,
                               double **dd_B_copy, double **dd_A, bool *d_go_on, int *d_opt_test, int *d_infoArray, int *d_init_lp_idxs, int *d_lp_id_dict, int n_cycles, int M)
{
    if (!d_go_on[blockIdx.x] || d_infoArray[blockIdx.x]) return;
    if (!d_opt_test[blockIdx.x]) {
        d_opt_test[blockIdx.x] = 1;
        return;
    }

    int new_b = dd_non_b[blockIdx.x][d_th_idxs[d_init_lp_idxs[blockIdx.x]]];

    __syncthreads();
    
    if (!threadIdx.x) {
        dd_non_b[blockIdx.x][d_th_idxs[d_init_lp_idxs[blockIdx.x]]] = dd_b[blockIdx.x][d_min_idxs[d_init_lp_idxs[blockIdx.x]]];
        dd_b[blockIdx.x][d_min_idxs[d_init_lp_idxs[blockIdx.x]]] = new_b;
        dd_cb[blockIdx.x][d_min_idxs[d_init_lp_idxs[blockIdx.x]]] = d_c_copy[new_b];
    }

    dd_B_copy[blockIdx.x][d_min_idxs[d_init_lp_idxs[blockIdx.x]] * M + threadIdx.x] = dd_A[blockIdx.x][new_b * M + threadIdx.x];
}

__global__ void pivot_kernel_1_2(double **dd_col, int **dd_b, int **dd_is_b, double **dd_xb, int *d_min_idxs, double **dd_c_copy, double **dd_cb,
                               double **dd_B_copy, double **dd_A, const int data_size, bool *d_go_on, int *d_init_lp_idxs, int M, int N, int M_2, double tol_plus)
{
    if (!d_go_on[blockIdx.x]) return;

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ char array[];

    double *vals = (double *)array;
    int *idxs = (int *)(M * sizeof(double) + array);

    if (tid >= data_size) return;
    if (dd_col[blockIdx.x][threadIdx.x] > tol_plus) {
        idxs[threadIdx.x] = threadIdx.x;
        vals[threadIdx.x] = dd_xb[blockIdx.x][threadIdx.x] / dd_col[blockIdx.x][threadIdx.x];
    } else {
        idxs[threadIdx.x] = -1;
        vals[threadIdx.x] = 10.0;
    }
    __syncthreads();

    for (int i = M_2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i && threadIdx.x + i < M)
            if ((idxs[threadIdx.x + i] >= 0) &&
                (idxs[threadIdx.x] < 0 || vals[threadIdx.x] > vals[threadIdx.x + i] || (vals[threadIdx.x] == vals[threadIdx.x + i] && idxs[threadIdx.x] > idxs[threadIdx.x + i])))  // Ensure that the minimum value obtained is the first one
            {
                vals[threadIdx.x] = vals[threadIdx.x + i];
                idxs[threadIdx.x] = idxs[threadIdx.x + i];
            }
        __syncthreads();
    }

    if (threadIdx.x == idxs[0]) {
        dd_is_b[blockIdx.x][dd_b[blockIdx.x][threadIdx.x]] = 0;
        dd_b[blockIdx.x][threadIdx.x] = d_min_idxs[blockIdx.x];
        dd_is_b[blockIdx.x][d_min_idxs[blockIdx.x]] = 1;
        dd_cb[blockIdx.x][threadIdx.x] = dd_c_copy[blockIdx.x][d_min_idxs[blockIdx.x]];  // NOTE: different from pivot_kernel_2 here
    }
    __syncthreads();
    dd_B_copy[blockIdx.x][idxs[0] * M + threadIdx.x] = dd_A[blockIdx.x][d_min_idxs[blockIdx.x] * M + threadIdx.x];
}


__global__ void pivot_kernel_2(double **dd_col, int **dd_b, int **dd_is_b, double **dd_xb, int *d_min_idxs, double *d_c_copy, double **dd_cb,
                               double **dd_B_copy, double **dd_A, const int data_size, bool *d_go_on, int M, int N, int M_2, double tol_plus)
{
    if (!d_go_on[blockIdx.x]) return;

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ char array[];

    double *vals = (double *)array;
    int *idxs = (int *)(M * sizeof(double) + array);

    if (tid >= data_size) return;
    if (dd_col[blockIdx.x][threadIdx.x] > tol_plus) {
        idxs[threadIdx.x] = threadIdx.x;
        vals[threadIdx.x] = dd_xb[blockIdx.x][threadIdx.x] / dd_col[blockIdx.x][threadIdx.x];
    } else {
        idxs[threadIdx.x] = -1;
        vals[threadIdx.x] = 10.0;
    }
    __syncthreads();

    for (int i = M_2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i && threadIdx.x + i < M)
            if ((idxs[threadIdx.x + i] >= 0) &&
                (idxs[threadIdx.x] < 0 || vals[threadIdx.x] > vals[threadIdx.x + i] || (vals[threadIdx.x] == vals[threadIdx.x + i] && idxs[threadIdx.x] > idxs[threadIdx.x + i])))  // Ensure that the minimum value obtained is the first one
            {
                vals[threadIdx.x] = vals[threadIdx.x + i];
                idxs[threadIdx.x] = idxs[threadIdx.x + i];
            }
        __syncthreads();
    }

    if (threadIdx.x == idxs[0]) {
        dd_is_b[blockIdx.x][dd_b[blockIdx.x][threadIdx.x]] = 0;
        dd_b[blockIdx.x][threadIdx.x] = d_min_idxs[blockIdx.x];
        dd_is_b[blockIdx.x][d_min_idxs[blockIdx.x]] = 1;
        dd_cb[blockIdx.x][threadIdx.x] = d_c_copy[d_min_idxs[blockIdx.x]];
    }

    __syncthreads();
    dd_B_copy[blockIdx.x][idxs[0] * M + threadIdx.x] = dd_A[blockIdx.x][d_min_idxs[blockIdx.x] * M + threadIdx.x];
}


void simplex_gpu_feasibility(double *A_i, int *b, bool *d_comb_bool, bool **dd_comb_bool, bool *d_feasibilities, int *d_lp_id_dict, double *xb_i, int n_c, const int M, const int N, int n_lps,
                             int output_flag, int n_splits, int nTPB, double tol_plus, double tol_minus_1, cublasHandle_t handle, cudaStream_t stream)
{
    int n_active_lps = n_lps;

    int M_2 = M - 1;
    M_2 = M_2 | (M_2>>1);
    M_2 = M_2 | (M_2>>2);
    M_2 = M_2 | (M_2>>4);
    M_2 = M_2 | (M_2>>8);  // this handles up to 16-bit integers
    M_2 = (M_2 + 1) >> 1;
    // cout << M << " >> " << M_2 << endl;

    int N_2 = N - 1;
    N_2 = N_2 | (N_2>>1);
    N_2 = N_2 | (N_2>>2);
    N_2 = N_2 | (N_2>>4);
    N_2 = N_2 | (N_2>>8);  // this handles up to 16-bit integers
    N_2 = (N_2 + 1) >> 1;
    // cout << N << " >> " << N_2 << endl;

    // show memory usage of GPU
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status;
    double free_db = 0;
    double total_db = 0;
    double used_db = 0;

    // ********* gpu initialise **********
    thrust::device_vector<bool> thst_go_on(n_lps, true);
    bool *d_go_on = thrust::raw_pointer_cast(thst_go_on.data());
    thrust::device_vector<int> thst_lp_idxs(n_lps);
    int *d_lp_idxs = thrust::raw_pointer_cast(thst_lp_idxs.data());
    thrust::sequence(thst_lp_idxs.begin(), thst_lp_idxs.end());

    int *d_infoArray = nullptr;
    int *d_pivotArray = nullptr;
    int *d_root_idxs = nullptr;
    int *d_min_differ = nullptr;
    int *d_b = nullptr;
    int *d_is_b = nullptr;
    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_B_copy = nullptr;
    double *d_c = nullptr;
    double *d_c_copy = nullptr;
    double *d_cb = nullptr;
    double *d_xb_copy = nullptr;
    double *d_xb = nullptr;
    double *d_col = nullptr;
    double *d_z_final = nullptr;

    int *d_min_idxs;
    int *d_init_lp_idxs;
    double *d_min_vals;

    // show memory usage of GPU begin
    cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }
    free_db = (double)free_byte;
    total_db = (double)total_byte;
    used_db = total_db - free_db;
    if (output_flag > 1) printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    // show memory usage of GPU end

    CUDA_CHECK(cudaMalloc(&d_infoArray, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_pivotArray, sizeof(int) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_root_idxs, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_min_differ, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_is_b, sizeof(int) * N * n_lps));

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * M * N * n_lps));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(double) * M * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_B_copy, sizeof(double) * M * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_c, sizeof(double) * N * n_lps));
    CUDA_CHECK(cudaMalloc(&d_c_copy, sizeof(double) * N * n_lps));
    CUDA_CHECK(cudaMalloc(&d_cb, sizeof(double) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_xb, sizeof(double) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_xb_copy, sizeof(double) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_col, sizeof(double) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_z_final, sizeof(double) * n_lps));

    CUDA_CHECK(cudaMalloc(&d_min_idxs, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_init_lp_idxs, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_min_vals, sizeof(double) * n_lps));

    CUDA_CHECK(cudaMemcpyAsync(d_b, b, sizeof(int) * M, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A_i, sizeof(double) * M * N, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_xb_copy, xb_i, sizeof(double) * M, cudaMemcpyHostToDevice, stream));
    
    int **dd_b = nullptr;
    int **dd_is_b = nullptr;
    double **dd_A = nullptr;
    double **dd_B = nullptr;
    double **dd_B_copy = nullptr;
    double **dd_c = nullptr;
    double **dd_c_copy = nullptr;
    double **dd_cb = nullptr;
    double **dd_xb = nullptr;
    double **dd_col = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_b), sizeof(int *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_is_b), sizeof(int *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_A), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_B), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_B_copy), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_c), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_c_copy), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_cb), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_xb), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_col), sizeof(double *) * n_lps));

    // CUDA_CHECK(cudaStreamSynchronize(stream));
    replicate_b_kernel_1<<<(n_active_lps*M+nTPB-1)/nTPB, nTPB>>>(d_b, n_active_lps, M);
    replicate_xb_kernel_1<<<(n_lps*M+nTPB-1)/nTPB, nTPB>>>(d_xb_copy, n_lps, M);

    int h_infoArray;
    
    double alpha = -1.;
	double beta = 1.;

    int n_active_lps_new = n_lps;
    int n_new_finished_lps = 0;

    CUDA_CHECK(cudaMemset(d_is_b, 0, N * n_lps * sizeof(int)));

    initialise_dd<<<(n_lps+nTPB-1)/nTPB, nTPB>>>(d_A, dd_A, d_B, dd_B, d_B_copy, dd_B_copy, d_b, dd_b, d_is_b, dd_is_b, d_c, dd_c, d_c_copy, dd_c_copy,
                                                 d_cb, dd_cb, d_xb, dd_xb, d_col, dd_col, n_lps, M, N);
    // CUDA_CHECK(cudaDeviceSynchronize());

    initialise_c<<<(n_lps*N+nTPB-1)/nTPB, nTPB>>>(dd_c_copy, d_c_copy, dd_comb_bool, d_lp_id_dict, n_lps, N, N - M, n_c);
    // CUDA_CHECK(cudaDeviceSynchronize());
    initialise_is_b_cb<<<(n_active_lps*M+nTPB-1)/nTPB, nTPB>>>(dd_b, dd_is_b, dd_c_copy, dd_cb, n_active_lps, M, N);
    initialise_init_lp_idxs<<<(n_lps+nTPB-1)/nTPB, nTPB>>>(d_init_lp_idxs, n_lps);
    
    get_B_kernel_1<<<(n_active_lps*M*M+nTPB-1)/nTPB, nTPB>>>(d_A, dd_b, dd_B_copy, n_active_lps, M, M * M);
    // CUDA_CHECK(cudaDeviceSynchronize());

    replicate_A_kernel_1<<<(n_lps*M*N+nTPB-1)/nTPB, nTPB>>>(dd_A, n_lps, M * N);

    CUDA_CHECK(cudaMemcpyAsync(d_B, d_B_copy, sizeof(double) * M * M * n_active_lps, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    int cum_n_active_lps = 0;
    int n_cycles = 0;

    int *split_idxs = new int[n_splits];

    if (output_flag > 0) cout << "Splits:";

    for (int i = 1; i <= n_splits; i++) {
        if (i == n_splits) {
            split_idxs[i - 1] = n_lps;
            if (output_flag > 0) cout << " " << split_idxs[i - 1];
            break;
        }
        split_idxs[i - 1] = n_lps / n_splits * i;
        if (output_flag > 0) cout << " " << split_idxs[i - 1];
    }
    if (output_flag > 0) cout << endl;

    int copy_idx = 0;
    
    n_active_lps = *split_idxs;

    bool just_copy = false;
    
    for (n_cycles = 0; n_cycles < 500; n_cycles++)
    {
        CUBLAS_CHECK(cublasDgetrfBatched(handle, M, dd_B, M, d_pivotArray, d_infoArray, n_active_lps));  // d_pivotArray
        
        CUDA_CHECK(cudaMemcpyAsync(d_xb, d_xb_copy, sizeof(double) * n_active_lps * M, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaStreamSynchronize(stream));

        copy_cb_kernel_1<<<(n_active_lps*M+nTPB-1)/nTPB, nTPB>>>(dd_col, dd_cb, n_active_lps, M);

        cublasDgetrsBatched(handle, CUBLAS_OP_T, M, 1, dd_B, M, d_pivotArray, dd_col, M, &h_infoArray, n_active_lps);  // dd_col = cb * B_I

        CUBLAS_CHECK(cublasDgetrsBatched(handle, CUBLAS_OP_N, M, 1, dd_B, M, d_pivotArray, dd_xb, M, &h_infoArray, n_active_lps));

        copy_c_kernel_1<<<(n_active_lps*N+nTPB-1)/nTPB, nTPB>>>(dd_c, dd_c_copy, n_active_lps, N);

        cublasDgemvBatched(handle, CUBLAS_OP_T, M, N, &alpha, dd_A, M, dd_col, 1, &beta, dd_c, 1, n_active_lps);  // dd_c = dd_c - dd_col * dd_A

        // *********** Find the minimum value in dd_c *******
        size_t shared_mem_size = N*sizeof(double) + N*sizeof(int);
        min_idx_kernel<<<n_active_lps, N, shared_mem_size>>>(dd_c, dd_is_b, d_min_idxs, d_min_vals, d_go_on, dd_cb, dd_xb, d_z_final, d_feasibilities, d_init_lp_idxs, M, N, M_2, N_2, tol_minus_1);

        n_active_lps_new = thrust::reduce(thst_go_on.begin(), thst_go_on.begin() + n_active_lps, 0, thrust::plus<int>());

        n_new_finished_lps = n_active_lps - n_active_lps_new;
        if (!n_active_lps_new) break;
        
        copy_col_kernel<<<(n_active_lps*M+nTPB-1)/nTPB, nTPB>>>(dd_A, dd_col, d_min_idxs, n_active_lps*M, M);

        cublasDgetrsBatched(handle, CUBLAS_OP_N, M, 1, dd_B, M, d_pivotArray, dd_col, M, &h_infoArray, n_active_lps);  // B_I.dot(A[:, j])

        pivot_kernel_1_2<<<n_active_lps, M, M*sizeof(double) + M*sizeof(int)>>>(dd_col, dd_b, dd_is_b, dd_xb, d_min_idxs, dd_c_copy, dd_cb, dd_B_copy, dd_A, n_active_lps*M, d_go_on, d_init_lp_idxs, M, N, M_2, tol_plus);
        
        if (n_active_lps < 30 && copy_idx < n_splits - 1) {
            int n_copy_lps = *(split_idxs+1) - *split_idxs;

            copy_basis_feasibility_1<<<n_copy_lps, nTPB>>>(*split_idxs, dd_comb_bool, d_lp_id_dict, d_root_idxs, d_min_differ, d_feasibilities, d_z_final, n_c, nTPB);

            copy_basis_feasibility_2<<<(N*n_copy_lps+nTPB-1)/nTPB, nTPB>>>(*split_idxs, n_copy_lps, d_root_idxs, d_min_differ, d_feasibilities, d_b, d_is_b, d_A, d_B_copy, d_c_copy, d_cb, M, N);

            reset_d_go_on_feasi<<<(n_lps+nTPB-1)/nTPB, nTPB>>>(d_go_on, d_feasibilities, d_lp_idxs, *split_idxs, n_copy_lps, n_active_lps, n_lps);

            n_active_lps_new = thrust::reduce(thst_go_on.begin(), thst_go_on.begin() + *(split_idxs+1), 0, thrust::plus<int>());
            n_active_lps = n_active_lps_new;

            split_idxs++;
            copy_idx++;
            just_copy = true;
        }

        if (n_new_finished_lps > 0 || just_copy) {
            thrust::sort_by_key(thst_go_on.begin(), thst_go_on.end(), thst_lp_idxs.begin(), thrust::greater<bool>());

            move_kernel_1<<<(n_active_lps+nTPB-1)/nTPB, nTPB>>>(n_active_lps, d_lp_idxs, d_init_lp_idxs, dd_b, dd_is_b, dd_A, dd_B_copy, dd_cb, dd_c, dd_c_copy, n_cycles, M, N);
        }

        n_active_lps = n_active_lps_new;

        copy_B_kernel<<<(n_active_lps*M*M+nTPB-1)/nTPB, nTPB>>>(dd_B, dd_B_copy, n_active_lps, M * M);

        if (output_flag > 0) cout << "n_active_lps: " << n_active_lps << endl;
        cum_n_active_lps += n_active_lps;
    }

    /* free resources */
    CUDA_CHECK(cudaFree(d_infoArray));
    CUDA_CHECK(cudaFree(d_pivotArray));
    CUDA_CHECK(cudaFree(d_root_idxs));
    CUDA_CHECK(cudaFree(d_min_differ));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_is_b));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_B_copy));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_copy));
    CUDA_CHECK(cudaFree(d_cb));
    CUDA_CHECK(cudaFree(d_xb));
    CUDA_CHECK(cudaFree(d_xb_copy));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_z_final));
    CUDA_CHECK(cudaFree(d_min_idxs));
    CUDA_CHECK(cudaFree(d_init_lp_idxs));
    CUDA_CHECK(cudaFree(d_min_vals));

    CUDA_CHECK(cudaFree(dd_b));
    CUDA_CHECK(cudaFree(dd_is_b));
    CUDA_CHECK(cudaFree(dd_A));
    CUDA_CHECK(cudaFree(dd_B));
    CUDA_CHECK(cudaFree(dd_B_copy));
    CUDA_CHECK(cudaFree(dd_c));
    CUDA_CHECK(cudaFree(dd_c_copy));
    CUDA_CHECK(cudaFree(dd_cb));
    CUDA_CHECK(cudaFree(dd_xb));
    CUDA_CHECK(cudaFree(dd_col));

    // CUDA_CHECK(cudaDeviceReset());
}

std::tuple<int *, double *, int> dual_simplex_gpu_optimality(int batch_idx, double *A, int *b, int *non_b, int n_eqs, double *xb_i, bool **dd_comb_bool, int *d_lp_id_dict, int k_max,
                            int n_c, int n_x, int n_t, const int M, const int N, const int N_, int n_lps, int output_flag, int n_splits, int nTPB,
                            double tol_plus, double tol_minus, double tol_minus_1, double tol_minus_optimal, double dual_feasible_tol, double dual_stop_tol, cublasHandle_t handle, cudaStream_t stream)
{
    clock_t start, end;

    start = clock();

    volatile int n_active_lps = n_lps;

    int M_2 = M - 1;
    M_2 = M_2 | (M_2>>1);
    M_2 = M_2 | (M_2>>2);
    M_2 = M_2 | (M_2>>4);
    M_2 = M_2 | (M_2>>8);  // this handles up to 16-bit integers
    M_2 = (M_2 + 1) >> 1;

    int N_2 = N - 1;
    N_2 = N_2 | (N_2>>1);
    N_2 = N_2 | (N_2>>2);
    N_2 = N_2 | (N_2>>4);
    N_2 = N_2 | (N_2>>8);
    N_2 = (N_2 + 1) >> 1;

    int N_2_ = N_ - 1;
    N_2_ = N_2_ | (N_2_>>1);
    N_2_ = N_2_ | (N_2_>>2);
    N_2_ = N_2_ | (N_2_>>4);
    N_2_ = N_2_ | (N_2_>>8);
    N_2_ = (N_2_ + 1) >> 1;

    int K = N - M;
    int K_2 = K - 1;
    K_2 = K_2 | (K_2>>1);
    K_2 = K_2 | (K_2>>2);
    K_2 = K_2 | (K_2>>4);
    K_2 = K_2 | (K_2>>8);
    K_2 = (K_2 + 1) >> 1;

    int max_M_K = std::max(M, K);

    // ********* gpu initialise **********
    thrust::device_vector<bool> thst_go_on(n_lps, true);
    thrust::device_vector<bool> thst_is_feasible(n_lps, false);
    thrust::device_vector<int> thst_lp_idxs(n_lps);
    
    bool *d_go_on = thrust::raw_pointer_cast(thst_go_on.data());
    bool *d_is_feasible = thrust::raw_pointer_cast(thst_is_feasible.data());
    int *d_lp_idxs = thrust::raw_pointer_cast(thst_lp_idxs.data());
    // thrust::sequence(thst_lp_idxs.begin(), thst_lp_idxs.end());

    int *d_infoArray = nullptr;
    int *d_pivotArray = nullptr;
    int *d_root_idxs = nullptr;
    int *d_min_differ = nullptr;
    int *d_b = nullptr;
    int *d_non_b = nullptr;
    int *d_b_init = nullptr;
    int *d_non_b_init = nullptr;
    int *d_b_counter = nullptr;
    int *d_non_b_counter = nullptr;
    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_B_copy = nullptr;
    double *d_c = nullptr;
    double *d_c_copy = nullptr;
    double *d_cb = nullptr;
    double *d_xb_copy = nullptr;
    double *d_xb = nullptr;
    double *d_col = nullptr;
    double *d_cols = nullptr;
    double *d_z_final = nullptr;
    double *d_th_star = nullptr;
    double *d_min_vals = nullptr;
    int *d_th_idxs = nullptr;
    int *d_opt_test = nullptr;  // Test the optimality for dual simplex
    int *d_is_invertible = nullptr;
    int *d_min_idxs;
    int *d_init_lp_idxs;

    CUDA_CHECK(cudaMalloc(&d_b_init, sizeof(int) * M));
    CUDA_CHECK(cudaMalloc(&d_non_b_init, sizeof(int) * K));
    CUDA_CHECK(cudaMalloc(&d_infoArray, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_pivotArray, sizeof(int) * n_lps * M));
    CUDA_CHECK(cudaMalloc(&d_root_idxs, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_min_differ, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_non_b, sizeof(int) * K * n_lps));
    CUDA_CHECK(cudaMalloc(&d_b_counter, sizeof(int) * N * n_lps));
    CUDA_CHECK(cudaMalloc(&d_non_b_counter, sizeof(int) * N * n_lps));
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * M * N * n_lps));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(double) * M * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_B_copy, sizeof(double) * M * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_c, sizeof(double) * N * n_lps));
    CUDA_CHECK(cudaMalloc(&d_c_copy, sizeof(double) * N * n_lps));
    CUDA_CHECK(cudaMalloc(&d_cb, sizeof(double) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_xb, sizeof(double) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_xb_copy, sizeof(double) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_col, sizeof(double) * M * n_lps));
    CUDA_CHECK(cudaMalloc(&d_cols, sizeof(double) * M * K * n_lps));
    CUDA_CHECK(cudaMalloc(&d_z_final, sizeof(double) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_th_star, sizeof(double) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_th_idxs, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_opt_test, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_is_invertible, sizeof(int) * n_lps));

    CUDA_CHECK(cudaMalloc(&d_min_idxs, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_init_lp_idxs, sizeof(int) * n_lps));
    CUDA_CHECK(cudaMalloc(&d_min_vals, sizeof(double) * n_lps));

    CUDA_CHECK(cudaMemcpyAsync(d_b, b, sizeof(int) * M, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_non_b, non_b, sizeof(int) * K, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b_init, b, sizeof(int) * M, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_non_b_init, non_b, sizeof(int) * K, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeof(double) * M * N, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_xb_copy, xb_i, sizeof(double) * M, cudaMemcpyHostToDevice, stream));
    
    int **dd_b = nullptr;
    int **dd_non_b = nullptr;
    int **dd_b_counter = nullptr;
    int **dd_non_b_counter = nullptr;
    double **dd_A = nullptr;
    double **dd_B = nullptr;
    double **dd_B_copy = nullptr;
    double **dd_c = nullptr;
    double **dd_c_copy = nullptr;
    double **dd_cb = nullptr;
    double **dd_xb = nullptr;
    double **dd_col = nullptr;
    double **dd_cols = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_b), sizeof(int *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_non_b), sizeof(int *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_b_counter), sizeof(int *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_non_b_counter), sizeof(int *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_A), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_B), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_B_copy), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_c), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_c_copy), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_cb), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_xb), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_col), sizeof(double *) * n_lps));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_cols), sizeof(double *) * n_lps));

    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // replicate_A_kernel_1<<<(n_lps*M*N+nTPB-1)/nTPB, nTPB>>>(dd_A, n_lps, M * N);
    replicate_b_kernel_1<<<(n_lps*M+nTPB-1)/nTPB, nTPB>>>(d_b, n_lps, M);
    replicate_non_b_kernel_1<<<(n_lps*K+nTPB-1)/nTPB, nTPB>>>(d_non_b, n_lps, K);
    replicate_xb_kernel_1<<<(n_lps*M+nTPB-1)/nTPB, nTPB>>>(d_xb_copy, n_lps, M);
    replicate_c_kernel_1<<<(n_lps*N+nTPB-1)/nTPB, nTPB>>>(d_c_copy, d_b_counter, d_non_b_counter, n_eqs, n_lps, N);
    
    int h_infoArray;
    
    double alpha = -1.;
	double beta = 1.;

    int n_active_lps_new = n_lps;
    int n_new_finished_lps = 0;

    initialise_dd_2_1<<<(n_lps+nTPB-1)/nTPB, nTPB>>>(d_A, dd_A, d_B, dd_B, d_B_copy, dd_B_copy, d_b, dd_b, d_non_b, dd_non_b, d_c, dd_c, d_c_copy, dd_c_copy,
                                                     d_cb, dd_cb, d_xb, dd_xb, d_col, dd_col, d_cols, dd_cols, d_b_counter, dd_b_counter, d_non_b_counter, dd_non_b_counter,
                                                     d_z_final, d_opt_test, d_is_invertible, d_lp_idxs, n_lps, M, N, K);
    // CUDA_CHECK(cudaDeviceSynchronize());
    
    replicate_A_kernel_1<<<(n_lps*M*N+nTPB-1)/nTPB, nTPB>>>(dd_A, n_lps, M * N);
    // CUDA_CHECK(cudaDeviceSynchronize());

    initialise_A_kernel_1<<<(n_lps*n_c+nTPB-1)/nTPB, nTPB>>>(dd_A, dd_comb_bool, d_lp_id_dict, n_c, n_x, n_t, M, N, n_lps);
    initialise_cb<<<(n_active_lps*M+nTPB-1)/nTPB, nTPB>>>(dd_b, dd_c_copy, dd_cb, n_active_lps, M, N);

    initialise_init_lp_idxs<<<(n_lps+nTPB-1)/nTPB, nTPB>>>(d_init_lp_idxs, n_lps);

    // CUDA_CHECK(cudaDeviceSynchronize());

    get_B_kernel_2<<<(n_active_lps*M*M+nTPB-1)/nTPB, nTPB>>>(dd_A, dd_b, dd_B_copy, n_active_lps, M, M * M);
    // CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyAsync(d_B, d_B_copy, sizeof(double) * M * M * n_active_lps, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(d_xb, d_xb_copy, sizeof(double) * n_active_lps * M, cudaMemcpyDeviceToDevice, stream));

    /****************** Phase 1 ******************/

    clock_t start_time, end_time = 0;

    int cum_n_active_lps = 0;
    int n_cycles = 0;

    int split_idxs_array[n_splits];
    int *split_idxs = split_idxs_array;

    if (output_flag > 0) cout << "n_splits:";

    for (int i = 1; i <= n_splits; i++) {
        if (i == n_splits) {
            split_idxs[i - 1] = n_lps;
            if (output_flag > 0) cout << " " << split_idxs[i - 1];
            break;
        }
        split_idxs[i - 1] = n_lps / n_splits * i;
        if (output_flag > 0) cout << " " << split_idxs[i - 1];
    }
    if (output_flag > 0) cout << endl;

    int copy_idx = 0;
    n_active_lps = *split_idxs;

    bool just_copy = false;

    int n_cycles_same_lps = 0;

    // CUDA_CHECK(cudaStreamSynchronize(stream));

    for (n_cycles = 0; n_cycles < 10000; n_cycles++)
    {
        if (output_flag > 0) cout << "************ " << n_cycles << " ************" << endl;

        CUBLAS_CHECK(cublasDgetrfBatched(handle, M, dd_B, M, d_pivotArray, d_infoArray, n_active_lps));

        if (just_copy) {
            check_invertible_kernel_dual_2<<<(n_active_lps+nTPB-1)/nTPB, nTPB>>>(d_infoArray, d_init_lp_idxs, d_lp_id_dict, d_is_invertible, dd_B_copy, dd_A, dd_b, dd_non_b,
                                              d_b_init, d_non_b_init, n_active_lps, M, N, K);
            just_copy = false;
        }

        CUBLAS_CHECK(cublasDgetrsBatched(handle, CUBLAS_OP_N, M, 1, dd_B, M, d_pivotArray, dd_xb, M, &h_infoArray, n_active_lps));

        // ********* Find the minimum value in d_xb *********
        size_t shared_mem_size = M*sizeof(double) + M*sizeof(int);
        dual_min_idx_kernel<<<n_active_lps, M, shared_mem_size>>>(dd_xb, d_min_idxs, d_min_vals, d_th_idxs, d_go_on, dd_cb, d_z_final, d_init_lp_idxs,
                d_lp_id_dict, dd_b, dd_non_b, dd_b_counter, d_is_feasible, d_infoArray, n_cycles_same_lps, M, M_2, N, N_, dual_feasible_tol, dual_stop_tol);

        start_time = clock();
        copy_cb_kernel_1<<<(n_active_lps*M+nTPB-1)/nTPB, nTPB>>>(dd_col, dd_cb, n_active_lps, M);

        cublasDgetrsBatched(handle, CUBLAS_OP_T, M, 1, dd_B, M, d_pivotArray, dd_col, M, &h_infoArray, n_active_lps);  // dd_col = cb * B_I

        CUDA_CHECK(cudaMemcpyAsync(d_c, d_c_copy, sizeof(double) * n_active_lps * N, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // NOTE: some times dd_c is different when b is artifical variables
        cublasDgemvBatched(handle, CUBLAS_OP_T, M, N, &alpha, dd_A, M, dd_col, 1, &beta, dd_c, 1, n_active_lps);  // dd_c = dd_c - dd_col * dd_A  

        n_active_lps_new = thrust::reduce(thst_go_on.begin(), thst_go_on.begin() + n_active_lps, 0, thrust::plus<int>());
        n_new_finished_lps = n_active_lps - n_active_lps_new;

        if (!n_active_lps_new && copy_idx==n_splits-1) break;
        
        copy_multi_cols_kernel<<<(n_active_lps*K+nTPB-1)/nTPB, nTPB>>>(dd_A, dd_cols, dd_non_b, n_active_lps*K, M, K);  // copy A[:, ~bl] to d_cols

        cublasDgetrsBatched(handle, CUBLAS_OP_N, M, K, dd_B, M, d_pivotArray, dd_cols, M, &h_infoArray, n_active_lps);  // B_I * A[:, ~bl]

        dual_pivot_kernel_1<<<n_active_lps, K, K*sizeof(double) + K*sizeof(int)>>>(
            dd_cols, dd_non_b, dd_xb, d_min_idxs, dd_c, d_th_idxs, d_th_star, dd_non_b_counter, n_active_lps*M, d_go_on, d_opt_test, d_infoArray, d_init_lp_idxs, M, N, K, K_2, n_cycles_same_lps, batch_idx, output_flag, tol_minus);

        dual_pivot_kernel_2<<<n_active_lps, M>>>(dd_b, dd_non_b, d_th_idxs, d_min_idxs, d_c_copy, dd_c, dd_cb, dd_B_copy, dd_A, d_go_on, d_opt_test, d_infoArray, d_init_lp_idxs, d_lp_id_dict, n_cycles, M);

        if (n_active_lps < 30 && copy_idx < n_splits - 1) {
            int n_copy_lps = *(split_idxs+1) - *split_idxs;
            if (output_flag > 0) cout << "n_copy_lps: " << n_copy_lps << " = " << *(split_idxs+1) << " - " << *split_idxs << endl;
            
            copy_basis_dual_1<<<n_copy_lps, nTPB>>>(*split_idxs, dd_comb_bool, d_lp_id_dict, d_root_idxs, d_min_differ, d_z_final, n_c, nTPB);
            
            copy_basis_dual_2<<<(max_M_K*n_copy_lps+nTPB-1)/nTPB, nTPB>>>(*split_idxs, n_copy_lps, d_root_idxs, d_min_differ, d_b, d_non_b, d_A, d_B_copy, M, N, K, max_M_K);

            reset_d_go_on<<<(n_lps+nTPB-1)/nTPB, nTPB>>>(d_go_on, d_lp_idxs, *split_idxs, n_copy_lps, n_active_lps, n_lps);

            n_active_lps_new += n_copy_lps;
            n_active_lps += n_copy_lps;

            split_idxs++;
            copy_idx++;
            just_copy = true;
        }

        if (n_new_finished_lps > 0 || just_copy) {
            thrust::sort_by_key(thst_go_on.begin(), thst_go_on.end(), thst_lp_idxs.begin(), thrust::greater<bool>());
            move_kernel_dual<<<(n_active_lps+nTPB-1)/nTPB, nTPB>>>(n_active_lps, d_lp_idxs, d_init_lp_idxs, dd_b, dd_non_b, dd_A, dd_B, dd_B_copy, dd_cb, M, N, K);  // NOTE: do not need to move dd_cols since it will be reset in each cycle

        }

        if (n_active_lps == n_active_lps_new) {
            n_cycles_same_lps++;
        } else {
            n_cycles_same_lps = 0;
        }

        n_active_lps = n_active_lps_new;

        if (n_cycles_same_lps < 20) { 
            CUDA_CHECK(cudaMemcpyAsync(d_xb, d_xb_copy, sizeof(double) * n_active_lps * M, cudaMemcpyDeviceToDevice, stream));  // NOTE: In current problem: xb is same for all lps
            // CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            perturbation_kernel<<<(n_active_lps*M+nTPB-1)/nTPB, nTPB>>>(d_xb, d_xb_copy, 1, n_active_lps, M);
        }

        cum_n_active_lps += n_active_lps;

        if (output_flag > 0) cout << "n_active_lps: " << n_active_lps << endl;

        if (n_cycles_same_lps > 300 && copy_idx==n_splits-1) break;

        copy_B_kernel<<<(n_active_lps*M*M+nTPB-1)/nTPB, nTPB>>>(dd_B, dd_B_copy, n_active_lps, M * M);
    }

    /* free resources phase 1 */
    CUDA_CHECK(cudaFree(d_b_init));
    CUDA_CHECK(cudaFree(d_non_b_init));
    CUDA_CHECK(cudaFree(d_root_idxs));
    CUDA_CHECK(cudaFree(d_min_differ));
    CUDA_CHECK(cudaFree(d_non_b));
    CUDA_CHECK(cudaFree(d_non_b_counter));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_is_invertible));
    CUDA_CHECK(cudaFree(d_opt_test));
    CUDA_CHECK(cudaFree(d_th_star));
    CUDA_CHECK(cudaFree(d_th_idxs));

    CUDA_CHECK(cudaFree(dd_non_b));
    CUDA_CHECK(cudaFree(dd_non_b_counter));
    CUDA_CHECK(cudaFree(dd_cols));
    /* free resources phase 1 end */

    int n_feasible_lps = thrust::reduce(thst_is_feasible.begin(), thst_is_feasible.end(), 0, thrust::plus<int>());

    if (output_flag > 0) cout << "n_feasible_lps: " << n_feasible_lps << endl;

    int h_feasible_idxs[n_feasible_lps];
    double h_z[n_feasible_lps];

    if (n_feasible_lps > 0) {
        thrust::sequence(thst_lp_idxs.begin(), thst_lp_idxs.end());
        thrust::sort_by_key(thst_is_feasible.begin(), thst_is_feasible.end(), thst_lp_idxs.begin(), thrust::greater<bool>());
        CUDA_CHECK(cudaMemcpy(h_feasible_idxs, d_lp_idxs, sizeof(int) * n_feasible_lps, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_c_copy, 0, N_ * n_feasible_lps * sizeof(double)));
        
        prepare_phase2_kernel<<<(n_feasible_lps+nTPB-1)/nTPB, nTPB>>>(n_feasible_lps, d_lp_idxs, d_b, dd_b, d_A, dd_A, d_B_copy, dd_B_copy, d_c, d_c_copy, dd_c, d_cb, dd_cb, M, N, N_);
        
        if (output_flag > 0) cout << "cum_n_active_lps: " << cum_n_active_lps << endl;
        if (output_flag > 0) cout << "n_cycles: " << n_cycles << endl;

        if (output_flag > 0) cout << "Phase 1 time: " << double(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
        start = clock();

        if (output_flag > 0) cout << "Number of iterations: " << n_cycles << endl;

        /**************** Phase 2 ****************/
        n_active_lps = n_feasible_lps;

        if (output_flag > 0) printf("Phase 2 - n_active_lps: %d, M: %d, N_: %d\n", n_active_lps, M, N_);

        int *d_is_b = nullptr;
        int **dd_is_b = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_is_b), sizeof(int) * n_feasible_lps * N_));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dd_is_b), sizeof(int *) * n_feasible_lps));

        cudaMemset(d_is_b, 0, sizeof(int) * n_feasible_lps * N_);

        thrust::device_vector<bool> thst_go_on_2(n_feasible_lps, true);
        bool *d_go_on_2 = thrust::raw_pointer_cast(thst_go_on_2.data());
        thrust::device_vector<int> thst_lp_idxs_2(n_feasible_lps);
        int *d_lp_idxs_2 = thrust::raw_pointer_cast(thst_lp_idxs_2.data());
        thrust::sequence(thst_lp_idxs_2.begin(), thst_lp_idxs_2.end());

        initialise_is_b_cb_2<<<(n_feasible_lps*M+nTPB-1)/nTPB, nTPB>>>(dd_b, d_is_b, dd_is_b, d_c_copy, dd_cb, n_feasible_lps, M, N_);

        initialise_init_lp_idxs_2<<<(n_feasible_lps+nTPB-1)/nTPB, nTPB>>>(d_init_lp_idxs, n_feasible_lps, dd_b_counter, N_);

        copy_B_kernel<<<(n_feasible_lps*M*M+nTPB-1)/nTPB, nTPB>>>(dd_B, dd_B_copy, n_feasible_lps, M * M);
        
        for (n_cycles = 0; n_cycles < 800; n_cycles++)
        {
            if (output_flag > 0) cout << "************ " << n_cycles << " ************" << endl;

            CUBLAS_CHECK(cublasDgetrfBatched(handle, M, dd_B, M, d_pivotArray, d_infoArray, n_active_lps));

            CUDA_CHECK(cudaMemcpy(d_xb, d_xb_copy, sizeof(double) * n_active_lps * M, cudaMemcpyDeviceToDevice)); 
            
            copy_cb_kernel_1<<<(n_active_lps*M+nTPB-1)/nTPB, nTPB>>>(dd_col, dd_cb, n_active_lps, M);
            cublasDgetrsBatched(handle, CUBLAS_OP_T, M, 1, dd_B, M, d_pivotArray, dd_col, M, &h_infoArray, n_active_lps);  // dd_col = cb * B_I

            CUBLAS_CHECK(cublasDgetrsBatched(handle, CUBLAS_OP_N, M, 1, dd_B, M, d_pivotArray, dd_xb, M, &h_infoArray, n_active_lps));

            CUDA_CHECK(cudaMemcpy(d_c, d_c_copy, sizeof(double) * n_active_lps * N_, cudaMemcpyDeviceToDevice));
            
            cublasDgemvBatched(handle, CUBLAS_OP_T, M, N_, &alpha, dd_A, M, dd_col, 1, &beta, dd_c, 1, n_active_lps);  // dd_c = dd_c - dd_col * dd_A

            // *********** Find the minimum value in dd_c *******
            size_t shared_mem_size = N_*sizeof(double) + N_*sizeof(int);
            min_idx_kernel_optimal<<<n_active_lps, N_, shared_mem_size>>>(dd_c, dd_is_b, d_min_idxs, d_min_vals, d_go_on_2, dd_cb, dd_xb, d_z_final, dd_b_counter, d_init_lp_idxs, M, N_, M_2, N_2_, tol_minus_optimal);

            n_active_lps_new = thrust::reduce(thst_go_on_2.begin(), thst_go_on_2.begin() + n_active_lps, 0, thrust::plus<int>());
            n_new_finished_lps = n_active_lps - n_active_lps_new;

            // cout << "n_active_lps_new: " << n_active_lps_new << endl;
            if (!n_active_lps_new) break;
            
            copy_col_kernel<<<(n_active_lps*M+nTPB-1)/nTPB, nTPB>>>(dd_A, dd_col, d_min_idxs, n_active_lps*M, M);

            cublasDgetrsBatched(handle, CUBLAS_OP_N, M, 1, dd_B, M, d_pivotArray, dd_col, M, &h_infoArray, n_active_lps);  // B_I.dot(A[:, j])

            pivot_kernel_2<<<n_active_lps, M, M*sizeof(double) + M*sizeof(int)>>>(dd_col, dd_b, dd_is_b, dd_xb, d_min_idxs, d_c_copy, dd_cb, dd_B_copy, dd_A, n_active_lps*M, d_go_on_2, M, N_, M_2, tol_plus);

            if (n_new_finished_lps > 0) {
                thrust::sort_by_key(thst_go_on_2.begin(), thst_go_on_2.end(), thst_lp_idxs_2.begin(), thrust::greater<bool>());
                move_kernel_3<<<(n_active_lps+nTPB-1)/nTPB, nTPB>>>(n_active_lps, d_lp_idxs_2, d_lp_idxs, d_init_lp_idxs, dd_b, dd_is_b, dd_A, dd_B_copy, dd_cb, M, N, N_);
                // CUDA_CHECK(cudaDeviceSynchronize());
                n_active_lps = n_active_lps_new;
            }

            copy_B_kernel<<<(n_active_lps*M*M+nTPB-1)/nTPB, nTPB>>>(dd_B, dd_B_copy, n_active_lps, M * M);

            if (output_flag > 0) cout << "n_active_lps: " << n_active_lps_new << endl;
        }

        cudaMemcpyAsync(h_z, d_z_final, sizeof(double) * n_feasible_lps, cudaMemcpyDeviceToHost, stream);
        // CUDA_CHECK(cudaStreamSynchronize(stream));

        if (output_flag > 0) cout << "Phase 2 time: " << double(clock() - start) / CLOCKS_PER_SEC << "s" << endl;

        CUDA_CHECK(cudaFree(d_is_b));
        CUDA_CHECK(cudaFree(dd_is_b));
    }
    
    /* free resources phase 2 */
    CUDA_CHECK(cudaFree(d_infoArray));
    CUDA_CHECK(cudaFree(d_pivotArray));
    
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_b_counter));
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_B_copy));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_copy));
    CUDA_CHECK(cudaFree(d_cb));
    CUDA_CHECK(cudaFree(d_xb));
    CUDA_CHECK(cudaFree(d_xb_copy));
    CUDA_CHECK(cudaFree(d_col));
    
    CUDA_CHECK(cudaFree(d_z_final));
    CUDA_CHECK(cudaFree(d_min_idxs));
    CUDA_CHECK(cudaFree(d_init_lp_idxs));
    CUDA_CHECK(cudaFree(d_min_vals));

    CUDA_CHECK(cudaFree(dd_b));
    CUDA_CHECK(cudaFree(dd_b_counter));
    CUDA_CHECK(cudaFree(dd_A));
    CUDA_CHECK(cudaFree(dd_B));
    CUDA_CHECK(cudaFree(dd_B_copy));
    CUDA_CHECK(cudaFree(dd_c));
    CUDA_CHECK(cudaFree(dd_c_copy));
    CUDA_CHECK(cudaFree(dd_cb));
    CUDA_CHECK(cudaFree(dd_xb));
    CUDA_CHECK(cudaFree(dd_col));
    
    return {h_feasible_idxs, h_z, n_feasible_lps};
}
