import numpy as np
import scipy.io
import ctypes
import time
import matplotlib.pyplot as plt
from utils import MPQPProb, active_sets_to_regions, active_sets_to_regions_parallel
from ppopt.mpqp_program import MPQP_Program
from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm


def calculate_mpqp_with_CUDA(data_set, test_ids):
    class GPUResults(ctypes.Structure):
        _fields_ = [
            ('n_batches', ctypes.c_int),
            ('n_splits', ctypes.c_int),
            ('n_feasible_lps', ctypes.c_int),
            ('n_final_lps', ctypes.c_int),
            ('time_feas', ctypes.c_double),
            ('time_opti', ctypes.c_double),
            ('time_GPU', ctypes.c_double),
            ('active_sets', ctypes.POINTER(ctypes.c_int))
        ]

    # Load the compiled cuda code
    MPQP_CUDA = ctypes.cdll.LoadLibrary("./mpqp_cuda.so")

    # Set function argument types
    MPQP_CUDA.mpqp.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # AxT
        ctypes.c_int,                     # n_c
        ctypes.c_int,                     # n_x
        ctypes.c_int,                     # n_t
        ctypes.c_int,                     # k_min
        ctypes.c_int,                     # k_max
        ctypes.c_int,                     # extra_batches
        ctypes.POINTER(ctypes.c_double),  # A1
        ctypes.POINTER(ctypes.c_double),  # xb1
        ctypes.POINTER(ctypes.c_int),     # b1
        ctypes.c_int,                     # M1
        ctypes.c_int,                     # N1
        ctypes.POINTER(ctypes.c_double),  # A2
        ctypes.POINTER(ctypes.c_double),  # xb2
        ctypes.POINTER(ctypes.c_int),     # b2
        ctypes.POINTER(ctypes.c_int),     # non_b2
        ctypes.c_int,                     # M2
        ctypes.c_int,                     # N2
        ctypes.c_int,                     # n_eqs
        ctypes.c_int                      # reserved_memory
    ]
    MPQP_CUDA.mpqp.restype = ctypes.POINTER(GPUResults)

    reserved_memory = 0

    time_feas_GPU = []
    time_opti_GPU = []
    time_GPU = []
    total_time = []
    cal_region_time = []

    MPQP_CUDA.warm_up()

    results = np.zeros((len(test_ids), 4))

    for idx, prob_id in enumerate(test_ids):  # arange(95, 100)

        print(f"============= {prob_id} =============")

        if prob_id == 90:
            reserved_memory = 2000

        problem = data_set[prob_id]

        A = problem['A']
        b = problem['b']
        c = problem['c']
        F = problem['F']
        Q = problem['Q']
        CRA = problem['CRA']
        CRb = problem['CRb']
        H = problem['Ht']

        prog = MPQPProb(A, b, c, H, Q, CRA, CRb, F)
        prog.process_constraints(scale_method=1)

        prog_copy = MPQPProb(A, b, c, H, Q, CRA, CRb, F)  # Create a copy using a normalisation method same to PPOPT for compare results

        num_c, num_x = prog.A.shape  # num_c: number of constraints for Axt, num_x: number of variables
        n_eqs = num_c + num_x  # number of equations in the optimality check
        num_cp, num_t = prog.A_t.shape  # num_cp: number of parameter constraints

        # Constrains for feasibility check
        A1_1 = np.hstack((prog.A, -prog.F))
        A1_2 = np.hstack((np.zeros((num_cp, num_x)), prog.A_t))
        A1 = np.vstack((A1_1, A1_2))
        M1 = A1.shape[0]
        A1 = np.hstack((A1, -A1.sum(axis=1, keepdims=True), np.eye(M1)))
        N1 = A1.shape[1]
        xb1 = np.vstack((prog.b, prog.b_t))

        A0 = A1[:num_c, :num_x].T

        # Constrains for optimality check
        A2_1 = np.hstack((np.zeros((num_c, num_x + num_t)), -np.eye(num_c), np.ones((num_c, 1))))
        A2_2 = np.hstack((np.zeros(num_x + num_t + num_c), -1))
        A2_3 = np.hstack((np.zeros((num_cp, num_x)), prog.A_t, np.zeros((num_cp, num_c + 1))))
        A2_4 = np.hstack((prog.Q, prog.H, prog.A.T, np.zeros((num_x, 1))))
        A2_5 = np.hstack((prog.A, -prog.F, np.eye(num_c), np.zeros((num_c, 1))))

        A2 = np.vstack((A2_1, A2_2, A2_3, A2_4, A2_5))
        xb2 = np.vstack((np.zeros((num_c + 1, 1)), prog.b_t, -prog.c, prog.b))

        neg_idx = xb2[:, 0] < 0
        xb2[neg_idx] = -xb2[neg_idx]
        A2[neg_idx, :] = -A2[neg_idx, :]
        M2 = A2.shape[0]

        A2 = np.hstack((A2, -A2[:, :num_x+num_t].sum(axis=1, keepdims=True), np.eye(M2)))
        N2 = A2.shape[1]

        # Initial basis for optimality check
        b1 = np.arange(N1 - M1, N1, dtype=np.intc)

        # Initial basis for dual simplex algorithm
        b2_1 = np.arange(num_x)
        b2_2 = np.arange(num_x + num_t, num_x + num_t + num_c)
        b2_3 = np.arange(N2 - M2, N2 - n_eqs)
        b2 = np.hstack((b2_1, b2_2, b2_3)).astype(np.intc)
        non_b2 = np.setdiff1d(np.arange(0, N2, dtype=np.intc), b2)

        A0 = np.asfortranarray(A0).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        A1 = np.asfortranarray(A1).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        A2 = np.asfortranarray(A2).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        xb1 = xb1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        xb2 = xb2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        b1 = b1.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        b2 = b2.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        non_b2 = non_b2.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        k_min = 0
        k_max = num_x
        extra_batches = 0

        start_time = time.time()
        results_GPU = MPQP_CUDA.mpqp(A0, num_c, num_x, num_t, k_min, k_max, extra_batches, A1, xb1, b1, M1, N1, A2, xb2, b2, non_b2, M2, N2, n_eqs, reserved_memory)
        add_region_time_start = time.time()

        prog_copy.process_constraints(scale_method=2)
        prog_copy.num_x = num_x

        prog_copy.n_active_sets = results_GPU.contents.n_final_lps
        prog.n_active_sets = results_GPU.contents.n_final_lps

        # TODO: use CUDA to accelerate active_sets_to_regions calculation
        if prog.n_active_sets < 200:
            solution = active_sets_to_regions(prog_copy, results_GPU.contents.active_sets)
        else:
            solution = active_sets_to_regions_parallel(prog_copy, results_GPU.contents.active_sets)

        total_time = time.time() - start_time
        add_region_time_each = time.time() - add_region_time_start

        print(f"{prob_id}: n_regions: {len(solution)}, GPU_time: {results_GPU.contents.time_GPU}, total_time: {total_time}")
        time_GPU.append(results_GPU.contents.time_GPU)
        time_feas_GPU.append(results_GPU.contents.time_feas)
        time_opti_GPU.append(results_GPU.contents.time_opti)
        cal_region_time.append(add_region_time_each)

        results[idx, 0] = prob_id
        results[idx, 1] = len(solution)
        results[idx, 2] = total_time
        results[idx, 3] = results_GPU.contents.time_GPU

    return results


def calculate_mpqp_with_PPOPT(data_set, test_ids, method = "Comb"):
    results = np.zeros((len(test_ids), 3))

    for idx, prob_id in enumerate(test_ids):
        problem = data_set[prob_id]

        A = problem['A']
        b = problem['b']
        c = problem['c']
        F = problem['F']
        Q = problem['Q']
        CRa = problem['CRA']
        CRb = problem['CRb']
        c_c = problem['cc']
        c_t = problem['ct']
        Q_t = problem['Qt']
        H = problem['Ht']

        prog = MPQP_Program(A, b, c, H, Q, CRa, CRb, F, c_c, c_t, Q_t)
        
        prog.process_constraints()

        start_time = time.time()
        if method == "Comb":
            solution = solve_mpqp(prog, mpqp_algorithm.combinatorial)
        elif method == "CombPara":
            solution = solve_mpqp(prog, mpqp_algorithm.combinatorial_parallel)

        total_time = time.time() - start_time

        results[idx, 0] = prob_id
        results[idx, 1] = len(solution.critical_regions)
        results[idx, 2] = total_time

    return results


if __name__ == '__main__':
    POP_mpQP1 = scipy.io.loadmat('POP_mpQP1.mat')

    # At a time ascend order
    # test_ids = [38,51,80,41,30,92,88,99,70,63,66,93,53,72,68,9,57,91,49,13,19,54,45,58,56,8,55,84,67,15,97,71,3,89,21,47,79,20,33,34,74,73,64,26,65,95,10,83,43,85,42,44,11,16,24,25,69,36,52,62,12,81,28,96,6,75,82,37,60,2,98,48,0,40,86,77,22,17,27,61,90,76,31,5,4,39,29,7,46,87,23,59,50,14,78,18,35,94,1,31]
    # test_ids = [38,51,80,41,30,92,88,99,70,63,66,93,53,72,68,9,57,91,49,13,19,54,45,58,56,8,55,84,67,15,97,71,3,89,21,47,79,20,33,34,74,73,64,26,65,95,10,83,43,85,42,44,11,16,24,25,69,36,52,62,12,81]
    test_ids = [38,51,80,41,30,92,88,99,70,63,66,93,53,72,68,9,57,91,49,13,19,54,45,58,56,8,55,84,67,15,97,71,3,89,21,47,79,20,33,34,74,73,64]

    data_set = POP_mpQP1['problem'][0]

    GPU_results = calculate_mpqp_with_CUDA(data_set, test_ids)
    PPOPT_results = calculate_mpqp_with_PPOPT(data_set, test_ids, method = "Comb")
    PPOPTPara_results = calculate_mpqp_with_PPOPT(data_set, test_ids, method = "CombPara")

    # Roughly calculation for accuracy
    print(f"The difference in results between GPU and PPOPT: {np.abs(GPU_results[:, 1] - PPOPT_results[:, 1]).sum()}")

    plt.figure(figsize=(10, 6))

    plt.plot(np.cumsum(GPU_results[:, 2]), range(len(test_ids)), label='GPU (RTX 4090)')
    plt.plot(np.cumsum(PPOPT_results[:, 2]), range(len(test_ids)), label='PPOPT')
    plt.plot(np.cumsum(PPOPTPara_results[:, 2]), range(len(test_ids)), label='PPOPT (Parallel)')

    plt.xscale('log')
    plt.xlabel('Time (Sec)')
    plt.ylabel('Percentage of POP_mpQP1 Solved (%)')

    plt.legend()
    plt.savefig('algorithms_performance.png')
    plt.show()
