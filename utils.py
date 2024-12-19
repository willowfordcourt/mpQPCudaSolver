import numpy as np
from ppopt.utils.constraint_utilities import (
    find_implicit_equalities,
    find_redundant_constraints,
    process_program_constraints,
)
from ppopt.utils.general_utils import (
    ppopt_block
)
from ppopt.solver import Solver

import gurobipy as gp
from gurobipy import GRB

from multiprocessing import Pool


def scale_constraint(A, b):
    norm_val = 1.0 / np.linalg.norm(A, axis=1, keepdims=True)
    return [A * norm_val, b * norm_val]


def compute_critical_region(A, b, c, H, inv_Q, F, CRA, CRb, active_set):
    num_t = CRA.shape[1]
    aux = A[active_set] @ inv_Q
    auxinv = np.linalg.pinv(aux @ A[active_set].T)

    inactive = [i for i in range(A.shape[0]) if i not in active_set]

    lagrange_A = -auxinv @ (aux @ H + F[active_set])
    lagrange_b = -auxinv @ (b[active_set] + aux @ c)

    aux = inv_Q @ A[active_set].T

    parameter_A = -aux @ lagrange_A - inv_Q @ H
    parameter_b = -aux @ lagrange_b - inv_Q @ c

    # lagrange constraints
    lambda_A, lambda_b = -lagrange_A[0:], lagrange_b[0:]
    
    # Inactive Constraints remain inactive
    inactive_A = A[inactive] @ parameter_A - F[inactive]
    inactive_b = b[inactive] - A[inactive] @ parameter_b

    # we need to check for zero rows
    lamba_nonzeros = [i for i, t in enumerate(lambda_A) if np.nonzero(t)[0].shape[0] > 0]
    ineq_nonzeros = [i for i, t in enumerate(inactive_A) if np.nonzero(t)[0].shape[0] > 0]

    # Block of all critical region constraints
    lambda_Anz = lambda_A[lamba_nonzeros]
    lambda_bnz = lambda_b[lamba_nonzeros]

    inactive_Anz = inactive_A[ineq_nonzeros]
    inactive_bnz = inactive_b[ineq_nonzeros]

    CR_A = np.vstack((lambda_Anz, inactive_Anz, CRA))
    CR_b = np.vstack((lambda_bnz, inactive_bnz, CRb))

    CR_As, CR_bs = scale_constraint(CR_A, CR_b)

    ############
    CR_cs = np.zeros((num_t + 1, 1))
    CR_cs[num_t][0] = -1

    const_norm = np.linalg.norm(CR_As, axis=1, keepdims=True)
    const_norm = const_norm[:, :1]

    A_ball = np.block([[CR_As, const_norm], [CR_cs.T]])
    b_ball = np.concatenate((CR_bs, np.zeros((1, 1))))

    ############
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    x = model.addMVar(num_t + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    model.setParam("Method", 0)

    model.addMConstr(A_ball, x, '<', b_ball.flatten())

    model.setObjective(x[-1], sense=GRB.MAXIMIZE)

    model.optimize()
    model.update()

    status = model.status
    # if not solved return None
    if (status != GRB.OPTIMAL and status != GRB.SUBOPTIMAL) or x.X[-1] < 1e-8:
        return None
    
    return {
        "parameter_A": parameter_A,
        "parameter_b": parameter_b,
        "lagrange_A": lagrange_A,
        "lagrange_b": lagrange_b,
        "CR_As": CR_As,
        "CR_bs": CR_bs,
        "active_set": active_set
    }


def process_active_set(args):
    A, b, c, H, Q, F, CRA, CRb, active_set = args
    return compute_critical_region(A, b, c, H, Q, F, CRA, CRb, active_set)


def active_sets_to_regions_parallel(prog, all_active_sets):
    A, b, c, H, Q, F, CRA, CRb = prog.A, prog.b, prog.c, prog.H, prog.inv_Q, prog.F, prog.A_t, prog.b_t

    active_sets = []
    for lp_idx in range(prog.n_active_sets):
        active_set = []
        for ele_idx in range(prog.num_x):
            if all_active_sets[lp_idx * prog.num_x + ele_idx] == -1:
                break
            else:
                active_set.append(all_active_sets[lp_idx * prog.num_x + ele_idx])
        active_sets.append((A, b, c, H, Q, F, CRA, CRb, active_set))

    solution = []
    with Pool() as pool:
        results = pool.map(process_active_set, active_sets)
        for region in results:
            if region is not None:
                solution.append(region)

    return solution


def active_sets_to_regions(prog, all_active_sets):
    A, b, c, H, Q, F, CRA, CRb = prog.A, prog.b, prog.c, prog.H, prog.inv_Q, prog.F, prog.A_t, prog.b_t

    solution = []
    for lp_idx in range(prog.n_active_sets):
        active_set = []
        for ele_idx in range(prog.num_x):
            if all_active_sets[lp_idx * prog.num_x + ele_idx] == -1:
                break
            else:
                active_set.append(all_active_sets[lp_idx * prog.num_x + ele_idx])

        region = compute_critical_region(A, b, c, H, Q, F, CRA, CRb, active_set)
        if region is not None:
            solution.append(region)

    return solution


class MPQPProb:
    def __init__(self, A, b, c, H, Q, CRa, CRb, F):
        self.A = A
        self.b = b
        self.c = c
        self.H = H
        self.Q = Q
        self.inv_Q = np.linalg.pinv(Q)
        self.A_t = CRa
        self.b_t = CRb
        self.F = F
        self.equality_indices = []
        self.solver = Solver()
        self.num_x = A.shape[1]
        self.n_active_sets = 0
    
    def process_constraints(self, scale_method=1):
        # Use the same way as PPOPT to handle constraints in order to compare results
        self.A, self.b, self.F, self.A_t, self.b_t = process_program_constraints(self.A, self.b, self.F, self.A_t,
                                                                                 self.b_t)

        if scale_method == 2:
            self.scale_constraints()

        self.A, self.b, self.F, self.equality_indices = find_implicit_equalities(self.A, self.b, self.F,
                                                                                 self.equality_indices)
        
        problem_A = ppopt_block([[self.A, -self.F], [np.zeros((self.A_t.shape[0], self.A.shape[1])), self.A_t]])
        problem_b = ppopt_block([[self.b], [self.b_t]])

        saved_indices = find_redundant_constraints(problem_A, problem_b, self.equality_indices,
                                                   solver=self.solver.solvers['lp'])
        
        saved_upper = [x for x in saved_indices if x < self.A.shape[0]]
        saved_lower = [x - self.A.shape[0] for x in saved_indices if x >= self.A.shape[0]]

        # remove redundant constraints
        self.A = self.A[saved_upper]
        self.F = self.F[saved_upper]
        self.b = self.b[saved_upper]

        # remove redundant constraints from the parametric constraints
        self.A_t = self.A_t[saved_lower]
        self.b_t = self.b_t[saved_lower]

        if scale_method == 1:
            QHc = np.block([self.Q, self.H, self.c])
            QHc_norm = np.max(np.abs(QHc)) / (2 ** 3)
            self.scale_AF()
            self.scale_At()
            self.Q = self.Q / QHc_norm
            self.H = self.H / QHc_norm
            self.c = self.c / QHc_norm

    def scale_constraints(self):
        H = np.block([self.A, -self.F])
        norm = np.linalg.norm(H, axis=1, keepdims=True)
        self.A = self.A / norm
        self.b = self.b / norm
        self.F = self.F / norm

    def scale_AF(self):
        """
        Scales the problem according to equilibration.
        Also normalises the right hand side vector by its maximum element.
        """
        A = np.block([self.A, self.F])

        R = np.max(np.abs(A), axis=1)
        R[R == 0] = 1
        R = 1 / R

        self.A = self.A*R.reshape(-1, 1)
        self.F = self.F*R.reshape(-1, 1)
        self.b = self.b*R.reshape(-1, 1)

    def scale_At(self):
        R = np.max(np.abs(self.A_t), axis=1)
        R[R == 0] = 1
        R = 1 / R

        self.A_t = self.A_t*R.reshape(-1, 1)
        self.b_t = self.b_t*R.reshape(-1, 1)
