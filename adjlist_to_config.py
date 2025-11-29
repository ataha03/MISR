#!/usr/bin/env python3
# z3_rectangle_solver.py
from z3 import Solver, Int, And, Or, Not, sat, unsat, set_param
import os
import re

def solve_rectangle_problem(n, adjacency_list):
    """Solve and display rectangle intersection constraints using Z3."""
    s = Solver()
    s.set("timeout", 300000)  # 30 seconds

    grid_size = 2 * n
    # declare variables with same names as in the earlier code
    x1 = [Int(f"x1_{i}") for i in range(n)]
    x2 = [Int(f"x2_{i}") for i in range(n)]
    y1 = [Int(f"y1_{i}") for i in range(n)]
    y2 = [Int(f"y2_{i}") for i in range(n)]

    # bounds and positive-area constraints
    for i in range(n):
        s.add(x1[i] >= 0,        x1[i] <= grid_size - 2)
        s.add(x2[i] >= 1,        x2[i] <= grid_size - 1)
        s.add(y1[i] >= 0,        y1[i] <= grid_size - 2)
        s.add(y2[i] >= 1,        y2[i] <= grid_size - 1)
        s.add(x1[i] <= x2[i] - 1)
        s.add(y1[i] <= y2[i] - 1)

    # pairwise constraints (preserve same logical formulation)
    for i in range(n):
        for j in range(i + 1, n):
            x_overlap = Or(
                And(x1[i] <= x1[j], x1[j] <= x2[i]),
                And(x1[i] <= x2[j], x2[j] <= x2[i]),
                And(x1[j] <= x1[i], x1[i] <= x2[j]),
                And(x1[j] <= x2[i], x2[i] <= x2[j])
            )
            y_overlap = Or(
                And(y1[i] <= y1[j], y1[j] <= y2[i]),
                And(y1[i] <= y2[j], y2[j] <= y2[i]),
                And(y1[j] <= y1[i], y1[i] <= y2[j]),
                And(y1[j] <= y2[i], y2[i] <= y2[j])
            )

            if j in adjacency_list.get(i, []):
                # must overlap both dims
                s.add(x_overlap)
                s.add(y_overlap)
            else:
                # cannot overlap in both dims simultaneously
                s.add(Or(Not(x_overlap), Not(y_overlap)))

    #Print out constraints
    constraints = s.assertions()

    print(f"Total number of constraints: {len(constraints)}")
    print("Constraints:")
    for cons in constraints[:20]: # print only first 20 constraints for brevity
        print(cons)

    print(f"Solving for {n} rectangles with adjacency list: {adjacency_list}")
    if s.check() == sat:
        m = s.model()
        print("\nSolution found! Rectangle coordinates:")
        for i in range(n):
            print(f"  Rect {i}: x=[{m[x1[i]]}, {m[x2[i]]}], y=[{m[y1[i]]}, {m[y2[i]]}]")

        # verification
        print("\nOverlap verification:")
        for i in range(n):
            for j in range(i + 1, n):
                x1_i, x2_i = m[x1[i]].as_long(), m[x2[i]].as_long()
                y1_i, y2_i = m[y1[i]].as_long(), m[y2[i]].as_long()
                x1_j, x2_j = m[x1[j]].as_long(), m[x2[j]].as_long()
                y1_j, y2_j = m[y1[j]].as_long(), m[y2[j]].as_long()

                x_overlap = (x1_i <= x1_j <= x2_i) or (x1_i <= x2_j <= x2_i) or \
                            (x1_j <= x1_i <= x2_j) or (x1_j <= x2_i <= x2_j)
                y_overlap = (y1_i <= y1_j <= y2_i) or (y1_i <= y2_j <= y2_i) or \
                            (y1_j <= y1_i <= y2_j) or (y1_j <= y2_i <= y2_j)

                has_edge = j in adjacency_list.get(i, [])
                print(f"  R{i}-R{j}: x_overlap={x_overlap}, y_overlap={y_overlap}, "
                      f"edge_expected={has_edge}, consistent={((x_overlap and y_overlap) == has_edge)}")
        return m
    else:
        print("No solution found")
        if s.check() == unsat:
            unsat_core = s.unsat_core()
            print("Unsatisfied constraints:", unsat_core)
        return None

def load_adjacency_list(adj_list):
    n = len(adj_list)

    solution = solve_rectangle_problem(n, adj_list)

    if solution:

        # reconstruct variable objects to read values from the model
        x1_vars = [Int(f"x1_{i}") for i in range(n)]
        x2_vars = [Int(f"x2_{i}") for i in range(n)]
        y1_vars = [Int(f"y1_{i}") for i in range(n)]
        y2_vars = [Int(f"y2_{i}") for i in range(n)]

        print("# Rectangle coordinates:")
        print("# node x1 x2 y1 y2")
        for i in range(n):
            xi1 = solution[x1_vars[i]].as_long()
            xi2 = solution[x2_vars[i]].as_long()
            yi1 = solution[y1_vars[i]].as_long()
            yi2 = solution[y2_vars[i]].as_long()
            print(f"{i} {xi1} {xi2} {yi1} {yi2}")

if __name__ == "__main__":
    adj_list = {0: [1, 19, 59, 22], 1: [2, 21, 24], 2: [3, 23, 26], 3: [4, 25, 28], 4: [5, 27, 30], 5: [6, 29, 32], 6: [7, 31, 34], 7: [8, 33, 36], 8: [9, 35, 38], 9: [10, 37, 40], 10: [11, 39, 42], 11: [12, 41, 44], 12: [13, 43, 46], 13: [14, 45, 48], 14: [15, 47, 50], 15: [16, 49, 52], 16: [17, 51, 54], 17: [18, 53, 56], 18: [19, 55, 58], 19: [57, 20], 20: [21, 59], 21: [22], 22: [23], 23: [24], 24: [25], 25: [26], 26: [27], 27: [28], 28: [29], 29: [30], 30: [31], 31: [32], 32: [33], 33: [34], 34: [35], 35: [36], 36: [37], 37: [38], 38: [39], 39: [40], 40: [41], 41: [42], 42: [43], 43: [44], 44: [45], 45: [46], 46: [47], 47: [48], 48: [49], 49: [50], 50: [51], 51: [52], 52: [53], 53: [54], 54: [55], 55: [56], 56: [57], 57: [58], 58: [59], 59: []}
    load_adjacency_list(adj_list)