import matplotlib.pyplot as plt
import matplotlib.patches as patches
from z3 import Int, Distinct, Solver, And, Or, sat
import gurobipy as gp
from gurobipy import GRB
from typing import Any, Dict, List, Tuple
import ast

Seq = List[int]
Instance = Tuple[Seq, Seq]
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2)) with x1<=x2, y1<=y2

def solve_rect_permutation(rects, adj):
    """
    rects: dict {id: (x1_orig, y1_orig, x2_orig, y2_orig)}
    adj: dict {id: [neighbor_ids]} (undirected adjacency)
    
    Returns: new_rects {id: (x1, y1, x2, y2)} with integer assignments,
    such that:
      - Each side coordinate is a unique integer in [1..2*n]
      - For every adjacent (i, j): rectangles intersect in interior
      - For every non‑adjacent (i, j): rectangles do *not* intersect (strictly separated)
    """
    n = len(rects)
    s = Solver()

    # create variables
    vars = {}
    for rid in rects:
        x1 = Int(f"x1_{rid}")
        y1 = Int(f"y1_{rid}")
        x2 = Int(f"x2_{rid}")
        y2 = Int(f"y2_{rid}")
        vars[rid] = (x1, y1, x2, y2)
        # domain: 1 .. 2*n (so enough distinct values)
        # to allow distinct values for all rectangle sides
        s.add(x1 >= 1, x1 <= 2*n)
        s.add(x2 >= 1, x2 <= 2*n)
        s.add(y1 >= 1, y1 <= 2*n)
        s.add(y2 >= 1, y2 <= 2*n)
        # ensure rectangles are valid
        s.add(x1 < x2)
        s.add(y1 < y2)

    # all x1,x2 among all rects must be distinct; similarly y1,y2 distinct
    # ensures that every rectangle side is unique, so we have a permutation of coordinates
    # automatically breaks all ties without manually shifting coordinates
    all_x = []
    all_y = []
    for rid in vars:
        x1, y1, x2, y2 = vars[rid]
        all_x.append(x1)
        all_x.append(x2)
        all_y.append(y1)
        all_y.append(y2)
    s.add(Distinct(*all_x))
    s.add(Distinct(*all_y))

    # adjacency constraints: rectangles that are adjacent must overlap in interior
    print(adj)
    for i, nbrs in adj.items():
        xi1, yi1, xi2, yi2 = vars[i]
        for j in nbrs:
            # only enforce once (i<j)
            if j <= i:
                continue
            xj1, yj1, xj2, yj2 = vars[j]
            # guarantee that intersecting rectangles overlap in both x and y axes
            # so they still intersect after tie-breaking
            s.add(And(xi1 < xj2, xi2 > xj1, yi1 < yj2, yi2 > yj1))

    # non‑adjacency constraints: rectangles not in adj list must *not* intersect (strictly separated)
    all_ids = list(rects.keys())
    for idx_i, i in enumerate(all_ids):
        for j in all_ids[idx_i+1:]:
            if j in adj.get(i, []):
                continue
            xi1, yi1, xi2, yi2 = vars[i]
            xj1, yj1, xj2, yj2 = vars[j]
            # ensure that non-intersecting rectangles do not touch edges or interiors
            s.add(Or(xi2 <= xj1, xi1 >= xj2, yi2 <= yj1, yi1 >= yj2))

    # solve, trying to satisfy all constraints simultaneously
    # so - each rectangle side gets unique integer coordinates, and intersections and non-intersections are preserved
    if s.check() != sat:
        raise RuntimeError("No solution found")
    m = s.model()
    result = {}
    for rid in vars:
        x1, y1, x2, y2 = vars[rid]
        result[rid] = (m[x1].as_long(), m[y1].as_long(), m[x2].as_long(), m[y2].as_long())
    return result

def plot_rectangles_with_adjacency(original, new, adj, title_info=None):
    """
    original: dict {id: (x1, x2, y1, y2)}
    new: dict {id: (x1, y1, x2, y2)}
    adj: dict {id: [neighbor_ids]}
    """

    # first reorder original coordinates to match new format
    original = {id: (x1, y1, x2, y2) for id, (x1, x2, y1, y2) in original.items()}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    title_info = [old_ratio, old_lp, old_ilp, new_ratio, new_lp, new_ilp]
    axes[0].set_title(f"Original Rectangles (zoomed), Ratio = {title_info[0]:.3f}, LP={title_info[1]:.1f}, ILP={title_info[2]:.1f}")
    axes[1].set_title(f"Z3 Tie-Broken Rectangles, Ratio = {title_info[3]:.3f}, LP={title_info[4]:.1f}, ILP={title_info[5]:.1f}")
    
    cmap = plt.get_cmap("tab20")

    for ax, rects, zoom in zip(axes, [original, new], [True, False]):
        ax.set_aspect('equal')
        # set axis limits
        if zoom:
            all_x = [v[0] for v in rects.values()] + [v[2] for v in rects.values()]
            all_y = [v[1] for v in rects.values()] + [v[3] for v in rects.values()]
            ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
            ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
        else:
            ax.set_xlim(0, max(v[2] for v in new.values()) + 2)
            ax.set_ylim(0, max(v[3] for v in new.values()) + 2)

        # draw rectangles
        for rid, (x1, y1, x2, y2) in rects.items():
            color = cmap(rid % 20)
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=1, edgecolor=color,
                                     facecolor=color, alpha=0.3)
            ax.add_patch(rect)
            ax.text(x1 + 0.2, y1 + 0.2, str(rid), fontsize=8, color='black')

        # draw adjacency lines between rectangle centers
        for i, neighbors in adj.items():
            xi1, yi1, xi2, yi2 = rects[i]
            ci = ((xi1 + xi2)/2, (yi1 + yi2)/2)
            for j in neighbors:
                if j <= i:
                    continue
                xj1, yj1, xj2, yj2 = rects[j]
                cj = ((xj1 + xj2)/2, (yj1 + yj2)/2)
                ax.plot([ci[0], cj[0]], [ci[1], cj[1]], 'k--', lw=0.5, alpha=0.5)

    plt.show()

def plot_rects(old_rects: List[Rect], new_rects: List[Rect]):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7)) 
    axes[0].set_title("Old Rectangles")
    axes[1].set_title("New Rectangles")

    for ax, rects in zip(axes, [old_rects, new_rects]):
        for i, ((x1,x2),(y1,y2)) in enumerate(rects, start=1):
            w = (x2 - x1); h = (y2 - y1)
            rect = plt.Rectangle((x1, y1), w, h, fill=False)
            ax.add_patch(rect)
            cx = x1 + w/2.0; cy = y1 + h/2.0
            ax.text(cx, cy, str(i), ha='center', va='center', fontsize=9)

        xs = [x for r in rects for x in (r[0][0], r[1][0])]
        ys = [y for r in rects for y in (r[0][1], r[1][1])]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        ax.set_xlim(xmin-0.5, xmax+0.5)
        ax.set_ylim(ymin-0.5, ymax+0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("H index")
        ax.set_ylabel("V index")
        ax.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def compute_adjacency(rects):
    """
    rects: {id: (x1, y1, x2, y2)}
    Returns: adjacency dict where two rectangles are adjacent if they intersect in interior
    """
    adj_new = {rid: [] for rid in rects}
    ids = list(rects.keys())
    for i_idx, i in enumerate(ids):
        xi1, yi1, xi2, yi2 = rects[i]
        for j in ids[i_idx+1:]:
            xj1, yj1, xj2, yj2 = rects[j]
            # check interior intersection
            if xi1 < xj2 and xi2 > xj1 and yi1 < yj2 and yi2 > yj1:
                adj_new[i].append(j)
                adj_new[j].append(i)
    return adj_new

def compare_adjacency(adj_old, adj_new):
    """
    Compare two adjacency dicts, prints differences
    """
    print("Adj old: ", adj_old)
    print("Adj new: ",adj_new)
    all_ids = sorted(adj_old.keys())
    lost = []
    gained = []
    for i in all_ids:
        old_set = set(adj_old[i])
        new_set = set(adj_new[i])
        lost.extend((i,j) for j in old_set - new_set)
        gained.extend((i,j) for j in new_set - old_set)
    print(f"Lost adjacencies (were in old, missing in new): {lost}")
    print(f"Gained adjacencies (new intersections not in old): {gained}")

def grid_points(rects: List[Rect]) -> List[Tuple[int,int]]:
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    return [(x,y) for x in xs for y in ys]

def covers_grid_closed(rects: List[Rect], pts: List[Tuple[int,int]]) -> List[List[int]]:
    C=[]
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                S.append(i)
        C.append(S)
    return C

def _norm_pair(a: int, b: int) -> Tuple[int, int]:
    a, b = int(a), int(b)
    return (a, b) if a <= b else (b, a)

def from_original(rect: Tuple[int, int, int, int]) -> Rect:
    # original: (x1, x2, y1, y2)
    x1, x2, y1, y2 = rect
    return (_norm_pair(x1, x2), _norm_pair(y1, y2))

def from_new(rect: Tuple[int, int, int, int]) -> Rect:
    # new: (x1, y1, x2, y2)
    x1, y1, x2, y2 = rect
    return (_norm_pair(x1, x2), _norm_pair(y1, y2))

def convert_rects(
    rects: Dict[Any, Tuple[int, int, int, int]],
    new_rects: Dict[Any, Tuple[int, int, int, int]]
) -> Tuple[Dict[Any, Rect], Dict[Any, Rect]]:
    return ([from_original(v) for k, v in rects.items()],
            [from_new(v) for k, v in new_rects.items()])

def solve_lp_ilp(rects: List[Rect], grb_threads: int = 0) -> Tuple[float, float]:
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)

    # LP
    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_lp.setParam('Threads', grb_threads)
    n = len(rects)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0

    # ILP
    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_ilp.setParam('Threads', grb_threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0
    ratio = (lp/ilp) if ilp > 0 else 0.0
    print(f"Selected rectangles in LP: {[i for i in range(n) if x[i].X >= 0.5]}")
    print(f"Selected rectangles in ILP: {[i for i in range(n) if y[i].X >= 0.5]}")
    return ratio, lp, ilp

def seq_spans(seq: Seq) -> List[Tuple[int, int]]:
    """Return [l_i, r_i] (indices of the two occurrences) for labels i=1..n."""
    first = {}
    spans = {}
    for idx, lab in enumerate(seq):
        if lab not in first:
            first[lab] = idx
        else:
            spans[lab] = (first[lab], idx)
    n = max(seq) if seq else 0
    return [spans[i] for i in range(1, n + 1)]

def build_rects(H: Seq, V: Seq) -> List[Rect]:
    X = seq_spans(H); Y = seq_spans(V)
    rects = []
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

def invert_rects(rects: List[Rect]) -> Instance:
    """
    Given rects = [((x1,x2),(y1,y2)), ...] with x1<x2 and y1<y2,
    reconstruct H and V of length 2n such that build_rects(H,V) = rects.
    """

    n = len(rects)
    H = [0] * (2 * n)
    V = [0] * (2 * n)

    for label, ((x1, x2), (y1, y2)) in enumerate(rects, start=1):
        # Fill the two occurrences in H
        H[x1 - 1] = label
        H[x2 - 1] = label

        # Fill the two occurrences in V
        V[y1 - 1] = label
        V[y2 - 1] = label

    return H, V

from typing import Dict, Tuple

def load_rectangles_from_file(path: str) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Reads a rectangle coordinate file of the format:

        # comments...
        # node x1 x2 y1 y2
        0 8 14 7 8
        1 9 10 5 11
        ...

    Returns:
        {node_id: (x1, x2, y1, y2), ...}
    """
    rects = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip blank and comment lines.
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 5:
                # Ignore lines that do not match expected format.
                continue

            node, x1, x2, y1, y2 = map(int, parts)
            rects[node] = (x1, x2, y1, y2)

    return rects

def load_adjacency_dict(path: str) -> Dict[int, List[int]]:
    """
    Reads an adjacency dictionary stored in a file as a Python literal, e.g.:

        {0: [1, 5, 6, 15], 1: [0, 2, 8, 17], ...}

    Returns it as a Python dict: {int: [int, ...]}.
    """
    with open(path, "r") as f:
        text = f.read().strip()
    return ast.literal_eval(text)

if __name__ == "__main__":
    """
    #example 1
    rects = {
        0:(9,13,9,13),   1:(8,9,8,12),    2:(4,9,5,8),     3:(2,4,2,5),
        4:(0,12,3,4),    5:(10,11,0,16),  6:(13,14,1,15),  7:(7,13,15,16),
        8:(7,8,12,15),   9:(5,9,13,14),   10:(5,6,5,14),   11:(3,8,10,11),
        12:(2,3,5,11),   13:(0,4,6,7),    14:(0,1,2,6),    15:(0,7,1,2),
        16:(7,12,0,1),   17:(12,13,0,4),
    }
    adj = {0: [1, 5, 6, 9], 1: [0, 2, 8, 11], 2: [1, 3, 10, 13], 3: [2, 4, 12, 15],
           4: [3, 5, 14, 17], 5: [0, 4, 7, 16], 6: [0, 7, 17], 7: [5, 6, 8],
           8: [1, 7, 9], 9: [0, 8, 10], 10: [2, 9, 11], 11: [1, 10, 12],
           12: [3, 11, 13], 13: [2, 12, 14], 14: [4, 13, 15], 15: [3, 14, 16],
           16: [5, 15, 17], 17: [4, 6, 16]}
    """
    """
    #example 2
    rects = {
        0:(4,7,8,13),    1:(7,13,11,12),  2:(11,12,1,11),  3:(1,14,0,1),
        4:(0,1,0,13),    5:(1,4,13,14),   6:(7,10,8,10),   7:(8,9,10,15),
        8:(4,9,14,15),   9:(5,6,7,15),    10:(1,5,6,7),    11:(2,3,4,13),
        12:(2,3,1,4),    13:(0,4,2,3),    14:(4,13,2,4),   15:(13,14,0,5),
        16:(13,14,5,11), 17:(10,13,8,9),
    }
    adj = {0: [1, 5, 6, 9], 1: [0, 2, 7, 16], 2: [1, 3, 14, 17], 3: [2, 4, 12, 15], 
           4: [3, 5, 10, 13], 5: [0, 4, 8, 11], 6: [0, 7, 17], 7: [1, 6, 8], 
           8: [5, 7, 9], 9: [0, 8, 10], 10: [4, 9, 11], 11: [5, 10, 12], 
           12: [3, 11, 13], 13: [4, 12, 14], 14: [2, 13, 15], 15: [3, 14, 16], 
           16: [1, 15, 17], 17: [2, 6, 16]}
    """

    rects = load_rectangles_from_file("configs/20_40.config")
    adj = load_adjacency_dict("adjlists/20_40.adjlist")

    print("Breaking ties...")
    new_rects = solve_rect_permutation(rects, adj)
    # print results
    print(new_rects)

    # calculate ratio
    orig_conv, new_conv = convert_rects(rects, new_rects)
    old_ratio, old_lp, old_ilp = solve_lp_ilp(orig_conv, grb_threads=8)
    new_ratio,new_lp, new_ilp = solve_lp_ilp(new_conv, grb_threads=8)
    title_info = [old_ratio, old_lp, old_ilp, new_ratio, new_lp, new_ilp]
    # ensure the configuration is still the same
    adj_computed = compute_adjacency(new_rects)
    compare_adjacency(adj, adj_computed)    

    # plot both
    plot_rectangles_with_adjacency(rects, new_rects, adj, title_info=title_info)

    # convert to H and V encoding
    new_rects = [((v[0], v[2]), (v[1], v[3])) for k,v in new_rects.items()]
    H, V = invert_rects(new_rects)

    print("H = ", H, "/nV = ", V)
    reconstructed_rects = build_rects(H, V)
    print("Reconstructed rects:", reconstructed_rects)

    #plot both
    plot_rects(new_rects, reconstructed_rects)
    
