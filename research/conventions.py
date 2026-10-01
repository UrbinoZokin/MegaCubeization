"""Phase 0 research: measure geometry conventions from PUBLIC sample saves.

The class->primitive mapping needs pivot offsets and axis conventions that are not stored
in the save. Instead of guessing them, this script measures how pieces snap to each other
in real saves (the parser repository's own test saves), e.g. "a wall standing on a 1 m
foundation sits 50 cm above the foundation's origin".

Usage:
    python research/conventions.py dump1.json [dump2.json ...]

where each dump comes from research/export_transforms.js.
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

FOUNDATION_H = {"Build_Foundation_8x1_01_C": 100, "Build_Foundation_8x2_01_C": 200, "Build_Foundation_8x4_01_C": 400}
RAMP_H = {"Build_Ramp_8x1_01_C": 100, "Build_Ramp_8x2_01_C": 200, "Build_Ramp_8x4_01_C": 400}
WALLS = ["Build_Wall_8x4_01_C", "Build_Wall_Orange_8x1_C"]


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N,4) quaternions (x, y, z, w) -> (N,3,3) rotation matrices (column-vector convention)."""
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def peaks(values, n=6):
    return collections.Counter(np.round(values).astype(int).tolist()).most_common(n)


class Dataset:
    def __init__(self, paths):
        rows, self.conveyors = [], []
        for p in paths:
            d = json.load(open(p))
            rows += d["buildables"]
            self.conveyors += d["conveyors"]
        self.cls = np.array([r[0] for r in rows])
        self.P = np.array([r[1:4] for r in rows], float)
        self.Q = np.array([r[4:8] for r in rows], float)
        self.S = np.array([r[8:11] for r in rows], float)

    def idx(self, names):
        return np.where(np.isin(self.cls, list(names)))[0]


def foundations(ds: Dataset):
    print("\n## Foundations: neighbour spacing and stacking")
    for name, h in FOUNDATION_H.items():
        i = ds.idx([name])
        if not len(i):
            continue
        tree, R = cKDTree(ds.P[i]), quat_to_mat(ds.Q[i])
        same_level, stacked = [], []
        for k, a in enumerate(i):
            for j in tree.query_ball_point(ds.P[a], 850):
                if i[j] == a:
                    continue
                d = R[k].T @ (ds.P[i[j]] - ds.P[a])
                if abs(d[2]) < 1:
                    same_level.append(max(abs(d[0]), abs(d[1])))
                elif abs(d[0]) < 1 and abs(d[1]) < 1:
                    stacked.append(abs(d[2]))
        print(f"{name} (n={len(i)}): same-level neighbour offset peaks {peaks(same_level, 3)}; "
              f"vertical stacking offset peaks {peaks(stacked, 4)}")


def walls_on_foundations(ds: Dataset):
    print("\n## Walls standing on foundation edges (wall origin in the foundation frame)")
    iF = ds.idx(FOUNDATION_H)
    treeF, RF = cKDTree(ds.P[iF]), quat_to_mat(ds.Q[iF])
    for wall in WALLS:
        iw = ds.idx([wall])
        RW = quat_to_mat(ds.Q[iw])
        dz = collections.defaultdict(list)
        axes = collections.Counter()
        for k, a in enumerate(iw):
            for j in treeF.query_ball_point(ds.P[a], 700):
                d = RF[j].T @ (ds.P[a] - ds.P[iF[j]])
                if abs(abs(d[0]) - 400) < 1 and abs(d[1]) < 401:
                    edge_axis = 1
                elif abs(abs(d[1]) - 400) < 1 and abs(d[0]) < 401:
                    edge_axis = 0
                else:
                    continue
                fname = ds.cls[iF[j]]
                dz[fname].append(d[2])
                if abs(d[2] - FOUNDATION_H[fname] / 2) < 1:  # standing on the top surface
                    M = RF[j].T @ RW[k]
                    width_axis = int(np.argmax(np.abs(M[edge_axis, :])))
                    axes["XYZ"[width_axis]] += 1
        for fname, v in dz.items():
            print(f"{wall} on {fname}: n={len(v)}, local dz peaks {peaks(v, 4)}  (foundation half-height {FOUNDATION_H[fname] / 2:g})")
        print(f"{wall}: wall-local axis running along the foundation edge (= wall width axis): {axes.most_common()}")


def ramps(ds: Dataset):
    print("\n## Ramps next to foundations (which local side is the high end?)")
    iF = ds.idx(FOUNDATION_H)
    treeF = cKDTree(ds.P[iF])
    for name, h in RAMP_H.items():
        ir = ds.idx([name])
        RR = quat_to_mat(ds.Q[ir])
        top_flush, dz = collections.Counter(), []
        for k, a in enumerate(ir):
            for j in treeF.query_ball_point(ds.P[a], 900):
                d = RR[k].T @ (ds.P[iF[j]] - ds.P[a])  # foundation origin in the ramp frame
                if not (abs(np.hypot(d[0], d[1]) - 800) < 1 and (abs(d[0]) < 1 or abs(d[1]) < 1)):
                    continue
                hf = FOUNDATION_H[ds.cls[iF[j]]]
                if ds.cls[iF[j]] == name.replace("Ramp", "Foundation"):
                    dz.append(d[2])
                if abs((d[2] + hf / 2) - h / 2) < 1:  # foundation top flush with ramp bbox top
                    top_flush["+X" if d[0] > 400 else "-X" if d[0] < -400 else "+Y" if d[1] > 400 else "-Y"] += 1
        print(f"{name} (n={len(ir)}): same-height foundation neighbour dz peaks {peaks(dz, 3)}; "
              f"side where a foundation is flush with the ramp's top: {top_flush.most_common(4)}")


def quaternion_convention(ds: Dataset):
    """Is the save's quaternion applied as v' = R(q) v (standard) or as its inverse?

    Ramps are asymmetric, so the side of the high end must come out the same for every yaw.
    Only the correct convention is consistent at 90/270 degrees.
    """
    print("\n## Quaternion convention check (ramp high-end side by yaw)")
    iF = ds.idx(FOUNDATION_H)
    treeF = cKDTree(ds.P[iF])
    for name, h in RAMP_H.items():
        ir = ds.idx([name])
        RR = quat_to_mat(ds.Q[ir])
        votes = {"standard": collections.Counter(), "inverse": collections.Counter()}
        for k, a in enumerate(ir):
            for j in treeF.query_ball_point(ds.P[a], 900):
                w = ds.P[iF[j]] - ds.P[a]
                for conv, M in (("standard", RR[k].T), ("inverse", RR[k])):
                    d = M @ w
                    if not (abs(np.hypot(d[0], d[1]) - 800) < 1 and (abs(d[0]) < 1 or abs(d[1]) < 1)):
                        continue
                    if abs((d[2] + FOUNDATION_H[ds.cls[iF[j]]] / 2) - h / 2) < 1:
                        votes[conv]["+X" if d[0] > 400 else "-X" if d[0] < -400 else "+Y" if d[1] > 400 else "-Y"] += 1
        print(f"{name}: standard {votes['standard'].most_common(2)} | inverse {votes['inverse'].most_common(2)}")


def conveyors(ds: Dataset):
    print("\n## Belts and lifts relative to the foundation top surface")
    iF = ds.idx(FOUNDATION_H)
    tops = ds.P[iF, 2] + np.array([FOUNDATION_H[c] for c in ds.cls[iF]]) / 2
    tree = cKDTree(ds.P[iF, :2])
    belt_dz, first_arrive, n_points, lift_dz, lift_h = (collections.Counter() for _ in range(5))

    def above_top(w):
        out = []
        for j in tree.query_ball_point(w[:2], 400 * 1.42):
            if abs(w[0] - ds.P[iF[j], 0]) <= 400 and abs(w[1] - ds.P[iF[j], 1]) <= 400 and -50 < w[2] - tops[j] < 400:
                out.append(round(float(w[2] - tops[j])))
        return out

    for b in ds.conveyors:
        t = b["t"]
        R = quat_to_mat(np.array([[t["rotation"][k] for k in "xyzw"]]))[0]
        o = np.array([t["translation"][k] for k in "xyz"])
        if "pts" in b:
            n_points[len(b["pts"])] += 1
            first_arrive[round(float(np.linalg.norm(b["pts"][0][1])), 2)] += 1
            for loc, _, _ in b["pts"]:
                belt_dz.update(above_top(o + R @ np.array(loc)))
        else:
            lift_h[round(b["top"][2])] += 1
            lift_dz.update(above_top(o))
    print(f"belt spline point height above foundation top: {belt_dz.most_common(6)}")
    print(f"spline points per belt: {n_points.most_common(6)}; |ArriveTangent| of first point: {first_arrive.most_common(3)}")
    print(f"lift origin height above foundation top: {lift_dz.most_common(4)}; lift top translation z: {lift_h.most_common(6)}")


def signs(ds: Dataset):
    print("\n## Signs mounted on walls (sign axes in the wall frame)")
    iw = ds.idx(["Build_Wall_8x4_01_C", "Build_Wall_Concrete_8x4_C", "Build_Wall_Window_8x4_02_C"])
    RW = quat_to_mat(ds.Q[iw])
    tree = cKDTree(ds.P[iw] + RW[:, :, 2] * 200)  # wall centres (origin is bottom centre, 4 m tall)
    isg = np.where(np.char.find(ds.cls.astype(str), "StandaloneWidgetSign") >= 0)[0]
    print("sign classes:", collections.Counter(ds.cls[isg].tolist()).most_common())
    stats = collections.defaultdict(collections.Counter)
    for a in isg:
        Rs = quat_to_mat(ds.Q[a][None])[0]
        best = None
        for j in tree.query_ball_point(ds.P[a], 600):
            d = RW[j].T @ (ds.P[a] - ds.P[iw[j]])
            if abs(d[1]) <= 400 and -1 <= d[2] <= 401 and abs(d[0]) < 60 and (best is None or abs(d[0]) < abs(best[1][0])):
                best = (j, d)
        if best is None:
            continue
        j, d = best
        M = RW[j].T @ Rs  # columns: sign-local X, Y, Z expressed in the wall frame
        n_axis = int(np.argmax(np.abs(M[0, :])))
        away = int(np.sign(M[0, n_axis] * np.sign(d[0])))
        up_axis = int(np.argmax(np.abs(M[2, :])))
        key = (f"normal=local {'XYZ'[n_axis]} (+ points away from wall: {away > 0})",
               f"up=local {'XYZ'[up_axis]}{'+' if M[2, up_axis] > 0 else '-'}",
               f"origin {round(abs(d[0]))} cm from wall centre plane")
        stats[ds.cls[a]][key] += 1
    for k, v in stats.items():
        print(k, v.most_common(3))


if __name__ == "__main__":
    ds = Dataset(sys.argv[1:])
    print(f"{len(ds.cls)} placed objects; non-unit scale: {int((np.abs(ds.S - 1).max(1) > 1e-6).sum())}")
    foundations(ds)
    walls_on_foundations(ds)
    ramps(ds)
    quaternion_convention(ds)
    conveyors(ds)
    signs(ds)
