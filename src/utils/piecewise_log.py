import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class _PiecewiseLogLog:
    def __init__(self, E, V, edge_flags=None):
        self.E = np.asarray(E, float)
        self.V = np.asarray(V, float)
        if edge_flags is None:
            edge_flags = np.array([""]*len(self.E))
        self.edge = np.asarray(edge_flags)
        seg_starts = [0]
        for i in range(len(self.E)):
            s = str(self.edge[i]).strip()
            if s:
                if i not in seg_starts:
                    seg_starts.append(i)
        seg_starts = sorted(set(seg_starts))
        self._segs = []
        for k, s in enumerate(seg_starts):
            e = (seg_starts[k+1]-1) if k+1 < len(seg_starts) else (len(self.E)-1)
            if e == s and s > 0:
                s -= 1
            Es = self.E[s:e+1]; Vs = self.V[s:e+1]
            m = (Es > 0) & (Vs > 0)
            Es, Vs = Es[m], Vs[m]
            if len(Es) < 2:
                continue
            f = interp1d(np.log(Es), np.log(Vs), kind="linear",
                         bounds_error=False, fill_value="extrapolate",
                         assume_sorted=True)
            self._segs.append((Es[0], Es[-1], f))
        if not self._segs:
            raise ValueError("no valid segments for log-log interpolation")

    def __call__(self, E_keV):
        E = np.asarray(E_keV, float)
        out = np.empty_like(E, float)
        for idx, e in np.ndenumerate(E):
            seg = None
            for (l, r, f) in self._segs:
                if (e >= l) and (e <= r or r == self._segs[-1][1]):
                    seg = (l, r, f); break
            if seg is None:
                seg = self._segs[0] if e < self._segs[0][0] else self._segs[-1]
            out[idx] = np.exp(seg[2](np.log(max(e, 1e-12))))
        return out
