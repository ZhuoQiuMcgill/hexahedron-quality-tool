import numpy as np


def OrthogonalQuality(hex_vert):
    """
    Orthogonal quality metric ( OQ ) for an eight‑node hexahedron.

    Idea
    ----
    At each vertex a perfect cube has three mutually perpendicular edge
    directions.  For every vertex *v* let the three incident (outgoing)
    edge vectors be **e₁**, **e₂**, **e₃**.  The worst deviation from
    orthogonality at that vertex is

        m_v = max( |cos ∠(**eᵢ**, **eⱼ**)| )    for i ≠ j.

    The vertex quality is  q_v = 1 − m_v, so q_v = 1 if all angles are
    exactly 90°, and decreases toward 0 as any angle departs from 90°.
    The element’s *orthogonal quality* is the minimum vertex quality:

        OQ = min_v  q_v     ∈ [0, 1].

    * 1 → perfectly orthogonal (ideal cube)
    * 0 → degenerate (colinear edges) or highly skewed

    Parameters
    ----------
    hex_vert : (8, 3) array‑like
        Coordinates of the hexahedron’s eight corner nodes in the standard
        FEM ordering (0‑1‑2‑3 bottom face CCW, 4‑5‑6‑7 top face).

    Returns
    -------
    float
        Orthogonal quality in the range [0, 1].  A value of ``0.0`` is
        returned if any incident edge has zero length.
    """
    v = np.asarray(hex_vert, dtype=float).reshape(8, 3)

    # Connectivity: list incident neighbour indices for each vertex
    nbr = [
        (1, 3, 4),  # 0
        (0, 2, 5),  # 1
        (1, 3, 6),  # 2
        (0, 2, 7),  # 3
        (0, 5, 7),  # 4
        (1, 4, 6),  # 5
        (2, 5, 7),  # 6
        (3, 4, 6)  # 7
    ]

    best = 1.0
    for i, nb in enumerate(nbr):
        # three outgoing, non‑repeated edges from vertex i
        e = [v[j] - v[i] for j in nb]
        lens = [np.linalg.norm(d) for d in e]

        if any(l == 0.0 for l in lens):  # collapsed edge
            return 0.0

        # normalize
        e = [d / l for d, l in zip(e, lens)]

        # max |cos| among the three angle pairs
        cos_vals = [abs(np.dot(e[a], e[b])) for a, b in ((0, 1), (1, 2), (0, 2))]
        m_v = max(cos_vals)

        q_v = 1.0 - m_v
        best = min(best, q_v)

    # Clip for numerical safety
    return float(np.clip(best, 0.0, 1.0))
