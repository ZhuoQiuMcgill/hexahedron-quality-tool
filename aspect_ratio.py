import numpy as np


def aspect_ratio(hex_vert):
    """
    Compute the *Aspect Ratio* of a linear, 8‑node hexahedron.

    Definition
    ----------
    Aspect Ratio (AR) = max_edge_length / min_edge_length  ≥ 1

    * 1.0  → all edges have identical length (ideal “cube‑like” element)
    * >1   → the element is stretched; the larger the value, the poorer
             the element quality.

    Parameters
    ----------
    hex_vert : (8, 3) array‑like
        Corner‑node coordinates.
        The routine assumes the conventional FEM ordering
        (0‑1‑2‑3 bottom face counter‑clockwise, then 4‑5‑6‑7 top face, and
        vertical edges 0‑4, 1‑5, 2‑6, 3‑7).  If your ordering differs,
        reorder the vertices first.

    Returns
    -------
    float
        AR ≥ 1.  Returns ``np.inf`` if any edge collapses (zero length).
    """
    v = np.asarray(hex_vert, dtype=float).reshape(8, 3)

    # List of the 12 edges in the standard hexahedron connectivity
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
    ]

    lengths = [np.linalg.norm(v[i] - v[j]) for i, j in edges]
    l_min = min(lengths)
    l_max = max(lengths)

    if l_min == 0.0:
        return float("inf")  # degenerate element

    return l_max / l_min
