import numpy as np


def Skewness(hex_vert):
    """
    Face‑angle skewness of a linear, eight‑node hexahedron.

    Definition
    ----------
    For every face, compute the interior angle θ at each of its four
    vertices.  In an ideal (orthogonal) cube θ = 90°.  The *skewness*
    is the maximum absolute deviation of any face angle from 90°:

        skewness = max |θ − 90°|     (degrees)

    * 0.0 → perfect right‑angled element
    * Larger values indicate increasing distortion.
      A value above ~10–15° generally signals a low‑quality element.

    Parameters
    ----------
    hex_vert : (8, 3) array‑like
        Corner coordinates in the conventional FEM ordering
        (0 1 2 3 bottom face, 4 5 6 7 top face).

    Returns
    -------
    float
        Skewness in degrees (≥ 0).  Returns ``np.inf`` if any face edge
        collapses to zero length.
    """
    v = np.asarray(hex_vert, dtype=float).reshape(8, 3)

    # Faces defined by vertex indices (counter‑clockwise when viewed from outside)
    faces = [
        (0, 1, 2, 3),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # side
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7)
    ]

    max_dev = 0.0
    for f in faces:
        for k in range(4):
            a = v[f[k]] - v[f[(k - 1) % 4]]  # prev → current
            b = v[f[(k + 1) % 4]] - v[f[k]]  # current → next

            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0.0 or nb == 0.0:
                return float("inf")  # degenerate face

            cos_theta = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
            theta_deg = np.degrees(np.arccos(cos_theta))
            max_dev = max(max_dev, abs(theta_deg - 90.0))

    return max_dev
