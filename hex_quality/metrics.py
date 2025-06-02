"""
Core quality metric implementations for hexahedral finite elements.

This module contains all the quality assessment functions for 8-node hexahedral elements.
All functions assume standard FEM vertex ordering and return appropriate quality measures.
"""

import numpy as np


def aspect_ratio(hex_vert):
    """
    Compute the aspect ratio of a linear, 8-node hexahedron.

    The aspect ratio is defined as the ratio of the longest edge to the shortest edge.
    An ideal cube has an aspect ratio of 1.0, while stretched elements have higher values.

    Parameters
    ----------
    hex_vert : array_like, shape (8, 3)
        Corner-node coordinates in standard FEM ordering:
        (0-1-2-3 bottom face counter-clockwise, then 4-5-6-7 top face, and
        vertical edges 0-4, 1-5, 2-6, 3-7).

    Returns
    -------
    float
        Aspect ratio >= 1.0. Returns np.inf if any edge collapses to zero length.

    Notes
    -----
    - 1.0: ideal "cube-like" element with all edges equal
    - >1: stretched element; larger values indicate poorer quality
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


def warpage(hex_vert):
    """
    Compute face-planarity (warpage) metric for an eight-node hexahedron.

    For each quadrilateral face, the warpage angle is computed by splitting the face
    into two triangles and measuring the angle between their normal vectors.
    The element warpage is the maximum warpage angle over all six faces.

    Parameters
    ----------
    hex_vert : array_like, shape (8, 3)
        Corner coordinates in standard FEM ordering (0-1-2-3 bottom, 4-5-6-7 top).

    Returns
    -------
    float
        Maximum face warpage angle in degrees (>= 0).
        Returns np.inf if any face degenerates (zero-area triangle).

    Notes
    -----
    - 0.0: ideal, all faces perfectly planar
    - Larger values indicate increasing non-planarity
    - Many meshing guidelines consider warpage <= 5-10° as acceptable
    """
    v = np.asarray(hex_vert, dtype=float).reshape(8, 3)

    # Faces (counter-clockwise when viewed from outside)
    faces = [
        (0, 1, 2, 3),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # sides
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7)
    ]

    max_warp = 0.0
    for f in faces:
        # triangles (v0, v1, v2) and (v0, v2, v3)
        tri1 = v[f[1]] - v[f[0]], v[f[2]] - v[f[0]]
        tri2 = v[f[2]] - v[f[0]], v[f[3]] - v[f[0]]

        n1 = np.cross(*tri1)
        n2 = np.cross(*tri2)

        a1 = np.linalg.norm(n1)
        a2 = np.linalg.norm(n2)
        if a1 == 0.0 or a2 == 0.0:
            return float("inf")  # degenerate triangle → undefined normal

        n1 /= a1
        n2 /= a2

        cos_ang = np.clip(np.dot(n1, n2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_ang))

        max_warp = max(max_warp, angle_deg)

    return max_warp


def skewness(hex_vert):
    """
    Compute face-angle skewness of a linear, eight-node hexahedron.

    The skewness measures the maximum deviation of any face angle from 90 degrees.
    In an ideal orthogonal cube, all face angles are exactly 90 degrees.

    Parameters
    ----------
    hex_vert : array_like, shape (8, 3)
        Corner coordinates in conventional FEM ordering
        (0-1-2-3 bottom face, 4-5-6-7 top face).

    Returns
    -------
    float
        Skewness in degrees (>= 0). Returns np.inf if any face edge
        collapses to zero length.

    Notes
    -----
    - 0.0: perfect right-angled element
    - Larger values indicate increasing distortion
    - Values above ~10-15° generally signal low-quality elements
    """
    v = np.asarray(hex_vert, dtype=float).reshape(8, 3)

    # Faces defined by vertex indices (counter-clockwise when viewed from outside)
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


def orthogonal_quality(hex_vert):
    """
    Compute orthogonal quality metric for an eight-node hexahedron.

    At each vertex, a perfect cube has three mutually perpendicular edge directions.
    This metric measures how close the element comes to this ideal by examining
    the angles between incident edges at each vertex.

    Parameters
    ----------
    hex_vert : array_like, shape (8, 3)
        Coordinates of the hexahedron's eight corner nodes in standard
        FEM ordering (0-1-2-3 bottom face CCW, 4-5-6-7 top face).

    Returns
    -------
    float
        Orthogonal quality in the range [0, 1].
        Returns 0.0 if any incident edge has zero length.

    Notes
    -----
    - 1.0: perfectly orthogonal (ideal cube)
    - 0.0: degenerate (collinear edges) or highly skewed
    - Values closer to 1 indicate better quality
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
        # three outgoing, non-repeated edges from vertex i
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


def jacobian(ref_hex, phys_hex):
    """
    Compute determinant of the affine Jacobian mapping ref_hex to phys_hex.

    This function computes the determinant of the linear transformation part
    of the affine mapping that transforms the reference hexahedron to the
    physical hexahedron.

    Parameters
    ----------
    ref_hex : array_like, shape (8, 3)
        Reference hexahedron vertex coordinates.
    phys_hex : array_like, shape (8, 3)
        Physical hexahedron vertex coordinates.

    Returns
    -------
    float
        Determinant of the Jacobian matrix. Positive values indicate
        proper orientation, negative values indicate inverted elements.
    """
    ref = np.asarray(ref_hex, dtype=float).reshape(8, 3)
    phys = np.asarray(phys_hex, dtype=float).reshape(8, 3)

    # Design matrix of the reference coordinates
    M = np.hstack((ref, np.ones((8, 1))))

    # Least-squares solve  M · P ≈ phys , where P = [A^T | b] (4×3)
    P, *_ = np.linalg.lstsq(M, phys, rcond=None)
    A = P[:3, :].T  # 3×3 linear part

    return float(np.linalg.det(A))


def scaled_jacobian(ref_hex, phys_hex):
    """
    Compute a scaled Jacobian quality metric in the range [0, 1].

    The scaled Jacobian normalizes the raw Jacobian determinant by the geometric
    upper bound given by Hadamard's inequality, then maps the result to [0, 1].

    Parameters
    ----------
    ref_hex : array_like, shape (8, 3)
        Reference hexahedron vertex coordinates (matching order with phys_hex).
    phys_hex : array_like, shape (8, 3)
        Physical hexahedron vertex coordinates.

    Returns
    -------
    float
        Scaled jacobian quality in [0, 1].

    Notes
    -----
    - 1.0: perfect, right-handed element
    - 0.0: degenerate or inverted element
    - 0.5: determinant is half of its geometric upper bound

    Raises
    ------
    ValueError
        If input arrays are not shaped (8, 3).
    """
    ref = np.asarray(ref_hex, dtype=float).reshape(8, 3)
    phys = np.asarray(phys_hex, dtype=float).reshape(8, 3)

    if ref.shape != (8, 3) or phys.shape != (8, 3):
        raise ValueError("Each hexahedron must be an (8, 3) array-like.")

    # --- Build affine map ---------------------------------------------------
    M = np.hstack((ref, np.ones((8, 1))))
    P, *_ = np.linalg.lstsq(M, phys, rcond=None)
    A = P[:3, :].T  # 3×3

    # --- Scaled Jacobian -----------------------------------------------------
    detA = np.linalg.det(A)
    col_norms = np.linalg.norm(A, axis=0)  # ||a₁||, ||a₂||, ||a₃||
    denom = np.prod(col_norms)

    # Handle degenerate case gracefully
    if denom == 0.0:
        return 0.0

    SJ = detA / denom  # ∈ [-1, 1]
    scaled = 0.5 * (SJ + 1.0)  # ∈ [0, 1]

    # Numerical safety clamp
    return float(np.clip(scaled, 0.0, 1.0))
