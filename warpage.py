import numpy as np


def Warpage(hex_vert):
    """
    Face‑planarity (warpage) metric for an eight‑node hexahedron.

    Definition
    ----------
    For each of the six quadrilateral faces, split the face into two
    triangles using the diagonal (v0, v2).  Let n₁ and n₂ be the unit
    normals of those triangles.  The *warpage angle* ω of the face is

        ω = arccos( n₁ · n₂ )   ∈ [0°, 180°].

    Face ω = 0°  → perfectly planar face.
    The hexahedron *warpage* is the maximum ω over all faces.

    * 0.0 → ideal, all faces planar.
    * Larger values indicate increasing non‑planarity.
      Many meshing guidelines regard ω ≲ 5–10° as acceptable.

    Parameters
    ----------
    hex_vert : (8, 3) array‑like
        Corner coordinates in the standard FEM ordering
        (0‑1‑2‑3 bottom, 4‑5‑6‑7 top).

    Returns
    -------
    float
        Maximum face warpage angle in **degrees** (≥ 0).
        Returns ``np.inf`` if any face degenerates (zero‑area triangle).
    """
    v = np.asarray(hex_vert, dtype=float).reshape(8, 3)

    # Faces (counter‑clockwise when viewed from outside)
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
