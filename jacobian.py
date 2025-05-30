import numpy as np


def jacobian(ref_hex, phys_hex):
    """
    Determinant of the affine Jacobian that maps `ref_hex` → `phys_hex`.
    See previous message for full doc‑string.
    """
    ref = np.asarray(ref_hex, dtype=float).reshape(8, 3)
    phys = np.asarray(phys_hex, dtype=float).reshape(8, 3)

    # Design matrix of the reference coordinates
    M = np.hstack((ref, np.ones((8, 1))))

    # Least‑squares solve  M · P ≈ phys , where P = [Aᵀ | b] (4×3)
    P, *_ = np.linalg.lstsq(M, phys, rcond=None)
    A = P[:3, :].T  # 3×3 linear part

    return float(np.linalg.det(A))


def scaled_jacobian(ref_hex, phys_hex):
    """
    Compute a *scaled Jacobian* quality metric in the range [0, 1].

    Definition
    ----------
    Let A be the 3 × 3 linear part of the affine mapping that sends the
    reference hexahedron to the physical one.

        • det A  measures the signed volume change.
        • ‖a₁‖‖a₂‖‖a₃‖  (product of the norms of A’s column vectors) is an
          upper bound on |det A| by Hadamard’s inequality.

    The classical *scaled Jacobian* is
        SJ = det A / (‖a₁‖‖a₂‖‖a₃‖) ∈ [−1, 1].

    We map this to [0, 1] by
        scaled = 0.5 × (SJ + 1).

    * 1.0  → perfect, right‑handed element
    * 0.0  → degenerate or inverted element
    * 0.5  → determinant is half of its geometric upper bound

    Parameters
    ----------
    ref_hex, phys_hex : (8, 3) array‑like
        Eight vertex coordinates (matching order) for the reference and
        physical hexahedra.

    Returns
    -------
    float
        A value in [0, 1] measuring element quality.
    """
    ref = np.asarray(ref_hex, dtype=float).reshape(8, 3)
    phys = np.asarray(phys_hex, dtype=float).reshape(8, 3)

    if ref.shape != (8, 3) or phys.shape != (8, 3):
        raise ValueError("Each hexahedron must be an (8, 3) array‑like.")

    # --- Build affine map ---------------------------------------------------
    M = np.hstack((ref, np.ones((8, 1))))
    P, *_ = np.linalg.lstsq(M, phys, rcond=None)
    A = P[:3, :].T  # 3×3

    # --- Scaled Jacobian -----------------------------------------------------
    detA = np.linalg.det(A)
    col_norms = np.linalg.norm(A, axis=0)  # ‖a₁‖, ‖a₂‖, ‖a₃‖
    denom = np.prod(col_norms)

    # Handle degenerate case gracefully
    if denom == 0.0:
        return 0.0

    SJ = detA / denom  # ∈ [−1, 1]
    scaled = 0.5 * (SJ + 1.0)  # ∈ [0, 1]

    # Numerical safety clamp
    return float(np.clip(scaled, 0.0, 1.0))
