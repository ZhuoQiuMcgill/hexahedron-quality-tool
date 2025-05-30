from jacobian import jacobian, scaled_jacobian

# Reference unit cube
ref = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
]

# Perfect 2× scaled cube (right‑handed)
phys = [
    [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
    [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]
]

print(jacobian(ref, phys))  # 8.0
print(scaled_jacobian(ref, phys))  # 1.0  (ideal)
