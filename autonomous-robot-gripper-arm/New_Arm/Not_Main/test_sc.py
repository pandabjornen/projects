import numpy as np

p = np.array([0.18623, -0.15884, 0.075])
RADIUS_OFFSET = -0.03

r = np.sqrt(p[0]**2 + p[1]**2)
theta = np.arctan2(p[1], p[0])

# Apply offset
r_new = r + RADIUS_OFFSET

# New positions for +θ and -θ
p_pos = np.array([r_new * np.cos(theta), r_new * np.sin(theta), p[2]])
p_neg = np.array([r_new * np.cos(-theta), r_new * np.sin(-theta), p[2]])

print("r =", round(r,5), "theta =", np.degrees(theta))
print("with +angle:", p_pos)
print("with -angle:", p_neg)
