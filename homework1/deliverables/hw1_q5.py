import matplotlib.pyplot as plt
import numpy as np
import random 

# false comes first because 0 = False 
# and 1 = True in python (duh)

a_cpt = 0.3
b_cpt = 0.6
c_cpt = np.array([0.5, 0.2])
d_cpt = np.array([0.4, 0.8])
e_cpt = np.array([0.1, 0.8])
f_cpt = 0.5
g_cpt = np.array(
    [[[0.8, 0.7], [0.6, 0.5]],
     [[0.4, 0.4], [0.2, 0.1]]]
)
h_cpt = np.array([0.4, 0.7])
i_cpt = np.array([[0.2, 0.4], [0.6, 0.8]])
j_cpt = np.array([[0.1, 0.9], [0.7, 0.2]])
k_cpt = np.array([0.3, 0.7])

# Problem 5: Find P(g | k, ~b, c)
# Output: "P(g | k, ~b, c) = x"

# Parents of g: C, D, E
k = 1
b = 0
c = 1

N = 500_000

lhd_normal = [0.0, 0.0]
lhd_probs = []

for _ in range(N):
    # sample the non-evidence values
    a = int(random.uniform(0,1) < a_cpt)
    # have B = ~b
    # have C = c
    d = int(random.uniform(0,1) < d_cpt[a])
    e = int(random.uniform(0,1) < e_cpt[b])
    f = int(random.uniform(0,1) < f_cpt)
    g = int(random.uniform(0,1) < g_cpt[c, d, e])
    # irrelevant: h = int(random.uniform(0,1) < h_cpt[e])
    i = int(random.uniform(0,1) < i_cpt[f, g])
    # irrelevant: j = int(random.uniform(0,1) < j_cpt[g, h])
    k = int(random.uniform(0,1) < k_cpt[i])

    weight = 1.0 
    weight *= k_cpt[i] if k else 1 - k_cpt[i]
    weight *= b_cpt if b else 1 - b_cpt 
    weight *= c_cpt[a] if c else 1 - c_cpt[a]

    # current sample is value in g
    lhd_normal[g] += weight
    lhd_probs.append((lhd_normal[1] / sum(lhd_normal)))

# fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# ax[0].plot(lhd_probs[20:], label='likelihood weighting')
# ax[0].legend()

# ax[1].plot(lhd_probs[20:10000], label='likelihood weighting')
# ax[1].legend()

# plt.show()

print(f'P(g | k, not b, c) = {round(lhd_probs[-1], 2)}')