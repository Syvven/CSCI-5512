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

# Problem 5: Find P(g | k, b, ~c)
# Output: "P(g | k, b, ~c) = x"

# Parents of g: C, D, E
k = 1
b = 1
c = 0

N = 500_000

probs = []
g_val = 0

# non-evidence = A, D, E, F, G, H, I, J

cur_a = 1
cur_d = 1
cur_e = 1
cur_f = 1
cur_g = 1
cur_h = 1
cur_i = 1
cur_j = 1

states = [[
    cur_a, cur_d, cur_e, cur_f,
    cur_g, cur_h, cur_i, cur_j
]]

count = 0
for _ in range(int(N/2)):
    # a: P(A) * P(C|A) * P(D|A)
    p_a_mb = np.zeros(2)
    for a in [0,1]:
        p_a = a_cpt if a else 1 - a_cpt 
        p_c = c_cpt[a] if c else 1 - c_cpt[a]
        p_d = d_cpt[a] if cur_d else 1 - c_cpt[a]

        p_a_mb[a] = p_a * p_c * p_d
    p_a_mb /= np.sum(p_a_mb)

    cur_a = int(random.uniform(0,1) < p_a_mb[1])
    # states.append([
    #     cur_a, cur_d, cur_e, cur_f,
    #     cur_g, cur_h, cur_i, cur_j
    # ])

    g_val += cur_g
    probs.append(g_val / (count + 1))
    count += 1

    # d: P(D|A) * P(G|D)
    p_d_mb = np.zeros(2)
    for d in [0,1]:
        p_d = d_cpt[cur_a] if d else 1 - d_cpt[cur_a]
        p_g = g_cpt[c, d, cur_e] if d else 1 - g_cpt[c, d, cur_e]
        p_d_mb[d] = p_d * p_g 
    p_d_mb /= np.sum(p_d_mb)

    cur_d = int(random.uniform(0,1) < p_d_mb[1])
    # states.append([
    #     cur_a, cur_d, cur_e, cur_f,
    #     cur_g, cur_h, cur_i, cur_j
    # ])

    g_val += cur_g
    probs.append(g_val / (count + 1))
    count += 1

    # e: P(E|B) * P(H|E) * P(G|E,D,C)
    p_e_mb = np.zeros(2)
    for e in [0,1]:
        p_e = e_cpt[b] if e else 1 - e_cpt[b]
        p_h = h_cpt[e] if cur_h else 1 - h_cpt[e]
        p_g = g_cpt[c, cur_d, e] if cur_g else 1 - g_cpt[c, cur_d, e]
        p_e_mb[e] = p_e * p_h * p_g 
    p_e_mb /= np.sum(p_e_mb)

    cur_e = int(random.uniform(0,1) < p_e_mb[1])
    # states.append([
    #     cur_a, cur_d, cur_e, cur_f,
    #     cur_g, cur_h, cur_i, cur_j
    # ])

    g_val += cur_g
    probs.append(g_val / (count + 1))
    count += 1

    # g: P(G|C,D,E) * P(I|F,G) * P(J|G,H)
    p_g_mb = np.zeros(2)
    for g in [0,1]:
        p_g = g_cpt[c, cur_d, cur_e] if g else 1 - g_cpt[c, cur_d, cur_e]
        p_i = i_cpt[cur_f, g] if cur_i else 1 - i_cpt[cur_f, g]
        p_j = j_cpt[g, cur_h] if cur_j else 1 - j_cpt[g, cur_h]
        p_g_mb[g] = p_g * p_i * p_j
    p_g_mb /= np.sum(p_g_mb)

    cur_g = int(random.uniform(0,1) < p_g_mb[1])
    # states.append([
    #     cur_a, cur_d, cur_e, cur_f,
    #     cur_g, cur_h, cur_i, cur_j
    # ]*4)

    g_val += cur_g
    probs.append(g_val / (count + 1))
    count += 1

    # cur_g won't change from now on so it doesnt matter

    # I
    g_val += cur_g
    probs.append(g_val / (count + 1))
    count += 1

    # F
    g_val += cur_g
    probs.append(g_val / (count + 1))
    count += 1

    # H
    g_val += cur_g
    probs.append(g_val / (count + 1))
    count += 1

    # J
    g_val += cur_g
    probs.append(g_val / (count + 1))
    count += 1

# fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# ax[0].plot(probs[20:], label='gibbs sampling precompute')
# ax[0].legend()

# ax[1].plot(probs[20:10000], label='gibbs sampling precompute')
# ax[1].legend()

# plt.show()

print(f"P(g | k, b, not c) = {round(probs[-1],2)}")