import numpy as np
import random
import matplotlib.pyplot as plt
import scipy

# Set up probability tables

evidences = np.array([
    1, 1, 0, 0, 1, 1, 1, 1, 0, 1
])

p_r0 = np.array([1/3])

T_ord = np.array([
    [[0.05,2],[0.4,1],[1,0]],
    [[0.2,0],[0.4,2],[1,1]],
    [[0.0,0],[0.5,1],[1,2]]
])

T = np.array([
    [0.60, 0.35, 0.05],
    [0.20, 0.60, 0.20],
    [0.00, 0.50, 0.50]
])

u_cpt = np.array([
    [0.00, 1.00],
    [0.05, 0.95],
    [0.40, 0.60]
])

# Set up our evidence (in this case, we will sample for only the first timestep)
u_val = 1

def get_randoms(N):
    return np.random.uniform(low=0, high=1, size=N)


# Set the number of particles which we will sample with 
# Then draw that many samples from our prior distribution
N = 1_000_000

res = get_randoms(N)

v1 = np.where(res < (1/3), 1, 0)
v2 = np.where((res >= (1/3)) & (res < (2/3)), 2, 0)
v3 = np.where(res >= (2/3), 3, 0)

res = v1 + v2 + v3 
res = res.astype(int)

if (((res != 1) & (res != 2) & (res != 3)).sum() != 0):
    print("booooo")
    exit()

# s_low = (res < (1/3)).astype(int)
# s_med = ((res >= (1/3)) & (res < (2/3))).astype(int)
# s_hih = (res >= (2/3)).astype(int)

# # Number of samples where it rains on day 0
# hih_p = np.sum(s_hih)
# med_p = np.sum(s_med)
# low_p = np.sum(s_low)

# plt.bar(['low day 0', 'med day 0', 'high day 0'], [low_p, med_p, hih_p])
# plt.show()

############################################################################################

# get labels 

for i in range(10):
    # Sample the values at the next time step according to our state transition matrix T
    for j in range(N):
        rand = random.uniform(0,1)
        prev_label = res[j] - 1
        row = T_ord[prev_label]
        if (rand < row[0, 0] and rand >= 0):
            res[j] = row[0, 1]+1
            continue
        
        if (rand < row[1, 0] and rand >= row[0, 0]):
            res[j] = row[1, 1]+1
            continue

        if (rand < row[2, 0] and rand >= row[1, 0]):
            res[j] = row[2, 1]+1
            continue

    # Weight each of the samples according to the evidence we have at this time step
    sample_weights = np.zeros(N)
    for j in range(N):
        cpt_row = u_cpt[res[j]-1]
        sample_weights[j] = cpt_row[evidences[i]]
    
    # We normalize samp_weight_1 so that it sums to 0 
    # We want to get the probability of selecting each of the samples from samp_vals_1
    sample_weights /= np.sum(sample_weights)

    # Now we resample from samp_vals_1
    res = np.random.choice(res, N, p=sample_weights)


c1 = np.sum(np.where(res == 1, 1, 0))
c2 = np.sum(np.where(res == 2, 1, 0))
c3 = np.sum(np.where(res == 3, 1, 0))

plt.bar(['low day 1', 'med day 1', 'high day 1'], [c1, c2, c3])
plt.show()

print("P(X10 = Low  | u_1..10) = {:.3f}".format(c1 / N))
print("P(X10 = Med  | u_1..10) = {:.3f}".format(c2 / N))
print("P(X10 = High | u_1..10) = {:.3f}".format(c3 / N))