def decimal_range(start, stop, increment):
    while start < stop:
        yield start
        start += increment

def p6(sigma_t, sigma_x, sigma_e, depth):
    if depth == 0: return sigma_t

    sigma_tplus1 = sigma_t + sigma_x 
    sigma_tplus1 *= sigma_x 
    sigma_tplus1 /= (sigma_t + sigma_x + sigma_e)

    return p6(sigma_tplus1, sigma_x, sigma_e, depth-1)

vals = []
for i in decimal_range(0.0, 1.0, 0.01):
    vals.append((p6(sigma_t=1, sigma_x=i, sigma_e=0.75, depth=10), i))

apoint65 = [i for val,i in vals if val < 0.65]

print(apoint65[-1])
print(f"From Loop: {p6(1, 0.95, 0.75, 10)}")
print(f".01 Increment To Prove: {p6(1, 0.96, 0.75, 10)}")