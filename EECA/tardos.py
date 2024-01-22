import numpy as np
import random
import os

cwd = os.getcwd()

#'''

def equal_alpha_dirichlet_sampling(dim, alpha_0, v, num_samples):
    cnt = 0
    # samples = np.array([])
    samples = np.empty((0, 10))
    while cnt < num_samples:
        sample = np.random.dirichlet(np.full((dim,), alpha_0))
        if np.min(sample) > v:
            sample = sample.reshape(1, -1)
            samples = np.vstack([samples, sample])
            # samples = np.append(samples, sample)
            cnt = cnt + 1
            print(cnt)
    return samples

# Tardos-1
# C = 10, K = C, 5C, 10C
# k = 1/C = 0.1
# v = K^(-2/(1+1/C)) = 0.0152, 0.0008, 0.0002
# p_list = equal_alpha_dirichlet_sampling(10, 0.1, 0.0152, 400)

# Tardos-2
# C = 10, K = C, 5C, 10C
# k = 1/1.1C = 0.091
# v = K^(-4/3) = 0.0054, 0.0022 (K = 5C, 10C)
# v = K^(-1.8) = 0.0158 (K = C)
p_list = equal_alpha_dirichlet_sampling(10, 0.091, 0.0158, 400)

os.makedirs(os.path.join(cwd, 'Coding_Sheet'), exist_ok=True)
save_path = os.path.join(cwd, 'Coding_Sheet', 'C10_K10_Tardos-2.txt')
np.savetxt(save_path, p_list, fmt='%0.4f')

#'''

'''
load_path = os.path.join(cwd, 'Coding_Sheet', 'C10_K10_Tardos-1.txt')
p_list = np.loadtxt(load_path)
print(type(p_list))
print(p_list)
sample_list = list(range(0, 10))
for i in range(0, 5):
    print(p_list[i])
    sample_result = random.choices(sample_list, weights=p_list[i], k=1) # k: number of samples
    print(sample_result)
'''