import numpy as np


coeff_sets = [0, 1000, 2000, 6000]
model_terms_count = 15

result = np.zeros((1,model_terms_count))


coeffs_all = np.zeros((1,model_terms_count), dtype=int)
for rep in range(0, 1000):
    sample = np.random.randint(len(coeff_sets), size=model_terms_count)
    i = 0
    while i < len(coeffs_all):
        if (coeffs_all[i] != sample).any():
            i = i+1
        else:
            sample = np.random.randint(len(coeff_sets), size=model_terms_count)
            i = 0
            continue
    coeffs_all = np.append(coeffs_all, sample.reshape(1,-1), axis=0)


np.save('random_generated_coeffs.npy', coeffs_all[1:])
print(coeffs_all[1:])
