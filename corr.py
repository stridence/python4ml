import numpy as np

a = np.random.normal(0, 1, size=[200, 150])
am = a - np.mean(a, axis=0)
bm = am.T
Sig = np.dot(bm, am) / (am.shape[0] - 1)
[U, V] = np.linalg.eig(Sig)
ind = np.argsort(-U)
eigenval = U[ind]
eigenvec = V[:, ind]

s = 0
total = np.sum(U)
thresh = 0.95
for k in range(len(U)):
    s += eigenval[k]
    if s / total > thresh:
        break

Vk = eigenvec[:, 0:k]

ar = np.dot(Vk, np.dot(Vk.T, a.T))
diff = np.linalg.norm(a - ar.T)
print(k, diff)
