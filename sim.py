import numpy as np


def euclid(a, b):
    assert len(a) == len(b)
    s = 0
    for i in range(len(a)):
        s += (a[i] - b[i]) ** 2
    return np.sqrt(s)


def smc(a, b):
    assert len(a) == len(b)
    f00 = 0
    f01 = 0
    f10 = 0
    f11 = 0
    for i in range(len(a)):
        if a[i] == 0 and b[i] == 0:
            f00 += 1
        if a[i] == 1 and b[i] == 0:
            f10 += 1
        if a[i] == 0 and b[i] == 1:
            f01 += 1
        if a[i] == 1 and b[i] == 1:
            f11 += 1
    return (f00 + f11) / (f00 + f01 + f10 + f11)


def cos(a, b):
    dotproduct = np.dot(a, b)
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    return dotproduct / (da * db)


def pearson(a, b):
    assert(len(a) == len(b))
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    cor = 0
    var_a = 0
    var_b = 0
    for i in range(len(a)):
        cor += (a[i] - avg_a) * (b[i] - avg_b)
        var_a += (a[i] - avg_a)**2
        var_b += (b[i] - avg_b)**2
    return cor / (var_a * var_b)

a = np.array([1, 2, 3])
b = np.array([1, 2, 4])
print('Euclid distance: %f' % (euclid(a, b)))

a = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1]
b = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
print('SMC coefficient: %f' % (smc(a, b)))
print('COS coefficient: %f' % (cos(a, b)))
print('pearson coefficient: %f' % (pearson(a, b)))
