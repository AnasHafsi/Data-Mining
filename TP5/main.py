
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets


def f(x):
    return (x-1)*(x-2)*(x-3)*(x-5)


def df(x):
    return (4*(x**3))-(33*(x**2))+(82*x)-61


def g(x, y, a, b):
    return (a*x+b-y)**2


def dg(x, y, a, b):
    return {'dx': 2*(a*x**2+b*x-y*x), 'dy': 2*(a*x+b-y)}


def gradient(max_iter, epsi, Etape, x0, df):
    x = x0
    for i in range(max_iter):
        x_prev = x
        x = x_prev - (Etape*df(x_prev))
        if abs(x-x_prev) < epsi:
            return [x, i]
    return [x, max_iter]


def gradient_reg(max_iter, epsi, Etape, x, y, df):
    nb_sample_data = x.shape[0]
    rg_sample = range(1, nb_sample_data)
    a = np.random.random(x.shape[1])
    b = np.random.random(x.shape[1])
    E = sum([a*x[j]+b-y[j] for j in rg_sample])
    print(E)
    for i in range(max_iter):
        grad_a = sum([2*(a*x[j]**2+b*x[j]-y[j]*x[j]) for j in rg_sample])
        grad_b = sum([2*(a*x[j]+b-y[j]) for j in rg_sample])
        tmp_a, tmp_b = (a - Etape * grad_a), (b - Etape * grad_b)
        a, b = tmp_a, tmp_b
        e = sum([a*x[j]+b-y[j] for j in rg_sample])
        if abs(E-e) <= epsi:
            return ({'a': a, 'b': b}, i)
        E = e

    return ({'a': a, 'b': b}, max_iter)


def main():
    # A- Descente de gradient.
    epsi = 0.0001
    maxi = 1000
    args = [{'Etape': 0.001, 'x0': 5},
            {'Etape': 0.01, 'x0': 5},
            {'Etape': 0.1, 'x0': 5},
            {'Etape': 0.17, 'x0': 5},
            {'Etape': 0.1, 'x0': 5},
            {'Etape': 0.001, 'x0': 0}]
    store_grad = []
    for i in range(len(args)):
        gd = gradient(max_iter=maxi, epsi=epsi, df=df, **args[i])
        store_grad.append(gd)
        print("[x0:{}\tEtape:{}] - {} - E(xmini) = {}".format(args[i]
                                                              ['x0'], args[i]['Etape'], gd, f(gd[0])))
    plt.show()


if __name__ == "__main__":
    main()
