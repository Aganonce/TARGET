import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from target import TARGET

def plot_pca(X):
    pca = PCA(n_components=2)
    X_r1 = pca.fit_transform(X)

    fig, ax = plt.subplots()
    for i in range(len(X_r1)):
        plt.scatter(X_r1[i][0], X_r1[i][1], c='r')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('PCA on TARGET')
    plt.show()

if __name__ == '__main__':
    infile = 'data/small_sample.csv'
    target = TARGET(verbose=True).train_csv(infile, save=True)

    for c in target.resf_:
        X = StandardScaler().fit_transform(target.resf_[c])
        plot_pca(X)