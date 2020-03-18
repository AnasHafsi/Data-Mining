import matplotlib.pyplot as plt
import numpy as np
from sklearn import *
from sklearn.datasets import fetch_openml

def main():
    iris = datasets.load_iris()
    print("Data : {}".format(np.array_str(iris.data)))
    print("Class Name : {}".format(iris.target_names))
    print("Features Name : {}".format(iris.feature_names))

    # Question B - 3
    for i in range(len(iris.data)):
        print("Data : {}, Class : {}".format(
            iris.data[i], iris.target_names[iris.target[i]].capitalize()))

    print("Data size : {}".format(iris.data.shape[0]))
    print("Features size : {}".format(iris.data.shape[1]))
    print("Class Size : {}".format(np.unique(iris.target).size))
    print("Mean : {}".format(iris.data.mean(0)))
    print("Min : {}".format(iris.data.min(0)))
    print("Max : {}".format(iris.data.max(0)))
    print("Std Deviation : {}".format(iris.data.std(0)))

    print("\n===============================================================\n")

    # Question C - 1
    print("Loading mnist ...")
    mnist = fetch_openml('mnist_784', version=1)

    # Question C - 2
    print("Data : {}".format(np.array_str(mnist.data)))
    print("Data size : {}".format(mnist.data.shape[0]))
    print("Features size : {}".format(mnist.data.shape[1]))
    print("Class Size : {}".format(np.unique(mnist.target).size))
    print("Mean : {}".format(mnist.data.mean(0)))
    print("Min : {}".format(mnist.data.min(0)))
    print("Max : {}".format(mnist.data.max(0)))
    print("Std Deviation : {}".format(mnist.data.std(0)))

    # Question D - 3
    blob = datasets.make_blobs(n_samples=1000, n_features=2, centers=4)
    blob_data = blob[0]
    blob_target = blob[1]

    # Question D - 4
    blob_100 = datasets.make_blobs(n_samples=100, n_features=2, centers=2)
    blob_500 = datasets.make_blobs(n_samples=500, n_features=2, centers=3)
    blob_data_600 = np.vstack((blob_100[0], blob_500[0]))
    blob_target_600 = np.hstack((blob_100[1], blob_500[1]))

    # Setting Plot using plt
    plot_style = [{'c': 'green', 'marker': 'o'},
                {'c': 'red',   'marker': 'o'},
                {'c': 'blue',  'marker': 'o'},
                {'c': 'cyan',  'marker': 'o'}]

    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

    # Plot 1
    plt.subplot(121)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel("Features")
    plt.ylabel("Value")
    plt.title("Jeux de 1000 donnees")
    for i in range(1000):
        plt.scatter(blob_data[i, 0], blob_data[i, 1], **plot_style[blob_target[i]])


    # Plot 2
    plt.subplot(122)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel("Features")
    plt.ylabel("Value")
    plt.title("Deux jeux de donnees concatenee")
    for i in range(600):
        plt.scatter(blob_data_600[i, 0], blob_data_600[i,
                                                    1], **plot_style[blob_target_600[i]])

    plt.show()


if __name__ == "__main__":
    main()