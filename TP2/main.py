import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets as datasets

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import *


def matrix_Data_Print(x):
    print("Data : \n{}\nMean : {}\nVariance : {}".format(
        np.array_str(x), x.mean(), x.var()))


def main():
    X = np.array([[1, -1, 2],
                  [2, 0, 0],
                  [0, 1, -1]], np.float64)

    # A- Data normalization (using scale)
    matrix_Data_Print(X)
    X_n = scale(X)
    print("\tData Normalized")
    matrix_Data_Print(X_n)
    # B- Data normalization (using MinMaxScaler)
    mmscaler = MinMaxScaler((0, 1))
    X_n2 = mmscaler.fit_transform(X)
    print("\tData Normalized (minmax)")
    matrix_Data_Print(X_n2)

    # C- Data visualization
    iris = datasets.load_iris()
    data_size = iris.data.shape[0]

    # We'll preprocess data by creating dataframes
    species_cols = iris.target.reshape(data_size, 1)
    feature_name = [''.join(e.replace('(', '').replace(
        ')', '').title().split()) for e in iris.feature_names]

    # Importing iris and creating it's dataframe
    DataFrame = pd.DataFrame(data=np.hstack((iris.data, species_cols)),
                             columns=feature_name + ["Species"])

    print("Dataframe :\n{}".format(DataFrame.head()))

    # Ploting
    pl = sns.pairplot(DataFrame, hue="Species", diag_kind="kde")

    # D- Reduction de dimensions et visualisation de donnees

    # PCA Reduction

    pca = PCA(n_components=2)
    irisPCA = pca.fit_transform(DataFrame.drop("Species", axis=1))

    # LDA Reduction
    lda = LDA(n_components=2)
    irisLDA = lda.fit_transform(DataFrame.drop(
        "Species", axis=1), DataFrame["Species"])

    # Creating data frames
    PCA_DataFrame = pd.DataFrame(data=np.hstack((irisPCA, species_cols)),
                                 columns=["x", "y", "Species"])

    LDA_DataFrame = pd.DataFrame(data=np.hstack((irisLDA, species_cols)),
                                 columns=["x", "y", "Species"])
    
    sns.lmplot(x="x", y="y", hue="Species", fit_reg=False, data=PCA_DataFrame)
    sns.lmplot(x="x", y="y", hue="Species", fit_reg=False, data=LDA_DataFrame)
    # Showing all plots
    plt.show()


if __name__ == "__main__":
    main()
