import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


def knn_plot_data_boudaries(X, y, model, K, folder):
    h = .01  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1],
                c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)" % K)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder + '/K-NN classification (k = %i)' % K)
    #plt.show()

def knn_accuracy_plot(k, accuracies, folder):
    plt.figure(figsize=(12,5))
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(k, accuracies, marker='o')
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder + '/accuracy.png')
    #plt.show()

def svm_plot_data_boudaries(X, y, model, C, folder, fileName):
    h = .01  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1],
                c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (c = %.3f)" % C)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder + '/' + fileName + '(C = %.0E)' % C)
    #plt.show()

def svm_accuracy_plot(c, accuracies, folder):
    plt.figure(figsize=(12,5))
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xscale("log")
    plt.plot(c, accuracies, marker='o')
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder + '/accuracy.png')
    #plt.show()

def svm_plot_data_boudaries2(X, y, model, C, gamma, folder, fileName):
    h = .01  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1],
                c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (c = %.3f, gamma = %f)" % (C, gamma))
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder + '/' + fileName + ' (C = %.0E, gamma = %.0E)' % (C, gamma))
    #plt.show()

def svm_accuracy_grid(c, gamma, accuracies, folder):
    plt.figure()
    plt.xscale('linear')
    axes = sns.heatmap(accuracies,
                       cmap='Oranges', linewidth=1,
                       xticklabels=gamma, yticklabels=c,
                       annot=True, vmin=0, vmax=1,
                       square=True)
    axes.invert_yaxis()
    axes.set_xlabel('gamma')
    axes.set_ylabel('C')
    axes.set_title("Accuracy")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder + '/c_gamma_grid.png')
    #plt.show()
