from sklearn import svm
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import myplotlib as myplt
from sklearn.decomposition import PCA

TRAIN_SIZE = 0.70
TEST_SIZE = 0.30

# Load the Wine dataset - X: features, y: labels
X, y = load_wine(True)

# Split the whole dataset into train set (70%) and test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=TEST_SIZE,
                                                    random_state=42,
                                                    shuffle=True)

# Standardize data using the StandardScaler()
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Select the first two principal components
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


#
# RBF SVM 5-Fold validation
#

#
# Grid search - tuning Gamma and C using 5-Fold validation
#
Gamma = [0.000000001, 0.0000001, 0.00001, 0.001, 0.1, 10, 100]
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
accuracies = []
maxAccuracy = -1
best_g = -1
best_C = -1

for c_value in C:
    c_accuracy = []
    for g_value in Gamma:
        # Create SVM Classifier
        rbfSVM = svm.SVC(kernel='rbf', gamma=g_value, C=c_value)
        # Evaluate the model using 5-Fold cross validation
        scores = cross_val_score(rbfSVM, X_train, y_train, cv=5)
        accuracy = scores.mean()
        c_accuracy.append(accuracy)

        # Update the max accuracy and best parameters
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            best_g = g_value
            best_C = c_value

    accuracies.append(c_accuracy)


# Plot accuracy on C and Gamma changes
myplt.svm_accuracy_grid(C, Gamma, accuracies, folder='Rbf_SVM_PCA_cv_plots')
print("Best accuracy (%.2f) obtained using gamma=%f, C=%f" % (maxAccuracy, best_g, best_C))

# Create SVM Classifier
rbfSVM = svm.SVC(kernel='rbf', gamma=best_g, C=best_C)
# Train the model on 70% of inizial dataset
rbfSVM.fit(X_train, y_train)
# Evaluate the model on the test set
accuracy = rbfSVM.score(X_test, y_test)
print("Accuracy on Test Set = %.2f, obtained using gamma=%f, C=%f" % (accuracy, best_g, best_C))

myplt.svm_plot_data_boudaries2(X_test, y_test,
                               rbfSVM, best_C, best_g,
                               folder='Rbf_SVM_PCA_cv_plots',
                               fileName='Rbf_SVM_cv classification gridCV')
