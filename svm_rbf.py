from sklearn import svm
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import myplotlib as myplt

TRAIN_SIZE = 0.70
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.20

# Load the Wine dataset - X: features, y: labels
X, y = load_wine(True)
# Select the first two features
X = X[:, 0:2]

# Split the whole dataset into train set (70%) and test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=TEST_SIZE,
                                                    random_state=42,
                                                    shuffle=True)

# Standardize data using the StandardScaler()
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Split the train set (70%) into new train set (50%) and validation set (20%)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                test_size=VALIDATION_SIZE/TRAIN_SIZE,
                                                                random_state=42,
                                                                shuffle=True)


#
# RBF SVM
#

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
accuracies = []
maxAccuracy = -1

for c_value in C:
    # Create SVM Classifier with RBF kernel
    rbfSVM = svm.SVC(kernel='rbf', gamma='auto', C=c_value)
    # Train the model on 50% of inizial dataset
    rbfSVM.fit(X_train, y_train)

    # Plot data and decision boundaries
    myplt.svm_plot_data_boudaries(X_train, y_train, rbfSVM, c_value,
                                  folder='Rbf_SVM_plots',
                                  fileName='RbfSVM classification')
    # Evaluate the model on the validation set
    accuracy = rbfSVM.score(X_validation, y_validation)
    accuracies.append(accuracy)

    # Update the max accuracy
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy

    print("Accuracy = %.2f, obtained using C=%f" % (accuracy, c_value))


# Plot accuracy on C changes
myplt.svm_accuracy_plot(C, accuracies, folder='Rbf_SVM_plots')


#Get the C with the best accuracy
best_C = C[accuracies.index(maxAccuracy)]
print("Best accuracy (%.2f) obtained using C=%f" % (maxAccuracy, best_C))

# Create SVM Classifier
rbfSVM = svm.SVC(kernel='rbf', gamma='auto', C=best_C)
# Train the model on 70% of inizial dataset (train + validation)
rbfSVM.fit(np.concatenate((X_train, X_validation)),
           np.concatenate((y_train, y_validation)))
# Evaluate the model on the test set
accuracy = rbfSVM.score(X_test, y_test)
print("Accuracy on Test Set = %.2f, obtained using C=%f" % (accuracy, best_C))


#
# Grid search - tuning gamma and C
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
        # Create SVM Classifier with RBF kernel
        rbfSVM = svm.SVC(kernel='rbf', gamma=g_value, C=c_value)
        # Train the model on 50% of inizial dataset
        rbfSVM.fit(X_train, y_train)
        # Evaluate the model on the validation set
        accuracy = rbfSVM.score(X_validation, y_validation)
        c_accuracy.append(accuracy)

        # Update the max accuracy and best parameters
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            best_g = g_value
            best_C = c_value

    accuracies.append(c_accuracy)


# Plot accuracy on C and Gamma changes
myplt.svm_accuracy_grid(C, Gamma, accuracies, folder='Rbf_SVM_plots')
print("Best accuracy (%.2f) obtained using gamma=%f, C=%f" % (maxAccuracy, best_g, best_C))

# Create SVM Classifier
rbfSVM = svm.SVC(kernel='rbf', gamma=best_g, C=best_C)
# Train the model on 70% of inizial dataset (train + validation)
rbfSVM.fit(np.concatenate((X_train, X_validation)),
           np.concatenate((y_train, y_validation)))
# Evaluate the model on the test set
accuracy = rbfSVM.score(X_test, y_test)
print("Accuracy on Test Set = %.2f, obtained using gamma=%f, C=%f" % (accuracy, best_g, best_C))

myplt.svm_plot_data_boudaries2(X_test, y_test,
                               rbfSVM, best_C, best_g,
                               folder='Rbf_SVM_plots',
                               fileName='Rbf_SVM classification grid')

#
# RBF SVM 5-Fold validation
#

# Merge the training and validation splits
X_train = np.concatenate((X_train, X_validation))
y_train = np.concatenate((y_train, y_validation))

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
myplt.svm_accuracy_grid(C, Gamma, accuracies, folder='Rbf_SVM_plots/Rbf_SVM_cv_plots')
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
                               folder='Rbf_SVM_plots/Rbf_SVM_cv_plots',
                               fileName='Rbf_SVM_cv classification gridCV')
