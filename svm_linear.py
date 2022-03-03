from sklearn import svm
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
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
# Linear SVM
#

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
accuracies = []
maxAccuracy = -1

for c_value in C:
    # Create SVM Classifier with linear kernel
    linearSVM = svm.SVC(kernel='linear', C=c_value)
    # Train the model on 50% of inizial dataset
    linearSVM.fit(X_train, y_train)

    # Plot data and decision boundaries
    myplt.svm_plot_data_boudaries(X_train, y_train, linearSVM, c_value,
                                  folder='Linear_SVM_plots',
                                  fileName='LinearSVM classification')
    # Evaluate the model on the validation set
    accuracy = linearSVM.score(X_validation, y_validation)
    accuracies.append(accuracy)

    # Update the max accuracy
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy

    print("Accuracy = %.2f, obtained using C=%f" % (accuracy, c_value))


# Plot accuracy on C changes
myplt.svm_accuracy_plot(C, accuracies, folder='Linear_SVM_plots')


#Get the C with the best accuracy
best_C = C[accuracies.index(maxAccuracy)]
print("Best accuracy (%.2f) obtained using C=%f" % (maxAccuracy, best_C))

# Create SVM Classifier
linearSVM = svm.SVC(kernel='linear', C=c_value)
# Train the model on 70% of inizial dataset (train + validation)
linearSVM.fit(np.concatenate((X_train, X_validation)),
              np.concatenate((y_train, y_validation)))
# Evaluate the model on the test set
accuracy = linearSVM.score(X_test, y_test)
print("Accuracy on Test Set = %.2f, obtained using C=%f" % (accuracy, best_C))
