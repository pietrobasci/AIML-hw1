from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
# K-Nearest Neighbors
#

K = [1, 3, 5, 7]
accuracies = []
maxAccuracy = -1

for k_value in K:
    # Create K-Nearest Neighbors Classifier
    neigh = KNeighborsClassifier(n_neighbors=k_value)
    # Train the model on 50% of inizial dataset
    neigh.fit(X_train, y_train)

    # Plot data and decision boundaries
    myplt.knn_plot_data_boudaries(X_train, y_train, neigh, k_value,
                                  folder='K-NN_plots')
    # Evaluate the model on the validation set
    accuracy = neigh.score(X_validation, y_validation)
    accuracies.append(accuracy)

    # Update the max accuracy
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy

    print("Accuracy = %.2f, obtained using K=%i" % (accuracy, k_value))


# Plot accuracy on K changes
myplt.knn_accuracy_plot(K, accuracies, folder='K-NN_plots')


#Get the K with the best accuracy
best_K = K[accuracies.index(maxAccuracy)]
print("Best accuracy (%.2f) obtained using K=%i" %(maxAccuracy, best_K))

# Create K-Nearest Neighbors Classifier
neigh = KNeighborsClassifier(n_neighbors=best_K)
# Train the model on 70% of inizial dataset (train + validation)
neigh.fit(np.concatenate((X_train, X_validation)),
          np.concatenate((y_train, y_validation)))
# Evaluate the model on the test set
accuracy = neigh.score(X_test, y_test)
print("Accuracy on Test Set = %.2f, obtained using K=%i" %(accuracy, best_K))
