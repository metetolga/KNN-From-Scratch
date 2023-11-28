import numpy as np 
import pandas as pd

from collections import Counter
from scipy.sparse import csr_matrix

# Define a function to calculate the Euclidean distance between two row values.
def euclidean(row1, row2):
    # One of the fastest way to get the Euclidean distance
    distance = np.linalg.norm(row1 - row2) 
    return distance

# Define a class for a k-Nearest Neighbors classifier
class KNeighborClassifier():
    def __init__(self, n_neighbors = 5, weighted = False) -> None:
        # Initialize the classifier with the number of neighbors and a flag for weighted classification
        self.n_neighbors = n_neighbors
        self.weighted = weighted

        # Define a lambda function to calculate the inverse distance with a small epsilon
        # it is needed because of the weighted classification
        self._inverse_distance = lambda x: 1/(1e-6 + x) 

    def fit(self, X_train, y_train):
        # Store the training data and labels
        self.X_train = X_train
        self.y_train = y_train
    
    def get_neighbors(self, test_row):
        # Calculate the distances between a test row and all rows in the training data using the Euclidean function
        # Since X_train is dataframe, it can be used with apply method.
        # .apply() basically operates some function on the whole dataframe rows or columns.
        # In the example below, we used with wiht args. args stands for the current test_row.
        # Every Iteration returns a distance between the test dataframe and test_row.
        distances = self.X_train.apply(euclidean, args=(test_row,), axis=1)  
        return distances
    
    def get_classes(self, neighbors):
        # Get the classes (labels) of the nearest neighbors based on their distances
        # 'neighbors' is a pandas Series of distances
        ordered = [neighbors.index[i] for i in range(self.n_neighbors)]        
        classes = [self.y_train.loc[i] for i in ordered]  
        return classes

    def single_predict(self, test_row):
        # Predict the label for a single test row
        neighbors = self.get_neighbors(test_row)
        neighbors.sort_values(inplace=True)
        neighbors = neighbors[:self.n_neighbors]
        classes = self.get_classes(neighbors)            
        prediction = Counter()

        if self.weighted == True:    
            # If weighted classification is enabled, apply the inverse distance function to the neighbors
            neighbors = neighbors.apply(self._inverse_distance)
            # Weighted prediction: Give more weight to closer neighbors
            for i, row in enumerate(neighbors.values):
                prediction[classes[i]] += row
            # Return the label with the highest weighted vote as the prediction
            return prediction.most_common(1)[0][0]
        
        # Unweighted prediction: Use a simple majority vote
        prediction.update(classes)    
        return prediction.most_common(1)[0][0]

    def predict(self, X_test):
        # Convert the test data to a pandas DataFrame if it's a sparse matrix
        if isinstance(X_test, csr_matrix):
            X_test = pd.DataFrame(X_test.toarray())
            
        # Predict labels for all test rows and return them as a pandas Series
        predictions = [self.single_predict(X_test.loc[index]) for index, row in X_test.iterrows()]
        return pd.Series(predictions)
    

class KNeighborRegressor:
    def __init__(self, n_neighbors=3, weighted = False):
        self.n_neighbors = n_neighbors
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def get_distances(self, X_train, x):
        distances = np.linalg.norm(self.X_train - x, axis=1) 
        return distances

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            distances = self.get_distances(self.X_train, x)
            sorted_indices = np.argsort(distances)
            k_nearest_neighbors = sorted_indices[:self.n_neighbors]

            if self.weighted:
                weights = 1 / (distances[k_nearest_neighbors] + 1e-8)  # Adding a samll number to prevent division by zero error
                weighted_avg = np.sum(weights * self.y_train[k_nearest_neighbors]) / np.sum(weights)
                predictions.append(weighted_avg)
            else:
                avg = np.sum(self.y_train[k_nearest_neighbors]) / self.n_neighbors
                predictions.append(avg)

        return np.array(predictions)

