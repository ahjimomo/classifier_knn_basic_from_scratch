# 1. Project Preparation
# 1A. Loading libraries
import os
import numpy as np
import pandas as pd
import math
import statistics as stats
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import classification_report, confusion_matrix

# 1B. Additional Information
"""
For our project, to ensure reproducitivity, random.seed will be set at `123` 
"""

# 2. Loading Dataset
# 2A. Function provided for loading data
def loadData(filename):
    # Load data from file into X
    X = []
    count = 0
    
    text_file = open(filename, "r")
    lines = text_file.readlines()
        
    for line in lines:
        X.append([])
        words = line.split(",")
        # Convert values of the first attribute into float
        for word in words:
            if (word=='M'):
                word = 0.333
            if (word=='F'):
                word = 0.666
            if (word=='I'):
                word = 1
            X[count].append(float(word))
        count += 1
    
    return np.asarray(X)

# 2B. TASK A: Loading dataset
#print(f"Current working directory: {os.getcwd()}")                                                              # Current working directory
#os.chdir("***")                                                                                                 # Changing working directory
#print(f"Updated working directory: {os.getcwd()}")                                                              # Updated working directory

a_raw = loadData("data/abalone.data")

# 2C. Checking that data has been loaded successfully
#print(f"shape of dataset: {a_raw.shape}")     # 4,177 rows and 9 columns
#print(f"Number of dimensions: {a_raw.ndim}")  # 2D
#print(f"Type of array: {a_raw.dtype}")        # Float64
#a_raw[0]

# 2D. Creating pandas dataframe with headers for ease of EDA
headers = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight", "rings"]
df = pd.DataFrame(a_raw, columns = headers)
#df.head()
#df.info()
#df.describe(include = "all").transpose()

#test = a_raw[0:3].copy()
#test

# 3. Data Preparation
# 3A. TASK B: Create function `dataNorm` that normalize dataset
def dataNorm(my_array):
    """
    dataNorm takes in an array and performs max-min normalization across all features within the range of [0:1]
    
    Input:
        - numpy array
    Output:
        - numpy array with normalized values
    Formula for mix-max normalization:
        ( row[i] - min(column) ) / ( (max(column) - min(column) ) 
    """
    # Duplicate array into result array
    result = my_array.copy()

    # Obtaining a list of all min, max values for each column
    minmax = []                             # Empty list to store values of every column
    for i in range(len(result[0]) - 1):     # Iterate through all independent variables (Columns)
        temp = []
        for row in result:                  # Iterate through all rows (rows)
            temp.append(row[i])             # Append values of row in temp list

        min_val = min(temp)                 # Obtain minimal value for column[i]
        max_val = max(temp)                 # Obtain maximum value for column[i]
        minmax.append([min_val, max_val])


    # Normalizing all attributes of dataset across each row
    for row in result:                      # Iterating through every row (record) in array
        for i in range(len(row) - 1):       # For every independent value in the row (columns)
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    # Output: Normalized array
    return result

# 3B. Function to test normalization function
def testNorm(X_norm):
    xMerged = np.copy(X_norm[0])
    # Merge datasets
    for i in range(len(X_norm)-1):
        xMerged = np.concatenate((xMerged,X_norm[i+1]))
    print(f"{np.mean(xMerged,axis=0)}\n")
    print(f"{np.sum(xMerged,axis=0)}\n")

# 3C. Implementation & check of max-min normalization
X_norm = dataNorm(a_raw)
testNorm([X_norm])

# 4. Splitting the normalized dataset
# 4A. TASK C(i): Train-and-Split Method
def splitTT(norm_array, percent):
    """
    Perform a percentage split split based on preferred precentage provided in decimal [0.0 - 0.1] on provided array and returns 2 arrays, x_train and x_test

    Input:
        - norm_array: Array provided for splitting into training and testing arrays
        - precent_splt: Float value between range of [0 - 1] which denotes the percentage of split for the training dataset
    Output:
        - X_split: A list containing two (2) arrays, [train_set, test_set]
    """
    # Create list to store results
    X_split = []

    # Set random seed to ensure reproducitivity
    np.random.seed(123)

    # Generate copy of array and shuffle dataset
    x_copy = norm_array.copy()
    np.random.shuffle(x_copy)        # Shuffling the rows within array

    # Tabulate indexes for splitting
    train_index = round(len(norm_array) * percent)

    # Create & splitted datasets
    train_set = norm_array[:train_index].copy()
    test_set = norm_array[train_index:].copy()
    X_split.append(train_set)
    X_split.append(test_set)

    return X_split

# 4B. Testing splitTT() Method
X_split = splitTT(X_norm, 0.7)
#testNorm(X_split)

# TASK 4C(ii):  k-fold split method
def splitCV(norm_array, k_value):
    """
    Perform a k-fold split split based on preferred k-value on provided array and returns k-number of arrays in a list

    Input:
        - norm_array: Array provided for splitting into training and testing arrays
        - k_value: Number of sets for arrays to be splitted into, floor division will be utilized with overflowing values (if any) placed within the last set
    Output:
        - X_split: A list containing k-number of lists containing splitted array
    """
    # Create list to store results
    X_split = []

    # Set random seed to ensure reproducitivity
    np.random.seed(123)

    # Generate copy of array and shuffle dataset
    x_copy = norm_array.copy()
    np.random.shuffle(x_copy)        # Shuffling the rows within array

    # Determine index for split
    split_index = len(x_copy) // k_value

    # Iterate through array & split rows into k-groups
    start_i = 0
    end_i = split_index

    for i in range(k_value):
        if i == (k_value - 1):                          # Case for last group
            X_split.append(x_copy[start_i:].copy())     # For last group, all leftover values to be added in
        else:
            X_split.append(x_copy[start_i:end_i].copy())
            start_i = end_i                             # Update start index from previous end index
            end_i += split_index                        # Update end index + another group of split index

    return X_split

# 4D. Testing splitCV() Method
cv_split = splitCV(X_norm, 10)
#len(cv_split)
#for i in cv_split:
#    print(len(i))

testNorm(cv_split)

# 5. Implementing KNN() and compute accuracy for comparison
# 5A. K-nearest neighbour algorithm with euclidean distance
def knn(X_train, X_test, K, out = 0):
    """
    K-nearest neighbour algorithm that takes in the training array and testing array, and runs the knn algorithm with euclidean distance based
    on specified k-value for a classification problem (use of mode), return the model's accuracy.

    Input:
        - X_train: array of training dataset with dependent variables
        - X_test: array of testing dataset with dependent variables
        - K: k-value for number of nearest neighbours based on euclidean distance
        - out (optional): 0 refers to accuracy, returns accuracy by default, not 0 will return list of prediction results and original class in list
    Output:
        - (Default, output = 0) Accuracy: Returns accuracy in decimal values [0:1]
        - (output = 1) List: Returns a list containing two lists, list of predicted results and list of true label values
    """
    # Helper function to calculate euclidean_distance between 2 points
    def euclidean_distance(x1, x2):
        # Result to store euclidean distance between 2 points
        e_distance = 0

        # Compute sum of euclidean distance before square : (x1 - x2) ^ 2
        for i in range(len(x1)):                            # Iterate through each row
            e_distance += math.pow(x1[i] - x2[i], 2)        # sum all sqaured difference in distances for every column

        return math.sqrt(e_distance)                        # Return square root of the sum of all squared differences

    # Help function to predict single row
    def single_knn(X_train, value, K):
        neighbours_table = []
        sorted_table = []

        # Run through each row in training data and tabulate the euclidean_distance
        for index, row in enumerate(X_train):                           # Iterate through every row
            e_distance = euclidean_distance(row[:-1], value[:-1])    # Tabulate euclidean_distance for independent variables
        
            neighbours_table.append((e_distance, index))                # Append distance and index sets to a table

        # Sort the collection of e_distance results from smallest to largest
        sorted_table = sorted(neighbours_table)

        # Obtain the k-number of results 
        k_sorted_table = sorted_table[:K]

        # Obtain the result of the top k-results based on the sorted_table
        results_table =[]
        for distance, index in k_sorted_table:
            results_table.append(X_train[index][-1])                # [index] brings us to the training set and -1 refers to the last column: Ring

        # Compute floor of mode (most common) of ring sizes
        return stats.mode(results_table)



    # Create table to store the collection of all the results
    results_table = []
    check_table = []
    correct_counter = 0

    # Run through each row of X_test to tabulate and append results into our results table
    for i in range(len(X_test)):
        results_table.append(single_knn(X_train, X_test[i], K))
        check_table.append(X_test[i][-1])

        if results_table[i] == X_test[i][-1]:   # Check if returned result == actual result
            correct_counter += 1

    # Tabulate accuracy
    accuracy = (correct_counter / len(X_test)) * 100

    # returns output depending of user's preference
    if out == 0:
        return accuracy
    else:
        return [results_table, check_table]

# 5B. Testing KNN() function
check = knn(X_split[0], X_split[1], 2)
print(check)

# 6. Obtain accuracy & timing of knn model based on requirements
# 6A. Function to prepare test and split for splitCV() datasets
def cv_prep(list_of_array):
    """
    Preparation for k-fold split models with last group as the test set and the last group [-1] as training set.
    """
    # Empty list to store training and testing set
    X_split = []

    # Create copy of input array
    train_copy = np.copy(list_of_array[:-1])
    test = np.copy(list_of_array[-1])
    train = np.copy(list_of_array[0])

    # Concatenate train_copy
    for i in range(len(train_copy)-1):
        train = np.concatenate((train, train_copy[i+1]))

    # Return training and testing sets
    X_split.append(train)
    X_split.append(test)
    return X_split

# 6B. Testing cv_prep() function
#merged_split = cv_prep(cv_split)
#len(merged_split)
#len(merged_split[0])
#len(merged_split[1])

# 6C. Preparation of test dataset
split_tset = [0.7, 0.6, 0.5]
fold_tset = [5, 10, 15]
k_val = [1, 5, 10, 15, 20]

# 6D. TASK 4D & 4F(i): Compute accuracy & run-time for train-and-test models
tt_accuracy = []                                    # list to store train-and-run accuracies
tt_run_time = []                                    # list to store train-and-run run-time

for i in range(len(split_tset)):                    # Iterate through split_tset of 0.7, 0.6, 0.5
    X_split = splitTT(X_norm, split_tset[i])        # Get test set and training set

    for j in range(len(k_val)):                     # Iterate through k_val of 1, 5, 10, 15, 20
        start_time = time.time()
        a = knn(X_split[0], X_split[1], k_val[j])   # Run k-means algorithm
        end_time = time.time()
        run_time = end_time - start_time            # Compute run time

        tt_accuracy.append(a)
        tt_run_time.append(run_time)

        print(f"For TT model with {split_tset[i]} size for training, and k-value of {k_val[j]}. "
              f"The accuracy is {a:.2f}%, and took a run time of {run_time:.2f} seconds.")

# 6E. TASK 4D & 4F(ii): Compute accuracy & run-time for K-fold models
kf_accuracy = []                                    # list to store cross-valid accuracies
kf_run_time = []                                    # list to store cross-valid run-time

for i in range(len(fold_tset)):                     # Iterate through split_tset of 5, 10, 15
    X_list = splitCV(X_norm, fold_tset[i])          # Get list of splitted array
    X_split = cv_prep(X_list)                       # Get the train and test set

    for j in range(len(k_val)):                     # Iterate through k_val of 1, 5, 10, 15, 20
        start_time = time.time()
        a = knn(X_split[0], X_split[1], k_val[j])   # Run knn algorithm
        end_time = time.time()
        run_time = end_time - start_time            # Compute run time

        kf_accuracy.append(a)
        kf_run_time.append(run_time)

        print(f"For Cross-validation model with {fold_tset[i]} k-value for training, and k-value of {k_val[j]}. "
              f"The accuracy is {a:.2f}%, and took a run time of {run_time:.2f} seconds.")

# 7. TASK E: Classification Report for 5-fold cross validation with K = 15
E_list = splitCV(X_norm, 5)                         # Get list of splitted array
E_split = cv_prep(E_list)                           # Get the train and test set

lst = knn(E_split[0], E_split[1], 15, 1)

print(classification_report(y_true = lst[1], y_pred = lst[0], zero_division = 0))
#confusion_matrix(y_true = lst[1], y_pred = lst[0])

# 8. Basic investigation into distribution of label
plt.figure(figsize=(10,10))
sns.histplot(df.rings)
plt.title('Distribution of Rings size')
plt.show()

# 9. Plotting of our knn() algorithm on run-time and accuracy
#9A. Function to split results up
def split_group(lst):
    result_lst = []
    start = 0
    end = 5

    for i in range(3):
        result_lst.append(lst[start:end])
        start = end
        end += 5
    return result_lst

# 9B. Preparation of lists for plot
tt_a_lst = split_group(tt_accuracy)
tt_r_lst = split_group(tt_run_time)
kf_a_lst = split_group(kf_accuracy)
kf_r_lst = split_group(kf_run_time)

# 9C. Accuracy plot
ax = sns.lineplot(x = k_val, y = tt_a_lst[0], label = "0.7 - 0.3")
ax = sns.lineplot(x = k_val, y = tt_a_lst[1], label = "0.6 - 0.4")
ax = sns.lineplot(x = k_val, y = tt_a_lst[2], label = "0.5 - 0.5")
ax = sns.lineplot(x = k_val, y = kf_a_lst[0], label = "5-Fold")
ax = sns.lineplot(x = k_val, y = kf_a_lst[1], label = "10-Fold")
ax = sns.lineplot(x = k_val, y = kf_a_lst[2], label = "15-Fold")
ax.set(title = "Models Accuracy across K-values on knn() algorithm")
ax.set_xlabel("K-value")
ax.set_ylabel("Accuracy (%)")
plt.show()

# 9D. Run time plot
ax = sns.lineplot(x = k_val, y = tt_r_lst[0], label = "0.7 - 0.3")
ax = sns.lineplot(x = k_val, y = tt_r_lst[1], label = "0.6 - 0.4")
ax = sns.lineplot(x = k_val, y = tt_r_lst[2], label = "0.5 - 0.5")
ax = sns.lineplot(x = k_val, y = kf_r_lst[0], label = "5-Fold")
ax = sns.lineplot(x = k_val, y = kf_r_lst[1], label = "10-Fold")
ax = sns.lineplot(x = k_val, y = kf_r_lst[2], label = "15-Fold")
ax.set(title = "Models Run-Time across K-values on knn() algorithm")
ax.set_xlabel("K-value")
ax.set_ylabel("Run Time (Seconds)")
plt.show()

