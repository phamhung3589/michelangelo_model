import numpy as np
import pandas as pd
from scipy import sparse


class Dataset:

    def __init__(self, path):

        # Get training set
        self.training_set = self.load_rating_data_to_sparse_matrix(path + "processed_data.csv")
        # Get test set
        self.test_rating = self.load_rating_file_as_list(path + "processed_data_test.csv")
        # Get negative file for calculate evaluation matrix of model
        self.test_negative = self.load_negative_file(path + "processed_data_test_negative.csv")

    @staticmethod
    def load_rating_data_to_sparse_matrix(file_name):
        """
        save data of rating into sparse matrix to decrease size of data and increase reading speed
        :param file_name:   file containing all rating of (user, package)
        :return:            sparse matrix of rating
        """

        # Read file and save to pandas dataframe
        df = pd.read_csv("../data/processed_data.csv")
        # Count number of users and items
        num_user = len(df["User"].unique())
        num_item = len(df["package"].unique())

        # Initial sparse matrix with size (num_user, num_item)
        sparse_mat = sparse.dok_matrix((num_user, num_item), dtype=np.float32)
        with open(file_name, "r") as f:
            line = f.readline()
            # Check condition of line
            while line is not None and line != "":
                # Split data into list (user, package, rating)
                arr = line.split(",")
                # Skip header
                if arr[0].isnumeric():
                    user, package, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if rating > 0:
                        # Saving rating into sparse matrix (only save values 1 instead of value of rating)
                        sparse_mat[user, package] = 1

                # Read next line
                line = f.readline()

        return sparse_mat

    @staticmethod
    def load_rating_file_as_list(file_name):
        """
        Load data from rating file and save it into list of rating and list of label
        :param file_name:   File containing all data
        :return:            list of pair (user, item) and list of labels
        """

        # Initial list of rating and labels
        ratings = []
        labels = []
        # Read line by line of test set
        with open(file_name, "r") as f:
            line = f.readline()
            # Check condition of line
            while line is not None and line != "":
                # Split data into list (user, package, rating)
                arr = line.split(",")
                # Skip header
                if arr[0].isnumeric():
                    # Get user, item and label from list of rating data
                    user, item, label = int(arr[0]), int(arr[1]), int(arr[2])
                    # add pair of (user, item) into lust rating
                    ratings.append([user, item])
                    # Assign label for pair (user, item) - if rating > 0 label = 1
                    if label > 0:
                        label = 1
                        labels.append(label)
                # Read next line
                line = f.readline()
        # return list of rating and labels
        rating_list = ratings, labels

        return rating_list

    @staticmethod
    def load_negative_file(file_name):
        """
        Reading negative package of all user in test set and save into a list
        :param file_name:   File containing data of negative package for all user
        :return:            list of negative package for all user in test set
        """

        # Initial negative list to save all negative packages that user've not been subscribed
        negative_list = []
        # Read line by line of negative packages data
        with open(file_name, "r") as f:
            line = f.readline()
            # Check condition
            while line is not None and line != "":
                # Split line into list of (pair(user, item), item_negative_1, item_negative_2, ...)
                arr = line.split(",")
                # Initial list of negative
                negatives = []
                # Skip pair of (user, item)
                for x in arr[1:]:
                    # Append all negatives package to list
                    negatives.append(int(x))
                # Append all negative packages of each user to list
                # Each row correspond to pair of (user, item) in test set
                negative_list.append(negatives)
                # Read next line
                line = f.readline()

        return negative_list


if __name__ == "__main__":
    dataset = Dataset("../data/")
    negative = dataset.test_negative
