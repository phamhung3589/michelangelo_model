import os
import sys
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from recommendation.model.constants import Constants
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import random
import implicit
import math
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 1000)
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))


class ImplicitFeedback:

    def __init__(self):
        # Folder containing all data
        self.HomeDir = "../data/"
        # self.HomeDir = "../recommendation/data/"
        # File containing all packages
        self.all_packages = "all_package.csv"
        # File containing all users
        self.all_users = "all_user.csv"
        # File containing processed data
        self.processed_data = "processed_data.csv"
        # File containing processed data for testing
        self.processed_data_test = "processed_data_test.csv"
        # Create instance variable packages to save all packages in database with name and id of it
        self.packages = pd.read_csv(join(self.HomeDir, self.all_packages)).to_dict().get("package")
        # Initial parameters for predict packages
        self.training_set, self.user_vecs, self.item_vecs, self.list_unique_user = None, None, None, None
        self.true_labels, self.packages_subscribed = None, None

    def import_data(self):
        """
        Load data from dataset store and save rating matrix into scipy sparse matrix
        :return: scipy sparse matrix
        """

        # Load data processed data (rating matrix) into pandas dataframe
        df_data = pd.read_csv(join(self.HomeDir, self.processed_data))
        # Create row (users) for sparse matrix
        rows = df_data["User"]
        # Create column (packages) for sparse matrix
        columns = df_data["package"]
        # Count number of item and number of user in all dataset
        num_users, num_items = len(rows.unique()), len(columns.unique())
        # rating data
        rating_offer = list(df_data["ratings"])
        # Create sparse matrix from (row, column, and rating offer)
        offer_purchased = sparse.csr_matrix((rating_offer, (list(rows), list(columns))), shape=(num_users, num_items))

        return offer_purchased

    def train_test_split(self, ratings, pct_test=0.05):
        """
        Splitting data into 2 types of dataset: 1 for training set and 1 for test set
        :param ratings:     rating data of all user (be used to split 2 types )
        :param pct_test:    percentage of data to remove from training set and move to test set
        :return:            training set and test set
        """

        training_set = ratings.copy()
        # Get random num-sample from all pairs of (user, item)
        # Disadvantage: Using random sample have many problems (ex: user only have subscribed 1 package =>
        # choose in testing => training don't have data about this user )
        # sample = random.sample(nonzero_pairs, num_sample)
        # solved: Using data have been separated from raw data
        sample = pd.read_csv(join(self.HomeDir, self.processed_data_test))[["User", "package"]].values
        # Return all user index from sample
        user_ind = [index[0] for index in sample]
        # return all item index from sample
        item_ind = [index[1] for index in sample]
        # return true labels corresponding to user and item
        # Set rating values for pair of (user, item) in sample from training set to 0
        training_set[user_ind, item_ind] = 0
        # Eliminate all sample from trianing set
        training_set.elminate_zeros()
        # Convert list to numpy array
        user_ind = np.array(user_ind)
        item_ind = np.array(item_ind)
        # Create list of unique user to evaluate
        self.list_unique_user = list(user_ind)
        # Convert to numpy array
        self.true_labels = item_ind.reshape((-1, 1))
        # Test set
        test_set = self.list_unique_user, self.true_labels
        # Initial instance variable training set
        self.training_set =training_set

        return self.training_set, test_set

    def train(self, training_set, alpha, factors, regularization, iterations):
        """
        Using lib implicit python for recommend packages for that user
        :param training_set:    training data
        :param alpha:           linear_scaling factor alpha
        :param factors:         latent vectors for each user and item
        :param regularization:  parameter for avoiding overfit data
        :param iterations:      iteration of algorithm
        :return:                initialize instance variable training_set, user_vecs and item_vecs
        """

        # Training algorithm with lib implicit an initial instance variable user_vecs and item_vecs
        self.user_vecs, self.item_vecs = \
            implicit.alternating_least_squares((training_set*alpha).astype('double'), factors=factors,
                                               regularization=regularization, iterations=iterations)

    def predict(self, user, num_items=10, print_item="true"):
        """
        Using user_vecs and item_vecs to recommend packages for only 1 user
        :param user:        user that need to recommend packages
        :param num_items:   number of items recommend
        :param print_item:  print packages of that user or not on console
        :return:            index of recommendation packages
        """

        # Get the ratings from training set rating matrix, # Add 1 to everything,
        # so that items not purchased yet become equate to 1
        pref_vec = self.training_set[user, :].toarray().reshape(-1) + 1
        # Make everything already purchased zaro
        pref_vec[pref_vec > 1] = 0
        # packages thar user've been subscribed
        packages_subscribed = np.argwhere(pref_vec == 0).reshape(-1)
        # Get dot product of user vector and all item vectors
        rec_vector = self.user_vecs[user, :].dot(self.item_vecs.T)
        # Scale this recommendation vector between 0 and 1
        rec_vector_scaled = MinMaxScaler().fit_transform(rec_vector.reshape(-1, 1))[:, 0]
        # Vector of recommendation for all item -> output format: numpy array with shape=(1, num_items)
        recommend_vector = pref_vec * rec_vector_scaled
        # Items already purchased have their recommendation multiplied by zero
        product_idx = np.argsort(recommend_vector, kind="mergesort")[::-1][:num_items]
        # Print packages that user've been subscribed and recommendation packages
        if print_item == "true":
            print("user: ", user)
            print("\tPackage subscribed: ", self.map_item_name(packages_subscribed))
            print("\tPackage recommended: ", self.map_item_name(product_idx))

        return product_idx

    def predict_list(self, user_list, num_items, print_item="true"):
        """
        Using user_vecs and item_vecs to recommend packages for list of user
        :param uesr_list:       list of users that need to recommend packages
        :param num_items:       number of items recommend
        :param print_item:      print packages od that user or not on console
        :return:                index of recommendation packages for each user in list
        """

        # Check length of each user - if list user is MSISDN - map into id of each user
        if user_list[0] > 100000000:
            # Read all user id with corresponding MSISDN - set index to column user
            df_all_user_id = pd.read_csv(join(self.HomeDir, self.all_users)).reset_index()
            df_all_user_id.set_index("user", inplace=True)
            # Read list user and save it dataframe with column name: user
            df_user_recommend = pd.DataFrame(list(user_list), columns=["user"])
            # set index of df with index of column user
            df_user_recommend.set_index("user", inplace=True)
            # Merge df contain list user for recommend with list of all user to get id of each msisdn user
            df_user_recommend = pd.merge(df_user_recommend, df_all_user_id, how="left", left_index=True, right_index=True)
            # Save id of user to user list
            user_list = df_user_recommend["index"].values.tolist()


        # Get the ratings from training set rating matrix, # Add 1 to everything,
        # so that items not purchased yet become equate to 1
        pref_vec = self.training_set[list(user_list), :].toarray() + 1
        # Make everything already purchased zaro
        pref_vec[pref_vec > 1] = 0
        # Check condition if user_list belongs to list user test or not
        if self.list_unique_user == user_list:
            packages_subscribed = self.true_labels
        else:
            # packages that user've been subscribed
            packages_subscribed = np.argwhere(pref_vec == 0)
            # Split data into k sub array with k = len(user_list) - Ex with user_list = [2, 3, 4]
            # [array([[0, 222], [0, 421]]), array([[1, 399]]), array([[2, 444], [2, 577]]) ]
            packages_subscribed = np.split(packages_subscribed, np.unique(packages_subscribed[:, 0],
                                                                          return_index=True)[1][1:], axis=0)
        # Initial instance variable
        self.packages_subscribed = packages_subscribed
        rec_vector = self.user_vecs[user_list, :].dot(self.item_vecs.T)
        # Scale this recommendation vector between 0 and 1
        rec_vector_scaled = MinMaxScaler().fit_transform(rec_vector.T).T
        # Vector of recommendation for all item -> output format: numpy array with shape=(1, num_items)
        recommend_vector = pref_vec * rec_vector_scaled
        # Items already purchased have their recommendation multiplied by zero: format: shape=(len(user_list), num_item)
        product_idx = np.argsort(recommend_vector, kind="mergesort")[:, ::-1][:, :num_items]
        # Print packages that user've been subscribed and recommendation packages
        if print_item == "true":
            for i in range(len(user_list)):
                print("user: ", user_list[i])
                # i is row of matrix packages_subscribed - corresponding to vector of recommendation packages for user
                if self.list_unique_user == user_list:
                    print("\tPackage subscribed: ", self.map_item_name(packages_subscribed[i]))
                else:
                    print("\tPackage subscribed: ", self.map_item_name(packages_subscribed[i][:, 1]))
                print("\tPackage recommended: ", self.map_item_name(product_idx[i]))

        return product_idx

    def map_item_name(self, list_item_id):
        """
        Map item id with name of it in list item bank
        :param list_user_id:    List of id items
        :return:                List containing all name corresponding with id of each item
        """

        return [self.packages.get(k) for k in list_item_id]

    def map_user_name(self, list_user_id):
        """
        Map item id with name of it in list user bank
        :param list_user_id:    List of id users
        :return:                List containing all name corresponding with id of each user
        """

        # Create list of users name
        user_name_list = pd.read_csv(join(self.HomeDir, self.all_users)).to_dict().get("user")

        return [user_name_list.get(k) for k in list_user_id]

    def evaluate(self, true_labels, predict_labels, type_metric="ndcg"):
        """
        Using evaluation metric to evaluate performance of algorithm
        :param true_labels:         Real packages that user subscribe
        :param predict_labels:      Recommendation packages for that user
        :param type_metric:         using ndcg and hit_rate
        :return:                    mean of ndcg or hit_rate over all user
        """

        # Using ndcg metric
        if type_metric == "ndcg":
            # Initial ndcg metric
            ndcgs = []
            # Loop over all row of predict label - corresponding to recommendation packages for each user
            for i in range(predict_labels.shape[0]):
                # Recommendation packages of i-th user
                item_rec = predict_labels[i]
                # loop over all real packages of i-th user
                for item in true_labels[i]:
                    # Append each ndcg of each user - package to list
                    ndcgs.append(self.get_ndcg(item, item_rec))
            # Compute mean over all ndcg metric of all user
            ndcg_mean = np.mean(ndcgs)

            return ndcg_mean

        # Using hit_rate metric:
        elif type_metric == "hit_rate":
            # Initial ndcg metric
            hit_rates = []
            # Loop over all row of predict label - corresponding to recommendation packages for each user
            for i in range(predict_labels.shape[0]):
                # Recommendation packages of i-th user
                item_rec = predict_labels[i]
                # loop over all real packages of i-th user
                for item in true_labels[i]:
                    # Append each ndcg of each user - package to list
                    hit_rates.append(self.get_hit_rate(item, item_rec))
            # Compute mean over all ndcg metric of all user
            hr_mean = np.mean(hit_rates)

            return hr_mean

        # Return 0 with all remaining cases
        else:
            return 0

    @staticmethod
    def get_ndcg(item, item_rec):
        """
        Using ndcg_metric metric to evaluate algorithm
        i = posistion of real package in list of recommendation packages
        formula: log(2) / (log(2+i))
        :param item:        Only 1 real item
        :param item_rec:    List of recommendation item
        :return:            ndcg_metric of item recommend
        """

        # initial ndcg_metric metric
        ndcg_metric = 0
        # Get position of real time in list of recommendation packages
        position = np.where(item_rec == item)[0].tolist()
        # Check condition of real time in list of recommendation packages or not
        if len(position) != 0:
            # Calculate ndcg_metric metric in formula
            ndcg_metric = math.log(2) / (math.log(2 + position[0]))

        return ndcg_metric

    @staticmethod
    def get_hit_rate(item, item_rec):
        """
        using ndcg metric to evaluate algorithm
        i = posistion of real package in list of recommendation packages
        formula: hit_rate = 1
        :param item:        Only 1 real item
        :param item_rec:    List of recommendation item
        :return:            hit_rate_metric of item recommend
        """
        # initial hit_rate metric
        hit_rate = 0
        # Get position of real time in list of recommendation packages
        position = np.where(item_rec == item)[0].tolist()
        # Check condition of real time in list of recommendation packages or not
        if len(position) != 0:
            # Calculate hit_rate metric in formula
            hit_rate = 1

        return hit_rate

    def save_model(self, file_user, file_item):
        """
        :param file_user:   File to save user_vec
        :param file_item:   File to save item_vec
        :return:            Saving user_vec anf item_vec (output of ALS model) in local
        """

        # Saving to "npy" file
        np.save(join(Constants.SAVE_MODEL, file_user), self.user_vecs)
        np.save(join(Constants.SAVE_MODEL, file_item), self.item_vecs)
        print("Saving user_vec and item_vec to npy has been done ")

    def load_model(self, file_user, file_item):
        """
        :param file_user:   File to load user_vec
        :param file_item:   File to load item_vec
        :return:            Loading user_vec anf item_vec (output of ALS model) from local
        """

        # Loading to "npy" file
        self.user_vecs = np.load(join(Constants.SAVE_MODEL, file_user))
        self.item_vecs = np.load(join(Constants.SAVE_MODEL, file_item))
        print("Loading user_vec and item_vec from npy file has been done ")

    def coverage_item(self, list_id_recommend):
        """
        Count number of items each item appear in list of recommendation items
        :param list_id_recommend:   List of all recommendation items
        :return:  List frequency of each items - Ex [ [item=3, freq=1], [item=4, freq=2], [item=5, freq=10] ]
        """

        # Flat numpy array into -D list
        list_flat = np.hstack(list_id_recommend)
        # Count frequency of each item in bin array - len(count_bin) = max(list_flat) + 1 -
        # Ex: Value 4 appear 7 times => position 5 in count_bin = 7
        count_bin = np.bincount(list_flat)
        # List of all values in list_id with order from smaller to bigger
        list_non_zeros = np.nonzero(count_bin)[0]
        # Zip values from values with frequency of itand convert to vertical matrix - Ex format [[item, freq], [], []]
        list_item_freq = np.vstack((list_non_zeros, count_bin[list_non_zeros])).T
        # Sorting column 1 (freq) with descending order
        list_item_freq = list_item_freq[list_item_freq[:, 1].argsort()][::-1]
        # Top 5 item with highest frequency
        top_5_items = list_item_freq[:5, 0]
        # Map 5 item with name
        top_5_items_name = self.map_item_name(top_5_items)

        # Return 5%, 20% and 50% item coverage
        num_item_recommend = list_item_freq.shape[0]
        total_recommend = list_item_freq.sum(axis=0)[1]
        # Return 5% item coverage - Formula: Total times of recommend of 5% highest frequency items / total recommend
        item_coverage_5 = list_item_freq[:int(num_item_recommend*0.05)].sum(axis=0)[1] / total_recommend
        # Return 5% item coverage - Formula: Total times of recommend of 5% highest frequency items / total recommend
        item_coverage_20 = list_item_freq[:int(num_item_recommend*0.2)].sum(axis=0)[1] / total_recommend
        # Return 5% item coverage - Formula: Total times of recommend of 5% highest frequency items / total recommend
        item_coverage_50 = list_item_freq[:int(num_item_recommend*0.5)].sum(axis=0)[1] / total_recommend
        # return list of item coverage
        item_coverage = {"item_count": self.training_set.shape[1], "item_coverage_5": item_coverage_5,
                                                                   "item_coverage_20": item_coverage_20,
                                                                   "item_coverage_50": item_coverage_50}

        # Get name of packages corresponding to id of packages
        name_package = self.map_item_name(list_item_freq[:, 0])
        # Get frequency of each package
        frequency = list_item_freq[:, 1]
        # Map name of package with frequency of it
        list_item_freq = [[name_package[i], int(frequency[i])] for i in range(len(name_package))]

        return list_item_freq, top_5_items_name, item_coverage

    def convert_msisdn(self, list_packages_id, list_user):
        """
        Convert pair of id (user, package) to name of it
        :param list_packages_id:    List of id_packages
        :param list_user:           List of user packages
        :return:                    dataframe pandas contain 3 columns: User, package_subscribed and package_recommend
        """

        # Initial list of user name
        # Check length of each user - if list user is MSISDN - return list user to list user name
        if list_user[0] > 100000000:
            list_user_name = list_user
        else:
            list_user_name = self.map_user_name(list_user)

        # Initiate list of package name
        list_packages_name = []
        # Loop over each row i of list package (corresponding to recommendation packages os user i)
        for row in range(list_packages_id.shape[0]):
            # Map id of package with name of it - using join function to concat all string values in list
            list_packages_name.append(",".join(self.map_item_name(list_packages_id[row, :])))
        # Create DataFrame with data = list_packages_name and index = list_user
        df_api = pd.DataFrame(data=list_packages_name, columns=["package_recommend"], index=list_user_name)
        # Reset index (User column)
        df_api.reset_index(inplace=True)
        # Rename name of column index to user
        df_api.rename(columns={"index": "User"}, inplace=True)
        # Format of list_package_subscribed: [array[ [user1, item1], [user1, item2]], array[ [user2, item1], ... ]] -
        # Skip all user and concat all items that user X subscribed into 1 list
        list_package_subscribed = [self.map_item_name(np.array(element)[:, 1]) for element in self.packages_subscribed]
        # Join all name of packages in 1 list with separator ","
        concat_list_packages = [",".join(element) for element in list_package_subscribed]
        # Create column package_subscribed for dataframe api
        df_api["package_subscribed"] = concat_list_packages

        return df_api

    @staticmethod
    def plot_graph(list_item):
        """
        Plot data using list of item or something
        :param list_item:   List containing data from frequency of list_item
        :return:            graph
        """

        plt.figure()
        df = pd.DataFrame(data=list_item, columns=["Item", "Frequency"])
        df.Frequency.plot.bar(color='g')
        plt.xlabel("Item_id")
        plt.ylabel("Frequency")
        plt.xticks([])
        plt.show()


if __name__ == "__main__":

    # Create object implicit
    implicit_feedback = ImplicitFeedback()

    # Create sparse matrix for recommendation
    offer_purchased_sparse = implicit_feedback.import_data()
    print("import done")
    # Split raw rating data to training set and test set
    training_set, test_set = implicit_feedback.train_test_split(offer_purchased_sparse)
    print("Split data done")
    # Create list of unique user to evaluate
    list_unique_user, true_labels = test_set

    # Initiate parameter for algorithm
    alphas = [28]
    factors = [8]
    regularizations = [1.05]
    iters = [100]

    best_alpha, best_factor, best_regularization, best_iter = 0, 0, 0, 0
    best_hr, best_ndcg = 0, 0

    for alpha in alphas:
        for factor in factors:
            for regularization in regularizations:
                for iteration in iters:
                    # training algorithm
                    implicit_feedback.train(training_set=training_set,
                                            alpha=alpha,
                                            factors=factor,
                                            regularization=regularization,
                                            iterations=iteration)

                    # Predict only 1 user
                    # product_id = implicit_feedback.predict(user=2, num_items=5, print_item="true")

                    # Predict list of user
                    list_product_id = implicit_feedback.predict_list(user_list=list_unique_user,
                                                                     num_items=10, print_item="false")

                    # Evaluate model
                    ndcg = implicit_feedback.evaluate(true_labels=true_labels, predict_labels=list_product_id, type_metric='ndcg')
                    hr = implicit_feedback.evaluate(true_labels=true_labels, predict_labels=list_product_id, type_metric='hit_rate')

                    print("Model implicit feedback with alpha={:d}, factor={:d}, regularization={:.3f}, iteration={:d}"
                          " have evaluation metric: ndcg={:.5f}, hit_rate={:.5f}"\
                          .format(alpha, factor, regularization, iteration, ndcg, hr))

                    # Pick best model
                    if ndcg > best_ndcg:
                        best_validation = best_ndcg
                        best_hr = hr
                        best_alpha = alpha
                        best_factor = factor
                        best_regularization = regularization
                        best_iter = iteration

    print("best model with alpha={:d}, factor={:d}, regularization={:.3f}, iteration={:d} and evaluation metric: "
          "ndcg={:.5f}, hit_rate={:.5f}".format(best_alpha, best_factor, best_regularization, best_iter, best_ndcg, best_hr))

    # training with best model
    implicit_feedback.train(offer_purchased_sparse, alpha=best_alpha,
                                                    factors=best_factor,
                                                    regularization=best_regularization,
                                                    iterations=best_iter)

    # Saving user_vec and item_vec to local
    implicit_feedback.save_model("user_vecs.npy", "item_vecs.npy")

    # # Loading user_vec and item_vec from local file
    # implicit_feedback.load_model("user_vecs.npy", "item_vecs.npy")
    # print("load model done")

    # Predict list of user
    list_product_id = implicit_feedback.predict_list(user_list=range(500), num_items=5, print_item="false")

    # Merge user_id, package_id with MSISDN and name of package
    df_user_item = implicit_feedback.convert_msisdn(list_product_id, list_user=range(500))
    print(df_user_item)
    list_product_freq, top_5_items, item_coverage = implicit_feedback.coverage_item(list_product_id)

    # Plotting data
    implicit_feedback.plot_graph(list_product_freq)
