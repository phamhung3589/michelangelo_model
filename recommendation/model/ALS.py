import os
from collections import Counter
from os.path import join
from os.path import join
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
from math import sqrt
from heapq import nlargest
from recommendation.model.constants import Constants
import math
import pandas as pd

# Setup environment to run with all necessary libs
os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/haitm121/anaconda3/bin/python"
os.environ["PYSPARK_PYTHON"] = "/home/haitm121/anaconda3/bin/python"

class ModelALS:

    def __init__(self):

        # Folder containing all data
        # self.HomeDir = "../data/"
        self.HomeDir = "../recommendation/data/"
        # File containing all packages
        self.all_packages = "all_package.csv"
        # File containing all users
        self.all_users = "all_user.csv"
        # File containing processed data
        self.processed_data = "processed_data.csv"
        # File containing processed data for training
        self.processed_data_training = "processed_data_training.csv"
        # Number of packages
        self.num_package = len(pd.read_csv(join(self.HomeDir, self.all_packages)).values)
        # Create instance variable packages to save all packages in database with name and id of it
        self.packages = pd.read_csv(join(self.HomeDir, self.all_packages)).to_dict().get("package")
        # Initial instance variable model
        self.model = None

    @staticmethod
    def create_spark_environment():
        """
        Setup environment for pyspark to run on cluster or local
        :return: spark config
        """

        # setup environment for pyspark
        conf = SparkConf().setAppName("packagingALS").setMaster("local[*]").\
            set("spark.executor.memory", "50g").\
            set("spark.executor.cores", "35").\
            set("spark.driver.memory", "50g").\
            set("spark.executor.memoryOverhead", "10g").\
            set("spark.memory.fraction", "0.2").\
            set("spark.driver.maxResultSize", "20g").\
            set("spark.driver.extraJavaOptions", "-Xss100m")

        sc = SparkContext(conf=conf)
        # Set log level to display on console screen
        sc.setLogLevel("ERROR")

        return sc

    @staticmethod
    def parse_rating(line):
        """
        Parse a rating record in format: userId::packagesID::rating::timestamp
        :param line:
        :return: (key, value) with key: package % 10 and value is user - package - rating
        """

        # Split line into list of string separated by character ","
        fields = line.strip().split(",")

        return int(fields[1]) % 10, (int(fields[0]), int(fields[1]), int(fields[2]))

    @staticmethod
    def compute_rmse(model, validation):
        """
        Compute RMSE (Root mean square error )
        :param model: ALS model
        :param validation: data for validate model with format: (user, package, rating)
        :return: RMSE metric
        """

        # Map validation data into format (user, package) to model trained predict value of rating
        validate = validation.map(lambda x: (x[0], x[1]))
        # Predict value of each pair of (user, package) - output (Rating(user=?, item=?, rating=?))
        predictions = model.predictAll(validate)
        # Join pair of (user, package) with rating prediction and real rating with format:
        # ( (user, package), (predict_rating, real)rating) ) - and only get value (predict_rating, real_rating )
        predictions_and_rating = predictions.map(lambda x: ((x[0], x[1]), x[2])).\
            join(validation.map(lambda x: ((x[0], x[1]), x[2]))).values()

        # Return Root mean square error of all (predict_rating - real_rating)^2
        return sqrt(predictions_and_rating.map(lambda x: (x[0] - x[1]) ** 2).mean())

    def evaluation_metric(self, model, validation, k=3, type_metric='ndcg'):
        """
        Using evaluation metric (hit_rate, ndcg) with model on validation data
        :param model:           model ALS using to predict
        :param validation:      Validation data
        :param k:               number of recommendation items for evaluate
        :param type_metric:     choose type of evaluation metric: ndcg or hit_rate
        :return:                mean of all ndcg or hit_rate on all users on validation data
        """

        # Map all users on validation data with all prediction packages
        user_all_packages = validation.map(self.map_user_2_prediction_package).flatMapValues(self.flat_map)
        # predict all packages for each user - return (Rating(user= , item= , rating= ))
        predictions = model.predictAll(user_all_packages)
        # Choose k recommendation packages for each user - top k of (Rating(user, item, rating), ...)
        predictions_top_k_packages = predictions.groupBy(lambda x: x[0]).\
            flatMap(lambda g: nlargest(k, g[1], key=lambda x: x[2]))
        # Real label for each user - skip rating for each pair of (user, package)
        # true_label = validation.map(lambda x: (x[0], x[1]) )
        # Map top k user prediction with same format as rdd of true_label
        predictions_top_k_without_rating = predictions_top_k_packages.map(lambda x: (x[0]. x[1]))
        # Group each user with list values of all packages - example: [(user 1, [3,4,5,6])]
        predictions_groupbykey = predictions_top_k_without_rating.reduceByKey(lambda x, y: x + y).sortByKey()

        # Join prediction packages and real packages for each user -
        # Example: [(user 1, [ item_predict([3,4,5,6])), real_item(3)]]
        predict_and_label = predictions_groupbykey.join(validation.map(lambda x: (x[0], x[1])))

        # Calculate type of evaluation metric
        metric = 0
        if type_metric == "ndcg":
            metric = predict_and_label.mapValues(self.get_ndcg).values().mean()

        if type_metric == "hit_rate":
            metric = predict_and_label.mapValues(self.get_hit_rate).values().mean()

        return metric

    @staticmethod
    def get_ndcg(X):
        # Values of ndcg return for each user
        ndcg = 0
        # List of n recommendation packages for user X - Example: [3,4,6,7,4,3]
        values = X[0]
        # Loop over all values of list n packages
        for i in range(len(values)):
            # Check if package user X've subscribed (X[1]) has been contained in list recommendation packages
            if X[1] == values[i]:
                # Calculate metric ndcg
                ndcg = math.log(2) / (math.log(i+2))

        return ndcg

    @staticmethod
    def get_hit_rate(X):
        # Values of hit_rate return for each user
        hr = 0
        # List of n recommendation packages for user X - Example: [3,4,6,7,4,3]
        values = X[0]
        # Loop over all values of list n packages
        for i in range(len(values)):
            # Check if package user X've subscribed (X[1]) has been contained in list recommendation packages
            if X[1] == values[i]:
                # Calculate metric hit rate
                hr = 1

        return hr

    def map_user_2_prediction_package(self, X):
        """
        :param X:
        :return: all pairs of (user, all_packages)
        """

        # Initial packages for user to predict
        prediction_packages = []
        # Loop over all packages in bank
        for i in range(self.num_package):
            # Add package to list
            prediction_packages.append(i)

        return X[0], prediction_packages

    @staticmethod
    def flat_map(X):
        """
        :param X:
        :return: return flat map of key for each values [(key, values), (key, values), ... ]
        """

        return X

    def load_data(self, sc):
        """
        Split dataset into 3 parts: data for training in model, data for validate model, data for testing model
        :param sc: SparkConf
        :return: all packages, training, validation and test
        """

        # Ratings is an RDD of (last digit of timestamp, (userID, movieID, rating))
        data_ratings = sc.textFile(join(self.HomeDir, self.processed_data_training))
        # Skip header
        data_skip_header = data_ratings.filter(lambda x: x[0].isdigit())
        # Split data into format (user, package, rating)
        ratings = data_skip_header.map(self.parse_rating)

        # Load data for testing
        test_data = sc.textFile(join(self.HomeDir, "processed_data_test.csv")).filter(lambda x: x[0].isdigit())
        rating_test = test_data.map(self.parse_rating)

        # Count number of ratings, users and packages
        num_ratings = ratings.count()
        num_users = ratings.map(lambda l: l[1][0]).distinct().count()
        num_items = ratings.map(lambda l: l[1][1]).distinct().count()
        print("Got %d ratings from %d users on %d items " %(num_ratings, num_users, num_items))

        # Split ratings into train (90%), and test (10%)
        # Based on last digit on timestamp and cache them
        # Training, validation, test are all RDDs of (userID, movieID, rating)

        test = rating_test.values()
        training = ratings.values()

        num_training = training.count()
        num_test = test.count()

        print("training: {:d}, Test: {:d} ".format(num_training, num_test))
        # Setup instance variable for packages to save all packages in dataset

        return training, test

    def train(self, num_recommend_item, evaluation_metric, sc):

        # Initialize spark config
        als_model = ModelALS()
        # sc = als_model.create_spark_environment()

        # Packages:     dict of all packages (id: name of packages)
        # Training:     data for training model (60%)
        # Validation:   data for validate score of model (20%)
        # Test:         data for testing (20%)
        training, validation = als_model.load_data(sc)

        # Training the model and evaluate them on validation set
        # ranks: number of hidden layer
        # lambdas: regularization
        ranks = [8]
        lambdas = [0.07, 0.09]
        num_iters = [20]
        best_model = None
        best_validation = -float("inf")
        best_rank = 0
        best_lambda = -1.0
        best_numiter = -1

        # For rank, lmbda, num_iter in itertools.product(ranks, lambdas, num_iters):
        for rank in ranks:
            for lmbda in lambdas:
                for num_iter in num_iters:
                    # train model
                    # model = ALS.train(training, rank=rank, iterations=num_iter, lambda_=lmbda)
                    model = ALS.trainImplicit(training, rank=rank, iterations=num_iter, lambda_=lmbda, alpha=16.0)
                    # Compute evaluation metric with validation data: RMSE, hit_rate, ndcg
                    validation_rmse = self.compute_rmse(model, validation)
                    validation_ndcg = self.evaluation_metric(model, validation, k=num_recommend_item, type_metric=evaluation_metric)
                    print("NDCG (validation) = %f for the model train with " % validation_ndcg +
                          "rank = %d, lambda = %.2f and num iter = %d." % (rank, lmbda, num_iter))

                    # Pick best model
                    if validation_ndcg > best_validation:
                        best_validation = validation_ndcg
                        best_rank = rank
                        best_lambda = lmbda
                        best_numiter = num_iter
                        best_model = model

        # Save best model to local
        best_model.save(sc, Constants.PATH_ALS_MODEL)
        print("choosed best model")
        test_ndcg = self.evaluation_metric(best_model, validation, k=num_recommend_item, type_metric=evaluation_metric)

        print("The best model was trained with rank = %d, lambda = %.2f and num_iter = %d, "
              "and its NDCG on the test set is: %f." % (best_rank, best_lambda, best_numiter, test_ndcg))

        # Compare the best model with a naive baseline that always returns the package with max rating
        package_max_rating = training.union(validation).map(lambda x: (x[1], x[2])).reduceByKey(lambda x, y: x + y)\
                                                                                   .max(key=lambda x: x[1])[0]
        # Map each user with list of (list_packages_subscribed, package_max_rating) -
        # Ex: [(user, [list_package_subscribed=[], max_package_rating]), ...]
        pair_user_packages_naive_ndcg = validation.map(lambda x: (x[0], x[1])).reduceByKey(lambda x, y: x + y)\
                                                  .map(lambda x: (x[0], [x[1], package_max_rating]))
        # Calculate ndcg for naive model
        naive_ndcg = pair_user_packages_naive_ndcg.mapValues(self.get_ndcg).values().mean()
        # Calculate improvement of each ALS model with naive model
        improvement = (test_ndcg - naive_ndcg) / test_ndcg * 100
        print("The best model improves the naive baseline by %.2f" % improvement + "%")

        # Setup instance variable to call after
        self.model = best_model

        return sc

    def load_model(self, spark_config):
        """
        Load model saved from local
        :param spark_config: spark config
        :return:
        """

        model = MatrixFactorizationModel.load(spark_config, Constants.PATH_ALS_MODEL)
        # Create instance variable model to call if in predict function
        self.model = model

    def predict(self, user, num_item_recommend, spark_config):
        """
        Predict user X in all data
        :param user:                user need to be recommend packages
        :param num_item_recommend:  number of item to recommend for user
        :param spark_config:        spark environment
        :return:                    print all packages user've subscribed and packages user've been recommended
        """

        if int(user) > 100000000:
            all_user_id = pd.read_csv(join(self.HomeDir, self.all_users))
            # return index of user corresponding to msisdn
            user = all_user_id[all_user_id["user"] == int(user)].index[0]

        # Read information rating of user: user
        packages_user = pd.read_csv(join(self.HomeDir, self.processed_data), index_col="User")
        # Get all packages which user've subscribed and return list of values - Ex: [4, 7, 34]
        packages_subscribed = packages_user.loc[packages_user.index == user]["package"].values
        # Make personal recommendation
        # Get all id of packages without packages've subscribed
        candidates = spark_config.parallelize([m for m in self.packages if m not in packages_subscribed])
        # Map all package with user, user
        candidates = candidates.map(lambda x: (user, x))
        # Using best model to predict all rating of packages for user: user
        predictions = self.model.predictAll(candidates).collect()
        # Sorting all rating from high to low and get 10 highest values
        packages_prediction = sorted(predictions, key=lambda x: x[2], reverse=True)[:num_item_recommend]

        # Print all packages user 2 have subscribed
        print("user've subscribed thest packages: ")
        for j in range(len(packages_subscribed)):
            print("{:d}: {:s}".format((j+1), self.packages[packages_subscribed[j]]))

        # Print 10 packages recommended for that user
        print("packages recommend: ")
        for i in range(len(packages_prediction)):
            print("{:d}: {:s}".format(i+1, self.packages[packages_prediction[i][1]]))

        # Clear spark config
        spark_config.stop()

        return [self.packages[packages_prediction[i][1]] for i in range(len(packages_prediction))]

    def predict_list(self, list_user, num_item_recommend, spark_config):
        """
        Predict recommendation packages for list of user
        :param list_user:           list of user need to be recommend packages
        :param num_item_recommend:  number of item to recommend for user
        :param spark_config:        spark environment
        :return:                    recommendation packages for each user
        """

        result_dict = {}
        df_user_recommend = None
        if list_user[0] > 100000000:
            # Read all user id with corresponding MSISDN - set index to column user
            df_all_user_id = pd.read_csv(join(self.HomeDir, self.all_users)).reset_index()
            df_all_user_id.set_index("user", inplace=True)
            # Read list user and save it dataframe with column name: user
            df_user_recommend = pd.DataFrame(list(list_user), columns=["user"])
            # set index of df with index of column user
            df_user_recommend.set_index("user", inplace=True)
            # Merge df contain list user for recommend with list of all user to get id of each msisdn user
            df_user_recommend = pd.merge(df_user_recommend, df_all_user_id, how="left", left_index=True, right_index=True)
            # Save id of user to user list
            list_user = df_user_recommend["index"].values.tolist()
            df_user_recommend.reset_index(inplace=True)
            df_user_recommend.set_index("index", inplace=True)

        # Initialize all pair of (user, item) for recommend:
        all_candidates = None
        # List to save all packages that user subscribed
        user_subscribed_packages = []
        # read information rating of user: user
        packages_user = pd.read_csv(join(self.HomeDir, self.processed_data), index_col="User")
        # Browse all user in list_user
        for user in list_user:
            # Get all packages which user've subscribed and return list of values - Ex: [4, 7, 34]
            packages_subscribed = packages_user.loc[packages_user.index == user]["package"].values
            # Save user with all packages that subscribed
            user_subscribed_packages.append((user, list(packages_subscribed)))
            # Make personal recommendation
            # Get all id of packages without packages've subscribed
            candidates = spark_config.parallelize([m for m in self.packages if m not in packages_subscribed])
            # Map all package with user, user
            candidates = candidates.map(lambda x: (user, x))

            # Concat candidates in all_candidates
            if all_candidates is None:
                all_candidates = spark_config.parallelize(candidates.collect())
            else:
                all_candidates = all_candidates.union(candidates)

        # Predict all rating for all packages with each user -
        # Ex - [Rating(user=2, product=600, rating=0.733132432121), ... ]
        predictions = self.model.predictAll(all_candidates)
        # Sorting all packages with descending rating - Ex: [Rating(user= , item= , rating= ) ]
        predictions_sort = predictions.groupBy(lambda x: x[0])\
                                      .flatMap(lambda g: sorted(g[1], key=lambda x: x[2], reverse=True))
        # Group pair of [(user=?, list_item=[?, ?, ?, ...]), ...]
        predictions_group_user_listitem = predictions_sort.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x+y)
        # Get k recommendation packages for each user
        prediction_top_k = predictions_group_user_listitem.map(lambda x: (x[0], x[1][0:num_item_recommend]))
        # Join prediction and real subscribed packages for each user -
        # Ex: [(user=?, (prediction=[3,4,6,22,45], subscribed=[345,235])), ... ]
        prediction_and_subscribed = prediction_top_k.join(spark_config.parallelize(user_subscribed_packages)).collect()

        # Print all recommendation packages for each user
        for user_data in prediction_and_subscribed:
            # print("user: ", user_data[0])
            # print("\tPackage subscribed: ", [self.packages.get(k) for k in user_data[1][1]])
            # print("\tPackage recommended: ", [self.packages.get(k) for k in user_data[1][0]])

            result_dict[str(df_user_recommend.iloc[user_data[0]]["user"])] = {
                "package_subscribed": [self.packages.get(k) for k in user_data[1][1]],
                "Package recommended": [self.packages.get(k) for k in user_data[1][0]]
            }

        # Clear spark config
        spark_config.stop()

        return result_dict


if __name__ == "__main__":

    ALS_model = ModelALS()
    spark_config = ALS_model.create_spark_environment()
    # Train model
    sc = ALS_model.train(num_recommend_item=5, evaluation_metric='ndcg', sc=spark_config)

    # Load model
    # ALS_model.load_model(spark_config)
    print("load model done")

    # recommend packages for user number 2
    ALS_model.predict(user=2, num_item_recommend=10, spark_config=sc)

    # recommend packages for list user
    prediction_and_subscribed = ALS_model.predict_list(list_user=[84961000014, 84961000021, 84961000023],
                                                       num_item_recommend=5, spark_config=sc)
