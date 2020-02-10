import pandas as pd
import numpy as np
import os
from pyspark import SparkConf, SparkContext
from recommendation.model.constants import Constants
from sklearn.preprocessing import MinMaxScaler


class Standardize:

    delimiter = ","

    def __init__(self):

        # All offer
        self.all_offer = os.path.abspath(os.path.dirname(__file__) + "/../data/all_offers.csv")
        # File containing standardize offer
        self.standardize_offer = os.path.abspath(os.path.dirname(__file__) + "/../data/standardized_offer.csv")
        # Filename containing data which does not standardize subscribed
        self.file_not_standardize_subscribed = os.path.abspath(os.path.dirname(__file__) + "/../data/offer_purchased.csv")
        self.file_standardize_subscriber = os.path.abspath(os.path.dirname(__file__) + "/../data/data.csv")
        # Output file to save all packages
        self.outputfile_package = os.path.abspath(os.path.dirname(__file__) + "/../data/all_package.csv")
        # Output file to save all users
        self.outputfile_user = os.path.abspath(os.path.dirname(__file__) + "/../data/all_user.csv")
        # Output of processed_data file
        self.processed_data = os.path.abspath(os.path.dirname(__file__) + "/../data/processed_data.csv")
        # Output of processed data for testing algorithm
        self.processed_data_test = os.path.abspath(os.path.dirname(__file__) + "/../data/processed_data_test.csv")
        # Output of processed data for training algorithm
        self.processed_data_training = os.path.abspath(os.path.dirname(__file__) + "/../data/processed_data_training.csv")
        # Output of processed data for test negative
        self.processed_data_test_negative = os.path.abspath(os.path.dirname(__file__) + "/../data/processed_data_test_negative.csv")

    @staticmethod
    def create_spark_environment():
        """
        setup environment for pyspark to run on cluster or local
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
    def eliminate_offer(user_offers):
        """
        Eliminate all offers that don't have information
        :param user_offers:     list of (user, list offer)
        :return:                (user, list_filtered_offer)
        """

        # Get list of offer to eliminate
        offers_eliminate = Constants.offer_filter
        # Get all offers
        all_offers = user_offers[1]
        # Create list_filtered_offer
        offer_filtered = []
        # Loop over all offer
        for offer in all_offers:
            boolean = False
            # If offer in list to eliminate => boolean = True
            for offer_eliminate in offers_eliminate:
                if offer.startswith(offer_eliminate):
                    boolean = True
                    break
            # Using boolean to save data to list_filtered offer or not
            if boolean is False:
                offer_filtered.append(offer)

        # Return list of user and filtered offer
        user_offers_filter = (user_offers[0], offer_filtered)

        return user_offers_filter

    def standardize_subscriber(self, spark_config):
        """
        Standardize subscriber from raw data
        :param spark_config:    environment for pyspark to run on  cluster or local
        :return:                Clean data which is group all standardize user
        """

        # Using spark to load data to standardize subscriber
        data_not_standardize = spark_config.textFile(self.file_not_standardize_subscribed)
        # Map data into key-values pair (user, list(packages)) - Ex: ( (8349587967, [43,67,234,1,54,34], (), ...) )
        data_user_list_package = data_not_standardize.map(lambda x: x.split(",")).map(lambda x: (x[0], x[1:]))
        # Replace "8416" to "843" in key MSISDN of all user. - parameter 1 indicate only first instances is replaced
        data_standardized = data_user_list_package.\
            map(lambda x: (x[0].replace("8416", "843", 1), x[1]) if x[0].startswith("8416") else (x[0]. x[1])).repartition(20)
        # Remove all offers start with string: "MOBILE"
        data_standardized = data_standardized.map(lambda x: (x[0], [ele
                                                             .replace("MOBILE-", "")
                                                             .replace("MOBILE- ", "")
                                                             .replace("MOBILE_", "") for ele in x[1]])).repartition(20)
        # Delete all offer not use
        data_standardized = data_standardized.map(lambda x: self.eliminate_offer(x))
        # Remove all user with no offer
        data_standardized = data_standardized.filter(lambda x: len(x[1]) != 0)
        # Group all user in 1 row and concat all packages in 1 list
        data_standardized_reduce_key = data_standardized.reduceByKey(lambda x, y: x + y).repartition(50)
        # Flatmap key to each value
        data_split_row = data_standardized_reduce_key.flatMapValues(lambda x: x)
        # Collect data in local
        data_collect = data_split_row.collect()
        # Save to dataframe - set columns name
        df_data_collect = pd.DataFrame(data=data_collect, columns=["MSISDN", "package"])
        # Save to csv file
        df_data_collect.to_csv(self.file_standardize_subscriber, index=False)

    def standardize_package(self):
        """
        :return:    dataframe of all packages
        """

        # Read data from csv file
        data_subscriber = pd.read_csv(self.file_standardize_subscriber)
        # Get package and user with unique values (unique MSISDN) and sorting name of packages
        df_unique_package = data_subscriber["package"].drop_duplicates().sort_values()
        df_unique_users = data_subscriber["MSISDN"].drop_duplicates().sort_values()
        # Set column for list package (list user)
        df_unique_package = pd.DataFrame(df_unique_package, columns=["package"])
        df_unique_users = pd.DataFrame(df_unique_users.values, columns=["user"])
        # Save to csv file
        df_unique_package.to_csv(self.outputfile_package, index=None)
        df_unique_users.to_csv(self.outputfile_user, index=None)

        return df_unique_package

    def merge_package_with_info(self):
        """
        Merge all package from all user with information of it into 1 file
        :return: file has been merged
        """

        # Read data from file containing all package and file containing packages that user in data subscribed
        package_raw = pd.read_csv(self.outputfile_package, index_col="package")
        package_all = pd.read_csv(self.all_offer, index_col="package")
        # Merge package with info
        package_raw = pd.merge(package_raw, package_all, how="left", left_index=True, right_index=True)
        # Reset index 2 times
        package_raw.reset_index(inplace=True)
        package_raw.reset_index(inplace=True)
        # Set index col to id col
        package_raw.rename(columns={"index": "id"}, inplace=True)
        # Save to csv file
        package_raw.to_csv(self.standardize_offer, index=False)

    def normalize_data(self):
        """
        Normalize raw data with csv format: user, package, rating
        :return: save to normalized file and return dataframe of processed data
        """

        # Read data from subscriber user
        data_subscriber = pd.read_csv(self.file_standardize_subscriber)
        # Read all user and package file - save to dataframe pandas
        df_user = pd.read_csv(self.outputfile_user)
        df_user.reset_index(inplace=True)
        # Set index to user to merge with data_rating with same index
        df_user.set_index("user", inplace=True)
        df_user.rename(columns={"index": "user_id"}, inplace=True)
        df_package = pd.read_csv(self.outputfile_package).reset_index()
        # Set index to package to merge with data_rating with same index
        df_package.set_index("package", inplace=True)
        df_package.rename(columns={"index": "package_id"}, inplace=True)

        # Group on all columns and call size the index indicates the duplicate values
        data_rating = data_subscriber.groupby(data_subscriber.columns.tolist(), as_index=False).size().reset_index()
        # Rename column 0 to rating
        data_rating.rename(columns={"MSISDN": "user", 0: "ratings"}, inplace=True)
        # Convert rating data to int
        data_rating["ratings"] = data_rating["ratings"].astype(int)
        # Merge data_rating with df_user to get id of each user
        df_processed_data = pd.merge(data_rating.set_index("user"),
                                     df_user, how="left", left_index=True, right_index=True).reset_index()
        # Merge data rating with df_package to get id of each item
        df_processed_data = pd.merge(df_processed_data.set_index("package"),
                                     df_package, how="left", left_index=True, right_index=True).reset_index()
        # Choose 3 columns (user_id, package_id, ratings)
        df_processed_data = df_processed_data[["user_id", "package_id", "ratings"]]
        # Rename columns of df_processed data to user - package - ratings
        df_processed_data.rename(columns={"user_id": "User", "package_id": "package"}, inplace=True)
        # Sort values for col "User"
        df_processed_data = df_processed_data.sort_values("User")
        # Save to csv file
        df_processed_data.to_csv(self.processed_data, index=None)

        return df_processed_data

    def normalize_rating_with_price(self):
        """
        Using minmaxScaler to normalize rating in range(0, 1)
        :return:  Save normalizing ratings into csv files
        """

        # Function of sklearn to normalize
        scaler = MinMaxScaler()
        # Save data into dataframe
        df_data = pd.read_csv(self.processed_data, index_col="package")
        df_offer = pd.read_csv(self.standardize_offer)[["id", "Price"]]
        # Rename column id to package - same as table processed data
        df_offer.rename(columns={"id": "package"}, inplace=True)
        # Set index of standardized offer to package to merge between 2 table
        df_offer.set_index("package", inplace=True)
        # Merge price of table 2 into table processed data - set index to column User
        df_data = pd.merge(df_data, df_offer, how="left", left_index=True, right_index=True).reset_index()
        df_data.set_index("User", inplace=True)
        # Sorting index with index-col user
        df_data.sort_index(inplace=True)
        # Create column rating = col(rating) * col(Price)
        df_data["rating"] = df_data["ratings"] * df_data["Price"]
        # using minmaxScaler to normalizing col-rating
        df_data["rating"] = scaler.fit_transform(df_data["rating"].values.reshape(-1, 1))
        # Only choose 2 col: package and rating
        df_data = df_data[["package", "rating"]]
        # Rename rating to ratings - same as table processed data
        df_data.reanme(columns={"rating": "ratings"}, inplace=True)
        # Reset index with column "user"
        df_data.reset_index(inplace=True)
        # Save to csv file
        df_data.to_csv("../data/processed_data_new.csv", index=True)

    def split_data(self):
        """
        split data into 2 groups of all user with 1 package and the rest
        :return: Save to file test, training and negative test file
        """

        # Save processed data into pandas dataframe
        df_processed = pd.read_csv(self.processed_data)
        # Initial training data
        df_processed_training = df_processed.copy()
        print("size of rating data: ", len(df_processed))

        # Get all package only have one and tow rating
        df_processed_package = df_processed.copy()
        # Create column freq_package to counting number of times package X has subscribed
        df_processed_package["freq_package"] = df_processed_package.groupby("package")["package"].transform("count")

        # Step 1: from raw data get all user with number of ratings > 3
        # (reason: get 1 rating from these users for testing )
        # Step 2: Keep pnly 1 rating for each user
        # Step 3: join matrix of user test with matrix of counting number of package've subscribed
        # Step 4: Remove all row with number of times of package = 1 (training set must have all ratings of all user
        # and package)
        # Step 5: Only get column need and save to csv file

        # Group data by user and counting number of times each user appear in dataset
        # (counting user X subscribe how many packages)
        df_processed['freq'] = df_processed.groupby("User")["User"].transform("count")
        # Counting number of package subscribed larger than 3
        df_processed = df_processed[df_processed["freq"] > 3]
        # Set index
        df_processed.set_index("User", inplace=True)
        # Only keep first rating of each user
        df_processed = df_processed[~df_processed.index.duplicated(keep='first')]
        # Reset index from column "User"
        df_processed.reset_index(inplace=True)
        # Set index of 2 matrix with 2 columns [User, package] and join them with index of matrix df_processed
        df_processed = df_processed.set_index(["User", "package"]).\
            join(df_processed_package.set_index(["User", "package"]), how="left", rsuffix='_b')
        # Remove all packages've subscribed 1 times
        df_processed = df_processed[df_processed["freq_package"] > 1]
        # Drop all columns unnecessary
        df_processed.drop(columns=["ratings_b", "freq_package", "freq"], inplace=True)
        # Save index from test data to remove all test data in training set
        index_remove = df_processed.index
        # Reset index
        df_processed.reset_index(inplace=True)
        # Save to processed_data_test_file
        df_processed.to_csv(self.processed_data_test, index=None)
        # Drop all data from test set
        df_processed_training = df_processed_training.set_index(["User", "package"]).drop(index=index_remove)
        # Reset index
        df_processed_training.reset_index(inplace=True)
        # Save training set to csv file
        df_processed_training.to_csv(self.processed_data_training, index=None)

        print("Size of training data ", len(df_processed_training))
        print("Size of test data ", len(df_processed))

        # Initializing test negative file
        print("saving packages for user to predict: ")
        # Get list id of all packages
        all_items = pd.read_csv(self.outputfile_package)["package"].index.values.tolist()
        # Initial test negatives to save
        test_negatives = []
        # Get all pair of (user, package) in test data
        user_item = df_processed[["User", "package"]].values
        # Count variable
        count = 0
        # Loop over all pair of (user, package) in test data to append negative package
        for user, item in user_item:
            # COunt number of row
            count += 1
            if count % 10000 == 0:
                print("Reading number of row: ", count)
            # Initial list of negative packages user X haven't been subscribed
            user_negative = list([])
            # Append pair of (user, package) - same with test dataset
            user_negative.append(np.array([user, item]))
            # Create copy data of all_items
            all_items_temp = all_items.copy()
            # remove item that user subscribed
            all_items_temp.remove(item)
            for package in all_items_temp:
                # Add of negative package to list user negative
                user_negative.append(package)
            # Save user's data of negative package in list of test negative
            test_negatives.append(user_negative)

        print("Saving to dataframe")
        # Create Dataframe from list test negative of all users
        df_test_negative = pd.DataFrame(data=test_negatives)
        # Save to csv file
        df_test_negative.to_csv(self.processed_data_test_negative, index=None, header=None)

        print("Data have saved")


if __name__ == "__main__":

    # create object standardize
    standardize = Standardize()

    # Standardize subscriber - remove all packages not need
    spark_conf = standardize.create_spark_environment()
    standardize.standardize_subscriber(spark_config=spark_conf)

    # Save raw data to 2 file containing all packages and all users
    standardize.standardize_package()

    # Merge all packages file with info
    standardize.merge_package_with_info()

    # Normalize data using recommendation algorithm
    standardize.normalize_data()

    # Split data into train and test split
    standardize.split_data()
