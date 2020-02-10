import os
import sys
import numpy as np
import pandas as pd
from os.path import join
from time import time
from keras.layers import Embedding, Input, Dense, Flatten, multiply, Concatenate
from keras.models import Model, load_model, model_from_json
from recommendation.preprocessing.preprocessing_ncf import Dataset
from recommendation.evaluation.evaluation_model import Evaluate
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "../"))

# Setup environment to run with all necessary libs
os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/haitm121/anaconda3/bin/python"
os.environ["PYSPARK_PYTHON"] = "/home/haitm121/anaconda3/bin/python"

class NeuralCF:

    def __init__(self, num_items):
        # Number of item (all packages in database)
        self.num_items = num_items
        # Folder containing all data
        self.HomeDir = "../data/"
        # File containing all packages
        self.all_packages = "all_package.csv"
        # Create instance variable packages to save all packages in database with name and id of it
        self.packages = pd.read_csv(join(self.HomeDir, self.all_packages)).to_dict().get("package")

    @staticmethod
    def get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf=0):
        assert len(layers) == len(reg_layers)
        # Number of layers in the MLP
        num_layer = len(layers)

        # Input variables
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user', input_length=1)
        MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item', input_length=1)

        MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] / 2, name="mlp_embedding_user",
                                       input_length=1)
        MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] / 2, name='mlp_embedding_item',
                                       input_length=1)

        # MF part
        mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
        mf_vector = multiply([mf_user_latent, mf_item_latent])  # element-wise multiply

        # MLP part
        mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
        mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
        mlp_vector = Concatenate(axis=-1)([mlp_user_latent, mlp_item_latent])

        for idx in range(1, num_layer):
            layer = Dense(layers[idx], activation='relu', name="layer%d" % idx)
            mlp_vector = layer(mlp_vector)

        # Concatenate MF and MLP parts
        predict_vector = Concatenate(axis=-1)([mf_vector, mlp_vector])

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', name="prediction")(predict_vector)

        model = Model(input=[user_input, item_input], output=prediction)

        return model

    @staticmethod
    def get_training_set(train, num_negative):
        """
        Add negative instances into training set (training set only have values before )
        :param train:           training set
        :param num_negative:    number of negative values add for each user
        :return:                user_input, item_input, labels
        """

        # Initiate output
        user_input, item_input, labels = [], [], []
        # Get number of items
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # add positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # add negative instances
            for t in range(num_negative):
                # Random integer number in range(0, num_items)
                j = np.random.randint(num_items)

                # Check condition whether (u, j) in training set or not
                while (u, j) in train.keys():
                    # reRandom j
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)

        return user_input, item_input, labels

    def train(self, training_set, batch_size, epochs):
        """
        Run model neural collaborative filtering
        :param training_set:
        :param batch_size:
        :param epochs:
        :return:            best model
        """

        # return number of users and items
        num_users, num_items = training_set.shape
        # Add negative values for pair (u, i)
        num_negatives = 2
        user_input, item_input, labels = self.get_training_set(training_set, num_negatives)
        print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d" %
              (time()-t1, num_users, self.num_items, training_set.nnz))

        # Build model
        model = self.get_model(num_users, self.num_items, 8, [64, 32, 16, 8], [0, 0, 0, 0], 0)

        # Summary of model
        # model.summary()

        # Compiling model with function adam optimizer
        model.compile(optimizer="adam", loss="binary_crossentropy")

        # Fitting model with real data of input of (user_input, item_input) and output is labels
        model.fit([np.array(user_input), np.array(item_input)], np.array(labels), batch_size=batch_size, epochs=epochs)
        
        return model 
    
    @staticmethod
    def save_model(trained_model):
        """
        Save model to database
        :param trained_model:   model for saving
        :return:                path to save model
        """

        # Save model to h5 file
        trained_model.save("model_recommendation_neumf.h5")

    @staticmethod
    def load_model_from_db():
        """
        Load model to database
        :return:    model
        """

        # Load odel from h5 file
        model = load_model("model_recommendation_neumf.h5")

        print("load model done")
        return model

    @staticmethod
    def evaluate_model(trained_model, test_rating, test_negative, num_thread):
        """
        Evaluate model using difference metrics: hit_rate, ndcg, rmse
        :param trained_model:   Model to predict
        :param test_rating:     contained pair of user - item (user subscribed) for testing
        :param test_negative:   corresponding to test_rating and all item that user X not subscribed
        :param num_thread:      Thread for running
        :return:                Mean of all evaluation metrics over all test data
        """

        # Initiate object evaluate
        evaluate = Evaluate()
        # Use all hit_rate and ndcg to evaluate model
        (hits, ndcgs) = evaluate.evaluate_model(model=trained_model, test_ratings=test_rating[0],
                                                test_negatives=test_negative, k=10, num_thread=num_thread)
        # Compute mean of hit_rate and ndcg over all testing data
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

        return hr, ndcg

    @staticmethod
    def score(trained_model, user_input, item_input, true_labels):
        """
        Using model to predict label for input X and compute loss between predict_label and true_label
        :param trained_model:   model using to predict
        :param user_input:      user to predict
        :param item_input:      list of input item that user subscribed
        :param true_labels:     true label of pair (user, item)
        :return:                score between predict_label and true_label
        """

        score = trained_model.evaluate(x=[user_input, item_input], y=true_labels, batch_size=32)

        return score

    def predict(self, user, trained_model):
        """
        Using trained model to recommend n items for that user
        :param user:            User to recommend packages
        :param trained_model:   Model have been trained
        :return:                List of k packages recommending for that user
        """

        # Load data from rating file into pandas dataframe
        df = pd.read_csv("../data/processed_data.csv")
        # get all packages that user ? has subscribed
        user_test_package = df[df["User"] == user]["package"].values

        # Initiate pair of user_test and item)test to predict values of each package to rating
        user_test, item_test = [], []

        for i in range(self.num_items):
            # Check condition if package in these that user ? has subscribed
            if i not in user_test_package:
                # Append to list of user and item
                user_test.append(user)
                item_test.append(i)

        # Using trained model to predict pair of (user_test, item_test)
        predicts = trained_model.predict(x=[user_test, item_test])
        # Save all rating prediction to pandas dataframe with index of item_index
        df = pd.DataFrame(data=predicts, index=item_test)
        # Sorting dataframe with column predict in descending
        df.sort_values(by=[0], axis=0, inplace=True, ascending=False)
        # Get 10 highest values of package
        idx = df[0:10].index
        # Map index of package into name of that package - save to pandas dataframe
        packages = pd.read_csv("../data/all_package.csv")

        # Print all packages that user've subscribed
        print("User {:d} has subscribed these packages: ".format(user))
        for i in range(len(user_test_package)):
            print("{:d} : {:s}".format((i+1), packages.iloc[user_test_package[i]]["package"]))

        # Print all package that recommend for that user
        print("Recommend for user number 2")
        for i in range(len(idx)):
            print("%2d: %s" % (i+1, packages.iloc[idx[i]]["package"]))

    def predict_list(self, list_user, num_item_recommend, trained_model):
        """
        Recommend packages for list of users
        :param list_user:               list of user to recommend packages
        :param num_item_recommend:      item recommend for each user
        :param trained_model:           Model have been trained
        :return:                        List of k packages recommending for all users
        """

        # Load data from rating file into pandas dataframe
        df_processed_data = pd.read_csv("../data/processed_data.csv")
        # initiate packages that user X've been subscribed
        packages_subscribed = []
        # Initiate pair of user_test and item)test to predict values of each package to rating
        user_test, item_test = [], []
        # Browse throw all user in list test user
        for user in list_user:
            # get all packages that user ? has subscribed
            user_test_package = df_processed_data[df_processed_data["User"] == user]["package"].values
            # Add all packages of user to list of packages subscribed of all users
            packages_subscribed.append(user_test_package)
            # Browse throw all items
            for i in range(self.num_items):
                # Check condition if package in these that user ? has subscribed
                if i not in user_test_package:
                    # Append to list of user and item
                    user_test.append(user)
                    item_test.append(i)

        # Using trained model to predict pair of (user_test, item_test)
        predicts = trained_model.predict(x=[user_test, item_test], batch_size=128)
        # Save all prediction with index of pair [user, item]
        df_predict = pd.DataFrame(data=predicts, index=[user_test, item_test], columns=["prediction"])
        # Sorting prediction values with each user
        # Step 1: Sort all prediction values and reset index
        # Step 2: Set_index to user and sort index=user
        df_predict = df_predict.sort_values(by="prediction", ascending=False).reset_index()
        # Change columns of dataframe
        df_predict.columns = ["user", "item", "prediction"]
        df_predict = df_predict.set_index(["user"]).sort_index(kind="mergesort")
        # With each user get k highest prediction recommendation item
        df_predict = df_predict.reset_index().groupby("user").head(num_item_recommend).set_index("user")
        # Group all item with each user
        df_predict = df_predict.reset_index().groupby("user").apply(lambda x: x["item"].tolist()).reset_index()
        # Change column of dataframe
        df_predict.columns = ["user", "packages_recommend"]
        # Set index to user for dataframe
        df_predict.set_index("user", inplace=True)
        # Add column for packages subscribed of each user - Ex:
        #           Item_recommend      Item_subscribed
        # User
        # 0            [2, 1]               [1, 2]
        # 1            [3, 5]               [3, 4]
        # 2            [6, 7]               [5, 6]
        df_predict["pakcages_subscribed"] = packages_subscribed
        print(df_predict)
        # Print name of packages for easy to follow
        for user in df_predict.index:
            print("user: ", user)
            print("\tPackage subscribed: ", [self.packages[k] for k in df_predict.loc[user]["packages_subscribed"]])
            print("\tPackage recommended: ", [self.packages[k] for k in df_predict.loc[user]["packages_recommend"]])

        return df_predict


if __name__ == "__main__":
    t1 = time()

    # Initiate object dataset to working with data
    data = Dataset("../data")
    # Get training set, test_set and test for negative packages for each user
    training_set, test_rating, test_negative = data.training_set, data.test_rating, data.test_negative
    num_of_items = training_set.shape[1]

    # initiate model object
    ncf = NeuralCF(num_of_items)
    # train model
    model = ncf.train(training_set=training_set, batch_size=512, epochs=2)

    # Save model
    ncf.save_model(trained_model=model)

    # Load model
    # model_ncf = ncf.load_model_from_db()

    # Evaluate model
    hr, ndcg = ncf.evaluate_model(model, test_rating=test_rating, test_negative=test_negative, num_thread=1)
    print("hr = {:.5f}, ndcg = {:.5f}".format(hr, ndcg))
    print("Time of calculating metrics: ", (time()-t1))

    # Score the model
    input_user, input_item, real_labels = ncf.get_training_set(training_set, num_negative=2)
    score = ncf.score(model, user_input=input_user, item_input=input_item, true_labels=real_labels)
    print("score = ", score)

    # Predict user
    ncf.predict_list(list_user=[2, 3, 4], num_item_recommend=5, trained_model=model)
