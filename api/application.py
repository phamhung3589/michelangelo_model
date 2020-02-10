from flask import Flask, request, jsonify
from flask_cors import CORS
from recommendation.model.implicit_model import ImplicitFeedback
import numpy as np
import collections


class AppRecommend:

    def	__init__(self):
        # Create app flask
        self.app = Flask(__name__)
        CORS(self.app, resources={r"/*": {"origins": "*"}})
        # Create object implicit
        self.implicit_model = ImplicitFeedback()
        #Create list ndcg and hit_rate
        self.list_ndcgs, self.list_hit_rates = {}, {}
        # Create performance metric
        self.list_product_freq, self.top_5_items, self.item_coverage, self.all_subscribed_package_freq = \
                np.array([]), [], [], []
        # Create sparse matrix for recommendation
        self.offer_purchased_sparse = self.implicit_model.import_data()

    def training(self, alpha, factor, regularization, iteration, num_items):
        """
        Training model with parameter and save model to database
        :param alpha:               alpha = 28
        :param factor:              factor = 8
        :param regularization:      regularization = 1.05
        :param iteration:           iteration = 200
        :param num_items:           num item = 5 - using to recommend num item for each user
        :return:                    ndcg and hit_rate
        """

        # Split raw rating data to training set and test set
        training_set, test_set = self.implicit_model.train_test_split(self.offer_purchased_sparse)
        # Create list of unique user to evaluate
        list_unique_user, true_labels = test_set
        # Training algorithm
        self.implicit_model.train(training_set, alpha=alpha, factors=factor,
                                  regularization=regularization, iterations=iteration)
        # Saving user_vec and item_vec to local
        # implicit_feedback.save_model("user_vecs.npy", "item_vecs.npy")

        # Predict list of user
        list_product_id = self.implicit_model.predict_list(user_list=list_unique_user,
                                                             num_items=num_items, print_item="false")

        # Evaluate model
        ndcg = self.implicit_model.evaluate(true_labels=true_labels, predict_labels=list_product_id, type_metric="ndcg")
        hr = self.implicit_model.evaluate(true_labels=true_labels, predict_labels=list_product_id, type_metric="hit_rate")

        return ndcg, hr

    def get_performance_metric(self):
        """
        Calculate performance metric
        :return: list_product_freq, top_5_items, item_coverage
        """

        # only get all items that all user subscribed
        all_package_subscribed = np.argwhere(self.offer_purchased_sparse != 0)[:, 1]
        # Count times of appear of each item and convert list of tuple to dict
        all_package_subscribed = dict(collections.Counter(all_package_subscribed))
        # Sort all frequency of all items
        all_package_subscribed = np.array(sorted(all_package_subscribed.items(), key=lambda kv: kv[1], reverse=True))
        # Map item id with name of it
        name_packages = self.implicit_model.map_item_name(all_package_subscribed[:, 0])
        # Get frequency from numpy array
        frequency = list(all_package_subscribed[:, 1])
        # Map all items name with frequency in 1 list
        self.all_subscribed_package_freq = [[name_packages[i], int(frequency[i])] for i in range(len(frequency))]

        # Get list of all product
        list_product_id = self.implicit_model.\
            predict_list(user_list=range(self.implicit_model.training_set.shape[0]), num_items=5, print_item="false")

        # Calculate coverage of all item in list recommendation
        self.list_product_freq, self.top_5_items, self.item_coverage = self.implicit_model.coverage_item(list_product_id)

    def load_saved_model(self, list_user):
        """
        load saved model from database and return metrics of performance for UI
        :param list_user: list of user for recommend packages
        :return: metrics - list product frequency, item coverage, dataframe of user - packages, top 5 best items
        """

        # Split raw rating data to training set and test set
        self.implicit_model.train_test_split(self.offer_purchased_sparse)
        # Loading user_vec and item_vec from local file
        self.implicit_model.load_model("user_vecs.npy", "item_vecs.npy")
        # using loaded model to recommend items for list user
        list_product_id = self.implicit_model.predict_list(user_list=list_user, num_items=5, print_item="false")
        # Merge user_id, package_id with MSISDN and name of package
        df_user_item = self.implicit_model.convert_msisdn(list_product_id, list_user=list_user)

        return df_user_item

    def route_api(self):
        """
        Define all route for all apis
        :return: api for communicate between model and UI
        """

        @self.app.route("/implicit/train", methods=["POST", "GET"])
        def train_model():
            """
            api for training model - using parameter from UI to set
            :return: save_model to database
            """
            evaluation_metric = None
            if request == "POST":
                # Get parameter through api with json format - return dict
                parameter_ui = request.json
                # Initial list of number of recommendation items
                list_num_item = [3, 5, 7 ,9, 11, 13]

                for k in list_num_item:
                    print("training model with k = {:d}".format(k))
                    # Run algorithm with difference num of items
                    if parameter_ui.get("alpha") is None:
                        evaluation_metric = self.training(28, 8, 1.05, 20, k)
                    else:
                        evaluation_metric = self.training(parameter_ui.get("alpha"),
                                                          parameter_ui.get("factor"),
                                                          parameter_ui.get("regularization"),
                                                          parameter_ui.get("iteration"), k)

                    # Add data to list ndcg
                    self.list_ndcgs[k] = evaluation_metric[0]
                    # Add data to list hit rate
                    self.list_hit_rates[k] = evaluation_metric[1]

                # Get all performance metric
                self.get_performance_metric()
                evaluation_metric = {"ndcg": self.list_ndcgs, "hit_rate": self.list_hit_rates}
                print("Calculate performance metric success")

            return jsonify({"top_5_items": self.top_5_items,
                            "item_coverage": self.item_coverage,
                            "list_product_freq": self.list_product_freq,
                            "all_subscribed_package_freq": self.all_subscribed_package_freq,
                            "evaluation_metric": evaluation_metric})

        @self.app.route("/implicit/metrics", methods=["POST"])
        def metrics():
            """
            Load model from database and calculating evaluation metric
            :return: evaluation metrics for UI to display
            """

            # Initial evaluation metrics
            df_user_item = None
            # Catch metrics for post
            if request.method == "POST":
                # Get parameter_ui from json format and return dict format
                list_user_json = request.json
                # Get parameter_ui from dict
                list_user = list_user_json.get("list_user")
                # Using parameter from parameter_ui and using it to loading model from database
                df_user_item = self.load_saved_model(list_user=list_user)

            return jsonify({"table_recommendation_item": df_user_item.to_dict()})


if __name__ == "__main__":
    api = AppRecommend()
    api.route_api()
    api.app.run(host="0.0.0.0", port=1234, debug=True)
