import math
import heapq
import multiprocessing
import numpy as np

# Global variables that are shared across processed
_model = None
_test_ratings = None
_test_negatives = None
_k = None


class Evaluate:

    def evaluate_model(self, model, test_ratings, test_negatives, k, num_thread):
        """
        Evaluate the performance (hit_ratio, ndcg) of top-k recommendation
        :param model:               Model using to predict test_user
        :param test_ratings:        matrix of rating of user test
        :param test_negatives:      all offers that user haven't subscribed
        :param k:                   top k item for recommend
        :param num_thread:          No. thread for parallelize
        :return:                    Score of each testing
        """

        global _model
        global _test_ratings
        global _test_negatives
        global _k

        _model = model
        _test_ratings = test_ratings
        _test_negatives = test_negatives
        _k = k

        # Initial list of hit_rate and ndcgs for all user
        hits, ndcgs = [], []
        # Case: multithread
        if num_thread > 1:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            print("create map function to evaluate", len(_test_ratings), multiprocessing.cpu_count())
            # map function evaluate_one_rating for each index of test set to multiprocessing
            res = pool.map(self.evaluate_one_rating, range(len(_test_ratings)))
            # res = pool.map(self.f, [1,2,3,4,5])
            print("map done", res)
            # Get hit_rate and ndcg from res
            hits = [r[0] for r in res]
            ndcgs = [r[1] for r in res]

            # pool.close()
            # pool.join()

            return hits, ndcgs

        # Case: single thread
        for idx in range(len(_test_ratings)):
            if idx % 1000 == 0:
                print("Row: ", idx)
            # Calculate each hit_rate and ndcg metric for each user, item
            (hr, ndcg) = self.evaluate_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)

        return hits, ndcgs

    @staticmethod
    def f(x):
        return x*x

    def evaluate_one_rating(self, idx):
        """
        Calculate all hit_rate and ndcg metric for all user in test set
        :param idx: index of user need to calculate hit_rate and ndcg metric
        :return: list of (hit_rae, ndcgs) for all pair (user, item)
        """

        # Get rating and item for all user
        rating = _test_ratings[idx]
        items = _test_negatives[idx]
        user = rating[0]
        item_subscribed = rating[1]
        # add item user've subscribed into list of items
        items.append(item_subscribed)
        # Get prediction scores
        map_item_score = {}
        # Create list of (user, user, user, ... ) with len = items
        users = np.full(len(items), user, dtype="int32")
        # Run model to predict value for all pair of user, item
        predictions = _model.predict([users, np.array(items)])

        # Add predict into dictionary (item: predict)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]

        # Free list items
        items.pop()

        # Evaluate top k recommendation item in rank list
        ranklist = heapq.nlargest(_k, map_item_score, key=map_item_score.get)
        hr = self.get_hit_ratio(ranklist, item_subscribed)
        ndcg = self.get_ndcg(ranklist, item_subscribed)

        return hr, ndcg

    @staticmethod
    def get_ndcg(rank_list, item_subscribed):
        """
        Evaluate ndcg of top-k recommendation
        formula: item i appear in position k of rank list => ndcg = log(2) / log(k+1)
        :param rank_list:           list of ranking packages
        :param item_subscribed:     item that user've subscribed
        :return:                    ndcg for that item've subscribed
        """

        ndcg = 0
        for i in range(len(rank_list)):
            item = rank_list[i]
            if item == item_subscribed:
                ndcg = math.log(2) / math.log(i+2)

        return ndcg

    @staticmethod
    def get_hit_ratio(rank_list, item_subscribed):
        """
        Evaluate ndcg of top-k recommendation
        formula: item i appear at any position of rank list => hit_rate = 1
        :param rank_list:           list of ranking packages
        :param item_subscribed:     item that user've subscribed
        :return:                    hit_rate for that item've subscribed
        """

        hr = 0
        for item in rank_list:
            if item == item_subscribed:
                hr = 1

        return hr


if __name__ == "__main__":
    evaluate = Evaluate()
