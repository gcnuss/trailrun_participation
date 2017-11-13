import cPickle as pickle
import pandas as pd

class RankEval(object):
    '''This class contains functions to calculate the rank as an evaluation metric
    for implicit recommender systems, as defined in the paper "Collaborative Filtering
    for Implicit Feedback Datasets" by Hu, Koren, and Volinsky.

    The basic quality measure is the expected percentile ranking of an attended
    event, with 0% being the top rank and 100% being the bottom rank.

    Lower values of rank_bar are more desirable, as they indicate ranking actually
    attended events closer to the top of the recommendation lists.  For random
    predictions, the expected value of rank_ui is 50% (placing event i in the
    middle of the sorted list).  Thus, rank_bar >= 50% indicates an algorithm
    no better than random.

    The class also has an option to calculate the rank using a simple popularity
    based recommendation list, as this is a reasonable point of comparison for a
    given model

    Parameters:
    preds - a pandas dataframe containing the predictions from a Spark ALS model.
    The df needs to include columns for user, item, actual outcome, and predicted outcome
    user, item, actual, prediction - strings indicating the name of each column in
    the predictions file for use in the function
    '''

    def __init__(self, preds, user, item, actual, prediction):
        self.preds = preds
        self.user = user
        self.item = item
        self.actual = actual
        self.prediction = prediction
        self.rank_bar = None
        self.popularity_rank_bar = None

    def calc_test_rank(self):
        '''calculate the rank for your provided predictions vs. actuals'''

        self.unique_personID = self.preds[self.user].unique()
        numerator_sum = 0
        denominator_sum = 0
        rank_list = []

        if self.preds[self.prediction].max() == 0.0:
            self.rank_bar = 100
            return self.rank_bar
        else:
            for person in sorted(self.unique_personID):
                temp_df = self.preds[self.preds[self.user] == person].copy()
                temp_df['rank_ui'] = temp_df[self.prediction].apply(lambda x: (
                                    1 - (x / temp_df[self.prediction].max()))*100
                                    if temp_df[self.prediction].max()>0 else 0)
                numerator_sum += sum(temp_df[self.actual] * temp_df['rank_ui'])
                denominator_sum += sum(temp_df[self.actual])
            self.rank_bar = numerator_sum / denominator_sum

            return self.rank_bar

    def calc_popular_rank(self):
        '''calculate the associated popularity-based rank for your provided set
        of data'''

        if self.preds[self.prediction].max() == 0.0:
            self.popularity_rank_bar = 100
            return self.popularity_rank_bar
        else:
            group_counts = self.preds[self.preds[self.actual]==1].groupby(
                            by=[self.item]).count()[self.actual].copy()
            group_counts.sort_values(ascending=False, inplace=True)
            popularity = pd.DataFrame({self.item: group_counts.index,
                                        'attendance': group_counts.values})
            popularity['rank_ui'] = list((popularity.index.values / float(
                                        len(group_counts - 1)))*100)
            merged_df = pd.merge(self.preds, popularity, how='left', on=self.item)

            numerator_sum = 0
            denominator_sum = 0
            rank_list = []

            for person in sorted(self.unique_personID):
                temp_df = merged_df[merged_df[self.user] == person].copy()
                numerator_sum += sum(temp_df[self.actual] * temp_df['rank_ui'])
                denominator_sum += sum(temp_df[self.actual])

            self.popularity_rank_bar = numerator_sum / denominator_sum

            return self.popularity_rank_bar
