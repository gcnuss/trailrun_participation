import re
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import defaultdict

import pyspark as ps    # for the pyspark suite
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, StringType, FloatType, DateType, TimestampType
import pyspark.sql.functions as F

spark = ps.sql.SparkSession.builder \
            .master("local[4]") \
            .appName("trailrun") \
            .getOrCreate()

sc = spark.sparkContext   # for the pre-2.0 sparkContext

from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

import cPickle as pickle
import rank_eval

class implicit_als(object):
    '''ADD DOC STRING, parameter and attribute definitions

    data_df should be a pandas dataframe containing all starting data cleaned
    by dataprep.py file'''

    def __init__(self, data_df, split_val=True):
        self.data_df = data_df
        self.split_val = split_val
        self.full_df = None
        self.trainval = None
        self.train = None
        self.validate = None
        self.test = None
        self.train_mat = None
        self.validate_mat = None
        self.test_mat = None
        self.trainval_mat = None
        self.base_model = None

    def prep_full_df(self):
        '''creates spark df with participated column containing both 1's and 0's,
        and adds other data for 0's rows as appropriate; takes the mean value for
        Total fee and Miles2 because we need one value for each event and those
        vary.'''

        unique_personIDs = self.data_df['PersonID'].unique()
        unique_eventIDs = self.data_df['EventID'].unique()

        D = defaultdict(list)

        for person in unique_personIDs:
            events_part = list(self.data_df[self.data_df['PersonID'] == person]['EventID'].values)
            D[person].append(events_part)

        data_w_zeros = []

        for event in unique_eventIDs:
            date = self.data_df[self.data_df['EventID'] == event]['Event_Date'].values[0]
            seriesid = self.data_df[self.data_df['EventID'] == event]['SeriesID'].values[0]
            eventtype = self.data_df[self.data_df['EventID'] == event]['EventTypeID'].values[0]
            totalfeeavg = int(round(self.data_df[self.data_df['EventID'] == event]['Total fee'].values.mean()))
            milesavg = int(round(self.data_df[self.data_df['EventID'] == event]['Miles2'].values.mean()))
            venuezip = self.data_df[self.data_df['EventID'] == event]['Venue_Zip'].values[0]
            for person in unique_personIDs:
                if event in D[person][0]:
                    data_w_zeros.append([person, event, 1, date, seriesid, eventtype, totalfeeavg, milesavg, venuezip])
                else:
                    data_w_zeros.append([person, event, 0, date, seriesid, eventtype, totalfeeavg, milesavg, venuezip])

        self.full_df = pd.DataFrame(data_w_zeros, columns=['PersonID',
                        'EventID', 'Participated', 'Event_Date', 'SeriesID',
                        'EventTypeID', 'Total_Fee_Avg', 'Miles2_Avg', 'Venue_Zip'])

        #return self.spark_full_df

    def train_val_test_split(self, test_prop=0.2, val_prop=0.2):
        '''Performs an 80/20 split for train/test, then splits train again at
        80/20 for train/validate datasets; before splitting, sorts the data based
        on event date so that the most recent events are in the test set and
        oldest are in the training set (good practice for recommenders)'''

        self.trainval = self.full_df.sort_values(by='Event_Date', ascending=True
                        ).iloc[:int(round(len(self.full_df)*(1-test_prop)))]
        self.test = self.full_df.sort_values(by='Event_Date', ascending=True
                        ).iloc[int(round(len(self.full_df)*(1-test_prop))):]
        self.train = self.trainval.sort_values(by='Event_Date', ascending=True
                        ).iloc[:int(round(len(self.trainval)*(1-val_prop)))]
        self.validate = self.trainval.sort_values(by='Event_Date', ascending=True
                        ).iloc[int(round(len(self.trainval)*(1-val_prop))):]

        print('TrainVal Size: {}'.format(round(len(self.trainval.values))))
        print('Train Size: {}'.format(round(len(self.train.values))))
        print('Validation Size: {}'.format(round(len(self.validate.values))))
        print('Test Size: {}'.format(round(len(self.test.values))))

        #return self.train, self.validate, self.test

    def print_train_val_test_info(self, event_param):
        print("participants in train: {}".format(len(self.train['PersonID'].unique())))
        print("participants in validate: {}".format(len(self.validate['PersonID'].unique())))
        print("participants in test: {}".format(len(self.test['PersonID'].unique())))
        print('\n')
        print("participants in both train & validate: {}".format(len(np.intersect1d(
                                        self.train['PersonID'].unique(),
                                        self.validate['PersonID'].unique()))))
        print("participants in both train & test: {}".format(len(np.intersect1d(
                                        self.train['PersonID'].unique(),
                                        self.test['PersonID'].unique()))))
        print('\n')

        print("{} in train: {}".format(event_param, len(self.train[event_param].unique())))
        print("{} in validate: {}".format(event_param, len(self.validate[event_param].unique())))
        print("{} in test: {}".format(event_param, len(self.test[event_param].unique())))
        print('\n')
        print("{} in both train & validate: {}".format(event_param, len(np.intersect1d(
                                        self.train[event_param].unique(),
                                        self.validate[event_param].unique()))))
        print("{} in both train & test: {}".format(event_param, len(np.intersect1d(
                                        self.train[event_param].unique(),
                                        self.test[event_param].unique()))))

    def create_participate_matrices(self, event_param):
        '''For ALS modeling purposes, strips down data to only PersonID, chosen
        item feature (e.g. EventID, SeriesID, Venue_Zip, etc), Participated,
        and Event_Date columns'''

        if self.split_val == True:
            self.train_mat = self.train[["PersonID", event_param, "Participated", "Event_Date"]]
            self.validate_mat = self.validate[["PersonID", event_param, "Participated", "Event_Date"]]
            self.test_mat = self.test[["PersonID", event_param, "Participated", "Event_Date"]]
        else:
            self.trainval_mat = self.trainval[["PersonID", event_param, "Participated", "Event_Date"]]
            self.test_mat = self.test[["PersonID", event_param, "Participated", "Event_Date"]]

        #return self.train_mat, self.validate_mat, self.test_mat

    def fit_ALS(self, rank=100, maxIter=10, regParam=0.1,
                userCol="PersonID", itemCol="EventID", ratingCol="Participated",
                nonnegative=True, implicitPrefs=True, alpha=1.0, coldStartStrategy="drop"):
        '''Fit a single baseline ALS model based on selected parameters for training data'''

        als = ALS(rank=rank, maxIter=maxIter, regParam=regParam, userCol=userCol,
                    itemCol=itemCol, ratingCol=ratingCol, nonnegative=nonnegative,
                    implicitPrefs=implicitPrefs, alpha=alpha,
                    coldStartStrategy=coldStartStrategy)

        if self.split_val == True:
            self.base_model = als.fit(spark.createDataFrame(self.train_mat))
        else:
            self.base_model = als.fit(spark.createDataFrame(self.trainval_mat))

        #return self.base_model

    def predict_ALS(self, model, event_param, scoring="rank"):
        '''run prediction on ALS model using provided scoring method, model, and
        either validation dataset if split_val=True or if False, predictions on
        both trainvalidate and test datasets'''

        if self.split_val == True:
            val_predictions = model.transform(spark.createDataFrame(self.validate_mat))

            pandas_preds = val_predictions.toPandas()
            valid_preds = val_pandas_preds[pd.notnull(val_pandas_preds['prediction'])]['Participated'].count()
            nan_preds = val_pandas_preds[pd.isnull(val_pandas_preds['prediction'])]['Participated'].count()
            print('Predictions includes {} valid values and {} nan values'.format(valid_preds, nan_preds))

            if scoring == "rank":
                import rank_eval
                rank_processing = rank_eval.RankEval(pandas_preds, "PersonID",
                            event_param, "Participated", "prediction")
                val_rank = rank_processing.calc_test_rank()
                pop_rank = rank_processing.calc_popular_rank()

                print("Model Rank = {} and Popular Rank = {}".format(val_rank, pop_rank))

                return pandas_preds, val_rank, pop_rank
            else:
                #initial evaluation using RMSE (root-mean-squared-error)
                #address nan values with mean prediction value (for cold starts) if coldStartStrategy is nan

                evaluator = RegressionEvaluator(metricName=scoring, labelCol="Participated",
                                    predictionCol="prediction")

                error = evaluator.evaluate(val_predictions.na.fill({'prediction':pandas_preds['prediction'].mean()}))

                print("Error of Type {} = {}".format(scoring, str(error)))

                return pandas_preds, error

        else:
            trainval_predictions = model.transform(spark.createDataFrame(self.trainval_mat))
            test_predictions = model.transform(spark.createDataFrame(self.test_mat))

            trainval_pandas_preds = trainval_predictions.toPandas()
            valid_preds = trainval_pandas_preds[pd.notnull(trainval_pandas_preds['prediction'])]['Participated'].count()
            nan_preds = trainval_pandas_preds[pd.isnull(trainval_pandas_preds['prediction'])]['Participated'].count()
            print('Trainval predictions includes {} valid values and {} nan values'.format(valid_preds, nan_preds))

            test_pandas_preds = test_predictions.toPandas()
            valid_preds = test_pandas_preds[pd.notnull(test_pandas_preds['prediction'])]['Participated'].count()
            nan_preds = test_pandas_preds[pd.isnull(test_pandas_preds['prediction'])]['Participated'].count()
            print('Test predictions includes {} valid values and {} nan values'.format(valid_preds, nan_preds))

            if scoring == "rank":
                import rank_eval
                trainval_rank_processing = rank_eval.RankEval(trainval_pandas_preds, "PersonID",
                            event_param, "Participated", "prediction")
                trainval_rank = trainval_rank_processing.calc_test_rank()
                trainval_pop_rank = trainval_rank_processing.calc_popular_rank()

                test_rank_processing = rank_eval.RankEval(test_pandas_preds, "PersonID",
                            event_param, "Participated", "prediction")
                test_rank = test_rank_processing.calc_test_rank()
                test_pop_rank = test_rank_processing.calc_popular_rank()

                print("Trainval Model Rank = {} and Popular Rank = {}".format(trainval_rank, trainval_pop_rank))
                print("Test Model Rank = {} and Popular Rank = {}".format(test_rank, test_pop_rank))

                return trainval_pandas_preds, trainval_rank, trainval_pop_rank, test_pandas_preds, test_rank, test_pop_rank
            else:
                #initial evaluation using RMSE (root-mean-squared-error)
                #address nan values with mean prediction value (for cold starts) if coldStartStrategy is nan

                evaluator = RegressionEvaluator(metricName=scoring, labelCol="Participated",
                                    predictionCol="prediction")

                error = evaluator.evaluate(test_predictions.na.fill({'prediction':test_pandas_preds['prediction'].mean()}))

                print("Error of Type {} = {}".format(scoring, str(error)))

                return predictions, error

if __name__ == '__main__':
    #'''placeholder - add code to save spark df's for future use once run one time'''
    #train_val_test_split(self.spark_full_df, .2, .2)
    #print_train_val_test_info("EventID")
    #create_participate_matrices("EventID")
    pass
