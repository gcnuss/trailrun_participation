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

    def __init__(self, data_df):
        self.data_df = data_df
        self.spark_onesdata_df = None
        self.spark_full_df = None
        self.train = None
        self.validate = None
        self.test = None
        self.train_mat = None
        self.validate_mat = None
        self.test_mat = None
        self.base_model = None
        self.tvs_model = None
        self.tvs_bestmodel = None
        self.tvs_trainpreds = None
        self.tvs_trainerror = None
        self.tvs_valpreds = None
        self.tvs_valerror = None

    def prep_spark_ones_df(self):
        '''creates spark df with a column of 1's added for participated labels'''
        participated_list = np.full((len(self.data_df), 1), 1)
        self.data_df.insert(2, 'Participated', participated_list)
        self.spark_onesdata_df = spark.createDataFrame(self.data_df)

        return self.spark_onesdata_df

    def prep_spark_full_df(self):
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

        self.spark_full_df = spark.createDataFrame(pd.DataFrame(data_w_zeros, columns=['PersonID',
                        'EventID', 'Participated', 'Event_Date', 'SeriesID',
                        'EventTypeID', 'Total_Fee_Avg', 'Miles2_Avg', 'Venue_Zip']))

        #return self.spark_full_df

    def train_val_test_split(self, test_prop=0.2, val_prop=0.2):
        '''Performs an 80/20 split for train/test, then splits train again at
        80/20 for train/validate datasets; before splitting, sorts the data based
        on event date so that the most recent events are in the test set and
        oldest are in the training set (good practice for recommenders)'''

        trainval = self.spark_full_df.sort('Event_Date', ascending=True).limit(int(round(
                                    self.spark_full_df.count()*(1-test_prop))))
        self.test = self.spark_full_df.sort('Event_Date', ascending=False).limit(int(round(
                                    self.spark_full_df.count()*test_prop)))
        self.train = trainval.sort('Event_Date', ascending=True).limit(int(round(
                                    trainval.count()*(1-val_prop))))
        self.validate = trainval.sort('Event_Date', ascending=False).limit(int(round(
                                    trainval.count()*(val_prop))))

        print('Train Size: {}'.format(round(self.train.count())))
        print('Validation Size: {}'.format(round(self.validate.count())))
        print('Test Size: {}'.format(round(self.test.count())))

        #return self.train, self.validate, self.test

    def print_train_val_test_info(self, event_param):
        print("participants in train: {}".format(self.train.select('PersonID').distinct().count()))
        print("participants in validate: {}".format(self.validate.select('PersonID').distinct().count()))
        print("participants in test: {}".format(self.test.select('PersonID').distinct().count()))
        print('\n')
        print("participants in both train & validate: {}".format(self.train.select('PersonID').distinct()\
                                         .join(self.validate.select('PersonID').distinct(),
                                               'PersonID', 'inner')\
                                         .count()))
        print("participants in both train & test: {}".format(self.train.select('PersonID').distinct()\
                                         .join(self.test.select('PersonID').distinct(),
                                               'PersonID', 'inner')\
                                         .count()))
        print('\n')
        print("{} in train: {}".format(event_param, self.train.select(event_param).distinct().count()))
        print("{} in validate: {}".format(event_param, self.validate.select(event_param).distinct().count()))
        print("{} in test: {}".format(event_param, self.test.select(event_param).distinct().count()))
        print('\n')
        print("{} in both train & validate: {}".format(event_param, self.train.select(event_param).distinct()\
                                         .join(self.validate.select(event_param).distinct(),
                                               event_param, 'inner')\
                                         .count()))
        print("{} in both train & test: {}".format(event_param, self.train.select(event_param).distinct()\
                                         .join(self.test.select(event_param).distinct(),
                                               event_param, 'inner')\
                                         .count()))

    def create_participate_matrices(self, event_param):
        '''For ALS modeling purposes, strips down data to only PersonID, chosen
        item feature (e.g. EventID, SeriesID, Venue_Zip, etc), Participated,
        and Event_Data columns'''

        self.train_mat = self.train.select("PersonID", event_param, "Participated", "Event_Date")
        self.validate_mat = self.validate.select("PersonID", event_param, "Participated", "Event_Date")
        self.test_mat = self.test.select("PersonID", event_param, "Participated", "Event_Date")

        #return self.train_mat, self.validate_mat, self.test_mat

    def fit_ALS(self, rank=100, maxIter=10, regParam=0.1,
                userCol="PersonID", itemCol="EventID", ratingCol="Participated",
                nonnegative=True, implicitPrefs=True, alpha=1.0, coldStartStrategy="drop"):
        '''Fit a single baseline ALS model based on selected parameters for training data'''

        als = ALS(rank=rank, maxIter=maxIter, regParam=regParam, userCol=userCol,
                    itemCol=itemCol, ratingCol=ratingCol, nonnegative=nonnegative,
                    implicitPrefs=implicitPrefs, alpha=alpha,
                    coldStartStrategy=coldStartStrategy)

        self.base_model = als.fit(self.train_mat)

        #return self.base_model

    def predict_ALS(self, model, event_param, scoring="rank"):
        '''run prediction on ALS model using provided scoring method, model, and
        validation dataset'''

        predictions = model.transform(self.validate_mat)

        pandas_preds = predictions.toPandas()
        valid_preds = pandas_preds[pd.notnull(pandas_preds['prediction'])]['Participated'].count()
        nan_preds = pandas_preds[pd.isnull(pandas_preds['prediction'])]['Participated'].count()
        print('Predictions includes {} valid values and {} nan values'.format(valid_preds, nan_preds))
        #print('\n')
        #print('Mean prediction is {}'.format(pandas_preds['prediction'].mean()))

        if scoring == "rank":
            import rank_eval
            rank_processing = rank_eval.RankEval(pandas_preds, "PersonID",
                        event_param, "Participated", "prediction")
            val_rank = rank_processing.calc_test_rank()
            pop_rank = rank_processing.calc_popular_rank()

            print("Model Rank = {} and Popular Rank = {}".format(val_rank, pop_rank))

            return predictions, val_rank, pop_rank
        else:
            #initial evaluation using RMSE (root-mean-squared-error)
            #address nan values with mean prediction value (for cold starts) if coldStartStrategy is nan

            evaluator = RegressionEvaluator(metricName=scoring, labelCol="Participated",
                                predictionCol="prediction")

            error = evaluator.evaluate(predictions.na.fill({'prediction':pandas_preds['prediction'].mean()}))

            print("Error of Type {} = {}".format(scoring, str(error)))

            return predictions, error

    def run_ALS_CV(event_param):

        estimator=ALS(userCol="PersonID", itemCol=event_param, ratingCol="Participated",
                    nonnegative=True, coldStartStrategy="drop", implicitPrefs=True)

        grid=ParamGridBuilder().addGrid(ALS.rank, [10, 50, 100]).addGrid(
            ALS.maxIter, [10, 50]).addGrid(ALS.regParam, [0.1, 0.01]).addGrid(
            ALS.alpha, [0.01, 1.0, 20]).build()

        evaluator=RegressionEvaluator(metricName="rmse", labelCol="Participated",
                                    predictionCol="prediction")

        cv = CrossValidator(estimator=estimator, estimatorParamMaps=grid,
                                    evaluator=evaluator, numFolds=3)

        cv_model = cv.fit(self.train_mat)
        cv_bestmodel = cv_model.bestModel

        print('Predictions on Training Data:')
        cv_trainpreds, cv_trainrmse = predict_ALS(cv_bestmodel, self.train_mat)

        print('Predictions on Validation Data:')
        cv_valpreds, cv_valrmse = predict_ALS(cv_bestmodel, self.validate_mat)

        return cv_model, cv_bestmodel, cv_trainpreds, cv_trainrmse, cv_valpreds, cv_valrmse

    def run_ALS_TVS(self, event_param, train_on_ones=False, predict_on_ones=False, scoring="rmse"):

        estimator=ALS(userCol="PersonID", itemCol=event_param,
                    ratingCol="Participated", nonnegative=True, implicitPrefs=True)

        grid=ParamGridBuilder().addGrid(ALS.rank, [10, 50, 100]).addGrid(
            ALS.maxIter, [10, 50]).addGrid(ALS.regParam, [0.1, 0.01]).addGrid(
            ALS.alpha, [0.01, 1.0, 20, 40]).addGrid(
            ALS.coldStartStrategy, ["nan", "drop"]).build()

        evaluator=RegressionEvaluator(metricName=scoring, labelCol="Participated",
                                    predictionCol="prediction")

        tvs = TrainValidationSplit(estimator=estimator, estimatorParamMaps=grid,
                                    evaluator=evaluator)

        if train_on_ones == True or predict_on_ones == True:
            train_wz_pd = self.train_mat.toPandas()
            train_nz = spark.createDataFrame(train_wz_pd[train_wz_pd['Participated'] == 1])
            val_wz_pd = self.val_mat.toPandas()
            validate_nz = spark.createDataFrame(val_wz_pd[val_wz_pd['Participated'] == 1])

        if train_on_ones == True:
            tvs_model = tvs.fit(train_nz)
        else:
            self.tvs_model = tvs.fit(self.train_mat)

        self.tvs_bestmodel = self.tvs_model.bestModel

        if predict_on_ones == True:
            print('Predictions on Training Data:')
            self.tvs_trainpreds, self.tvs_trainerror = self.predict_ALS(self.tvs_bestmodel, train_nz, scoring)
            print('Predictions on Validation Data:')
            self.tvs_valpreds, self.tvs_valerror = self.predict_ALS(self.tvs_bestmodel, validate_nz, scoring)

        else:
            print('Predictions on Training Data:')
            self.tvs_trainpreds, self.tvs_trainerror = self.predict_ALS(self.tvs_bestmodel, self.train_mat, scoring)
            print('Predictions on Validation Data:')
            self.tvs_valpreds, self.tvs_valerror = self.predict_ALS(self.tvs_bestmodel, self.validate_mat, scoring)

        #return self.tvs_model, self.tvs_bestmodel, self.tvs_trainpreds, self.tvs_trainerror, self.tvs_valpreds, self.tvs_valerror

if __name__ == '__main__':
    #'''placeholder - add code to save spark df's for future use once run one time'''
    #train_val_test_split(self.spark_full_df, .2, .2)
    #print_train_val_test_info("EventID")
    #create_participate_matrices("EventID")
    pass
