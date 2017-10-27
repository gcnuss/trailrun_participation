import pandas as pd
import numpy as np
from collections import defaultdict
import geocoder

class RegressionDataPrep(object):
    '''ADD DOC STRING, PARAMS, ATTRIBUTES

    als_predictions should be a spark df of predictions on training data from
    the als model with PersonID, EventID, Participated, Event_Date, and
    prediction columns

    spark_data_df should be a spark df of training data
    with PersonID, EventID, Participated, Event_Date, SeriesID, EventTypeID,
    Total_Fee_Avg, Miles2_Avg, and Venue_Zip

    After processing, train_gb will be a pandas df of training data with columns
    based on the spark_data_df (but without PersonID, EventID, Participated, and
    prediction columns), and SeriesID, EventTypeID, and Venue_Zip converted to
    categorical columns using get dummies function'''

    def __init__(self, spark_data_df, user_df, datasplit, als_predictions=None):
        if als_predictions != None:
            self.als_predictions_pd = als_predictions.toPandas()
        self.data_df_pd = spark_data_df.toPandas()
        self.user_df = user_df
        self.datasplit = datasplit
        self.train_gb = None #rename this to be gb_data; not necessarilly training!

    def format_gb_data(self):
        '''Reformat data and combine to create appropriate set of features aligned
        with labels using output from ALS model fit / predict'''

        if self.datasplit == 'train':
            self.train_gb = pd.merge(self.data_df_pd, self.als_predictions_pd,
                            how='left', on=['PersonID', 'EventID',
                            'Participated', 'Event_Date'])
        else:
            self.train_gb = self.data_df_pd

        self.format_gb_user_data()
        self.train_gb = pd.merge(self.train_gb, self.user_gb_df, how='left',
                        on='PersonID')

        if self.datasplit == 'train':
            self.train_gb['y_label'] = self.train_gb[['Participated',
                'prediction']].apply(lambda row: 1 if row[0] == 1 else row[1], axis=1)
            self.train_gb.drop(['PersonID', 'EventID', 'Participated', 'prediction'],
                axis=1, inplace=True)
        else:
            self.train_gb['y_label'] = self.train_gb['Participated']
            self.train_gb.drop(['PersonID', 'EventID', 'Participated'],
                axis=1, inplace=True)

        self.train_gb = pd.get_dummies(data=self.train_gb,
                    prefix=['SeriesID','EventTypeID','Venue_Zip', 'Gender'],
                    columns=['SeriesID', 'EventTypeID', 'Venue_Zip', 'Gender'],
                    drop_first=True)

    def format_gb_user_data(self):
        '''Add user data to the gradient boosting dataset (starts with just event
        data)'''

        user_gb_data = []
        unique_personIDs = self.user_df['PersonID'].unique()

        for person in unique_personIDs:
            gender = self.user_df[self.user_df['PersonID'] == person]['Gender'].values[0]
            #seriesid = self.data_df[self.data_df['EventID'] == event]['SeriesID'].values[0]
            #eventtype = self.data_df[self.data_df['EventID'] == event]['EventTypeID'].values[0]
            ageavg = int(round(self.user_df[self.user_df['PersonID'] == person]['Age2'].values.mean()))
            #milesavg = int(round(self.data_df[self.data_df['EventID'] == event]['Miles2'].values.mean()))
            #venuezip = self.data_df[self.data_df['EventID'] == event]['Venue_Zip'].values[0]
            user_gb_data.append([person, gender, ageavg])

        self.user_gb_df = pd.DataFrame(user_gb_data, columns=['PersonID',
                            'Gender', 'AgeAvg'])


if __name__ == '__main__':
    pass
