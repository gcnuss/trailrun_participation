import cPickle as pickle
import pandas as pd

def load_files(path):
    '''Loads a pickled file when provided a path'''
    with open (path, 'rb') as f:
        filename = pickle.load(f)
    return filename

def merge_data(data_df, preds_df, event_param):
    '''Merges ALS model prediction pandas df's into a combined dataframe for use
    in gradient boosted ensemble

    data_df must be a pandas df with columns PersonID, EventID, Participated,
    Event_Date, SeriesID, EventTypeID, Total_Fee_Avg, Miles2_Avg, and Venue_Zip

    preds_df must be a pandas df with columns PersonID, Participated, Event_Date,
    prediction, and the event_param used for itemCol in the ALS model

    event_param is the column used for itemCol in teh ALS model'''

    gb_data = pd.merge(data_df, preds_df, how='left',
                on=['PersonID', 'Participated', 'Event_Date', event_param])
    gb_data['{}_prediction'.format(event_param)] = gb_data['prediction']
    gb_data.drop(['prediction', event_param], axis = 1, inplace=True)

    return gb_data
