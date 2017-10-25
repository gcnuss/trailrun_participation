#FILE INWORK, NOT FUNCTIONING YET; CURRENTLY RUNNING OVERALL FLOW IN JUPYTER

import als_model
import gradboost_model
import prep_gbdata
import rank_eval
import cPickle as pickle

class TotalModelPipeline(object):
    '''This wrapper class runs end to end through the modeling for my trail run
    project.  This does assume you already have a cleaned dataframe prepared and
    saved in a pickled file as that effort does not need to be repeated for every
    model run.  If you need to create this file, run file dataprep.py.

    Parameters:
    TBD
    '''

    def __init__(self, cleaned_data_path):
        with open(cleaned_data_path, 'rb') as f:
            self.cleaned_data = pickle.load(f)
        self.als_model = None

    def run_ALS(self, event_param, ):
        '''instantiate a class instance for an ALS implicit recommender. '''

        self.als_model = als_model.implicit_als(self.cleaned_data)
        self.als_model.prep_spark_full_df()
        self.als_model.train_val_test_split()
        self.als_model.print_train_val_test_info(event_param)
        self.als_model.create_participate_matrices(event_param)
