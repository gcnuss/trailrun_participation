from sklearn.ensemble import GradientBoostingRegressor

class gradboost_model(object):
    '''ADD DOC STRING'''

    def __init__(self, train_data):
        self.train_data = train_data
        self.y = self.train_data['y_label']
        self.X = self.train_data.drop('y_label', axis=1)

    def instantiate_gb(self):
        self.gb_model = GradientBoostingRegressor()

    def baseline_fit(self):
        '''Runs fit method on gradient boosted model'''
        self.gb_model.fit(self.X, self.y)
