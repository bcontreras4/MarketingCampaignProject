import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit

pd.set_option('display.max_columns', None)

#********************** Classes *************************************#

# Class calculates how many Campaigns each costomer excepts and
# if they accepted any at all. Then, it includes this data as a new
# column in our numerical dataset.

class new_attributes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        print(X)
        X['AcceptedAnyCmp'] =((X["AcceptedCmp1"] + X["AcceptedCmp2"] + \
    X["AcceptedCmp3"] + X["AcceptedCmp4"] + X["AcceptedCmp5"]) >= 1)

        return X

# Class removes a costomer row if they included a nonsensical answer
# for their marital status.

class remove_customer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = (X[X['Marital_Status'] != "Alone"])
        X = (X[X['Marital_Status'] != "Absurd"])
        X = (X[X['Marital_Status'] != "YOLO"])
        return X
#!!!!!!!!!!! THIS MIGHT NEED TO BE CHANGED SINCE X IS AN ARRAY NOT A DF !!!!!!!!!!!!!#
# Class turns the date that each person became a customer
# into a number where the lower numbers = longer time as customer
'''
class Dt_Label(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.Dt_le = LabelEncoder()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        print(X)
        Dt_Labels = self.Dt_le.fit_transform(X[:,7])
        Dt_mappings = {index: label for index, label in
                            enumerate(self.Dt_le.classes_)}
        X[:,7] = Dt_Labels

        return X
'''
#************* Fetching Data and Preparing for Pipelines ***************#

# First we get our data and seperate it into it's numerical components
# and its categorical components so that we can work with it easily

campaign_info = pd.read_csv("marketing_campaign.csv", delimiter = ';')

cat_attributes = ["Dt_Customer", "Education", "Marital_Status"]
One_Hot_attributes = ["Education", "Marital_Status"]
Dt_Attributes = ['Dt_Customer']

cat_cmp_info = campaign_info[One_Hot_attributes]

#**************** Add "NumAcceptedCmp" early, so we can use it to stratify ************#


campaign_info["NumAcceptedCmps"] = (campaign_info["AcceptedCmp1"] + \
     campaign_info["AcceptedCmp2"] + campaign_info["AcceptedCmp3"] + \
          campaign_info["AcceptedCmp4"] + campaign_info["AcceptedCmp5"])

num_cmp_info = campaign_info.drop(cat_attributes, axis=1)

num_attributes = list(num_cmp_info) 

#print(campaign_info)

#****************** Splitting Data into Training and Test Set **********************#

split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(campaign_info, campaign_info["NumAcceptedCmps"]):
    strat_train_set = campaign_info.loc[train_index]
    strat_test_set = campaign_info.loc[test_index]

# We make a copy of the training set to use, so we always have an untouched version.
# Any pipelines applied to the training set will be done to this copied version.

cmp_info = strat_train_set.copy()

#***************** Categorical Data Pipeline ********************************#

# For our categorical data, we rep. it numerically using OneHotEncoder
# if it's Education or Marital_Status or with LabelEncoder for Dt_Customer,
# and we get rid of seven rows where the customer entered nonsensical
# answers for their marital status.

cat_pipeline = Pipeline([
                    ('cat_encoder', ColumnTransformer([
                        ('onehot',OneHotEncoder(),One_Hot_attributes),
                        ('Dt', OrdinalEncoder(), Dt_Attributes)])),\
                ])

#***************** Numerical Data Pipeline *************************#

# Here, we fill in empty income values with the median value,
# create two extra attributes (AcceptedAnyCmp and NumAcceptedCmp),
# and scale all of the numerical attributes.

num_pipeline = Pipeline([
                ('new_attributes', new_attributes()),\
                ('imputer', SimpleImputer(strategy='median')),\
                #('new_attributes', new_attributes()),\
                ('std_scaler', StandardScaler())
            ])

#************************ Full Pipeline ********************************#

# We put our two pipelines together into a single pipeline that
# handles both our categorical and numerical data.

final_pipeline = Pipeline([
                ('customer_remover', remove_customer()),\
                ('catnum', ColumnTransformer([
                ('num', num_pipeline, num_attributes),\
                ('cat', cat_pipeline, cat_attributes)]))
                ])

#!!!!!!!!!!!!!!!!!!!!!! Testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

#******************** Applying the pipeline ****************************#
cols = np.concatenate((cmp_info.columns, ['AcceptedAnyCmp'], cmp_info.Education.unique(),cmp_info.Marital_Status.unique(),['Dt_label']))

cols = np.delete(cols, [2,3,7,41,42,43])

cmp_info_prepared = final_pipeline.fit_transform(cmp_info)

print(pd.DataFrame(cmp_info_prepared, columns=cols))

#****************** Put next thing here ************************#

# !!!!!!!!!!!!!!!!!!!!!!!!!! Fix stnd scalar so it only effects neccesarry attributes and leaves others alone.
# !!!!!!!!!!!!!!!!!!!!!!!!!! You should also make it range (0,1) since 
# !!!!!!!!!!!!!!!!!!!!!!!!!! things like income shouldn't be negative