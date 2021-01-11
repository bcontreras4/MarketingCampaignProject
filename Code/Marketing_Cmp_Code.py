import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

#********************** Classes *************************************#

# Class calculates how many Campaigns each costomer excepts and
# if they accepted any at all. Then, it includes this data as a new
# column in our numerical dataset.

class new_attributes(BaseEstimater, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        AcceptedAnyCmp = (X["AcceptedCmp1"] + X["AcceptedCmp2"] + \
                        X["AcceptedCmp3"] + X["AcceptedCmp4"] + \
                        X["AcceptedCmp5"]) >=1

        NumAcceptedCmp = (X["AcceptedCmp1"] + X["AcceptedCmp2"] + \ 
                        X["AcceptedCmp3"] + X["AcceptedCmp4"] + \
                        X["AcceptedCmp5"])
        return np.c_[X, AcceptedAnyCmp, NumAcceptedCmp]

# Class removes a costomer row if they included a nonsensical answer
# for their marital status.

class remove_customer(BaseEstimater, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = (X[["Marital_Status"] != "Alone"]
        X = (X[["Marital_Status"] != "Absurd"]
        X = (X[["Marital_Status"] != "YOLO"]

        return X

#************* Fetching Data and Preparing for Pipelines ***************#

# First we get our data and seperate it into it's numerical components  #
# and its categorical components so that we can work with it easily     #

campaign_info = pd.read_csv("marketing_campaign.csv", delimiter = ';')

cat_attributes = ["Dt_Customer", "Education", "Marital_Status"]
One_Hot_attributes = ["Education", "Marital_Status"]
Dt_Attributes = ['Dt_Customer']

num_cmp_info = campaign_info.drop(cat_attributes, axis=1) 
cat_cmp_info = campaign_info[One_Hot_attributes]

#***************** Categorical Data Pipeline ********************************#

# For our categorical data, we rep. it numerically using OneHotEncoder       #
# if it's Education or Marital_Status or with LabelEncoder for Dt_Customer,  #
# and we get rid of seven rows where the customer entered nonsensical        #
# answers for their marital status.                                          #

cat_pipeline = Pipeline([
                    ('cat_encoder', ColumnTransformer([
                        ('onehot',OneHotEncoder(),One_Hot_attributes),\
                        ('Dt', LabelEncoder(), Dt_Attributes),\
                                    ])),\
                    ('customer_remover', remove_customer())
                ])

#***************** Numerical Data Pipeline *************************#

# Here, we fill in empty income values with the median value,       #
# create two extra attributes (AcceptedAnyCmp and NumAcceptedCmp),  #
# and scale all of the numerical attributes.                        #

num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),\
                ('new_attributes', new_attributes()),\
                ('std_scaler', StandardScaler()), \
            ])

#************************ Full Pipeline ********************************#

# We put our two pipelines together into a single pipeline that         #
# handles both our categorical and numerical data.                      #


