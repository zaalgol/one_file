import pandas as pd

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 19 columns):
Survived         891 non-null int64
Pclass           891 non-null int64
Name             891 non-null object
Sex              891 non-null object
Age              891 non-null float64
SibSp            891 non-null int64
Parch            891 non-null int64
Fare             891 non-null float64
Embarked         891 non-null object
FamilySize       891 non-null int64
IsAlone          891 non-null int64
Title            891 non-null object
FareBin          891 non-null category
AgeBin           891 non-null category
Sex_Code         891 non-null int64
Embarked_Code    891 non-null int64
Title_Code       891 non-null int64
AgeBin_Code      891 non-null int64
FareBin_Code     891 non-null int64
dtypes: category(2), float64(2), int64(11), object(4)
memory usage: 120.3+ KB
None
"""
from xgboost_classifier_class import XGBoostClassifier


class XGBoostClassifierTitanic(XGBoostClassifier):
    def clean_data(self):
        ###COMPLETING: complete or delete missing values in train and test/validation dataset
        for dataset in self.df:
            # complete missing age with median
            dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

            # complete embarked with mode
            dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

            # complete missing fare with median
            dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

        # delete the cabin feature/column and others previously stated to exclude in train dataset
        drop_column = ['PassengerId', 'Cabin', 'Ticket']
        self.df.drop(drop_column, axis=1, inplace=True)

    def feature_engineering(self):
        ###CREATE: Feature Engineering for train and test/validation dataset
        for dataset in self.df:
            # Discrete variables
            dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

            dataset['IsAlone'] = 1  # initialize to yes/1 is alone
            dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0  # now update to no/0 if family size is greater than 1

            # quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
            dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

            # Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
            # Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
            dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

            # Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
            dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

        # cleanup rare title names
        # print(data1['Title'].value_counts())
        stat_min = 10  # while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
        title_names = (self.df[
                           'Title'].value_counts() < stat_min)  # this will create a true false series with title name as index

        # apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
        self.df['Title'] = self.df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
