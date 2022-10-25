import sys
from packages import *
from ml_fairness import *
from standard_data import *
import joblib
import numpy as np


def DATA_PROCESSING(data):
    data.append(0)
    file_path = 'data/bank/bank-additional-full.csv'

    column_names = []
    na_values = ['unknown']

    df = pd.read_csv(file_path, sep=';', na_values=na_values)

    #### Drop na values
    dropped = df.dropna()
    count = df.shape[0] - dropped.shape[0]
    # print("Missing Data: {} rows removed.".format(count))
    df = dropped

    df.loc[len(df.index)] = data
    df['age'] = df['age'].apply(lambda x: np.float(x >= 25))

    # Create a one-hot encoding of the categorical variables.
    cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']

    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    def duration(dataframe):
        q1 = dataframe['duration'].quantile(0.25)
        q2 = dataframe['duration'].quantile(0.50)
        q3 = dataframe['duration'].quantile(0.75)
        dataframe.loc[(dataframe['duration'] <= q1), 'duration'] = 1
        dataframe.loc[(dataframe['duration'] > q1) & (dataframe['duration'] <= q2), 'duration'] = 2
        dataframe.loc[(dataframe['duration'] > q2) & (dataframe['duration'] <= q3), 'duration'] = 3
        dataframe.loc[(dataframe['duration'] > q3), 'duration'] = 4
        # print(q1, q2, q3)
        return dataframe

    df = duration(df)

    df.loc[(df['pdays'] == 999), 'pdays'] = 1
    df.loc[(df['pdays'] > 0) & (df['pdays'] <= 10), 'pdays'] = 2
    df.loc[(df['pdays'] > 10) & (df['pdays'] <= 20), 'pdays'] = 3
    df.loc[(df['pdays'] > 20) & (df['pdays'] != 999), 'pdays'] = 4

    df.loc[(df['euribor3m'] < 1), 'euribor3m'] = 1
    df.loc[(df['euribor3m'] > 1) & (df['euribor3m'] <= 2), 'euribor3m'] = 2
    df.loc[(df['euribor3m'] > 2) & (df['euribor3m'] <= 3), 'euribor3m'] = 3
    df.loc[(df['euribor3m'] > 3) & (df['euribor3m'] <= 4), 'euribor3m'] = 4
    df.loc[(df['euribor3m'] > 4), 'euribor3m'] = 5

    df['poutcome'].replace(['nonexistent', 'failure', 'success'], [1, 2, 3], inplace=True)
    pro_att_name = ['age']  # ['race', 'sex']
    priv_class = [1]  # ['White', 'Male']
    reamining_cat_feat = []
    seed = randrange(100)

    y1_data_orig, y1_X, y1_y = load_bank_data(df, pro_att_name, priv_class, reamining_cat_feat)
    sc2 = StandardScaler()
    y1_X_train = sc2.fit_transform(y1_data_orig.features)
    return [y1_X_train[-1]]


