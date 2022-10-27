import sys
from packages import *
from ml_fairness import *
from standard_data import *
import joblib
import numpy as np


def DATA_MAPPING_SVC(data):
    data.append(0)
    file_path = 'data/bank/bank-additional-full.csv'

    column_names = []
    na_values=['unknown']

    df = pd.read_csv(file_path, sep=';', na_values=na_values)

    # Drop na values
    dropped = df.dropna()
    count = df.shape[0] - dropped.shape[0]
    # print("Missing Data: {} rows removed.".format(count))
    df = dropped
    df.loc[len(df.index)] = data
    df['age'] = df['age'].apply(lambda x: np.float(x >= 25))

    df['poutcome'] = df['poutcome'].map({'failure': -1,'nonexistent': 0,'success': 1})
    df['default'] = df['default'].map({'yes': -1,'unknown': 0,'no': 1})
    df['housing'] = df['housing'].map({'yes': -1,'unknown': 0,'no': 1})
    df['loan'] = df['loan'].map({'yes': -1,'unknown': 0,'no': 1})

    nominal = ['job','marital','education','contact','month','day_of_week']
    df = pd.get_dummies(df, columns=nominal)
    # In[3]:
    pro_att_name = ['age'] # ['race', 'sex']
    priv_class = [1] # ['White', 'Male']
    reamining_cat_feat = []
    seed = randrange(100)

    y1_data_orig, y1_X, y1_y = load_bank_data(df, pro_att_name, priv_class, reamining_cat_feat)
    sc2 = StandardScaler()
    y1_X_train = sc2.fit_transform(y1_data_orig.features)
    return [y1_X_train[-1]]


def DATA_MAPPING_RANDOM(data):
    data.append(0)
    file_path = 'data/bank/bank-additional-full.csv'

    column_names = []
    na_values = ['unknown']

    df = pd.read_csv(file_path, sep=';', na_values=na_values)

    # Drop na values
    dropped = df.dropna()
    count = df.shape[0] - dropped.shape[0]
    # print("Missing Data: {} rows removed.".format(count))
    df = dropped
    df.loc[len(df.index)] = data
    df['age'] = df['age'].apply(lambda x: np.float(x >= 25))
    cat_feat = ['job', 'default', 'housing', 'contact', 'month', 'day_of_week', 'poutcome']
    df.drop(['marital', 'education'], axis=1, inplace=True)

    # Create a one-hot encoding of the categorical variables.
    df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
    pro_att_name = ['age']  # ['race', 'sex']
    priv_class = [1]  # ['White', 'Male']
    reamining_cat_feat = ['loan']
    y1_data_orig, y1_X, y1_y = load_bank_data(df, pro_att_name, priv_class, reamining_cat_feat)
    return [y1_X[-1]]

