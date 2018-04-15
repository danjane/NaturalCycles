# Warning, this module loads all the data (even on import) to perform the preprocessing

import random
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection
import xgboost

import scipy.sparse

data_file = './Data/anafile_challenge_170522.csv'

numerical_columns = ['Age', 'NumBMI', 'Weight', 'TempLogFreq', 'SexLogFreq', 'AnovCycles ']
categorical_columns = ['Country', 'Pill', 'NCbefore', 'FPlength', 'CycleVar', 'ExitStatus']
new_column = 'WeightOrBmiMissing'  # I will add this to categorical_columns
fitting_columns = ['DaysTrying', 'CyclesTrying']

simple_lasso_model = './Data/simple_lasso_model.csv'

# Use the same seed each time for repeatable results
same_random_seed = int.from_bytes('nc'.encode('utf-8'), 'little')  # cutesy
random.seed(same_random_seed)


def get_X_Y_from_df_simple(df):
    # Simple, just throw out rows with missing data
    df = df[df['Weight'] != 0]
    df = df[df['NumBMI'] != 0]
    Y = df['CyclesTrying'].as_matrix()
    X = df[numerical_columns].as_matrix()

    X = np.concatenate([X, X ** 2], axis=1)  # basic non-linear model
    return X, Y


def fix_pill(df):
    # Handle the Pill naming madness (cruel, cruel...)
    # I use ['Recent', 'LongAgo', 'Never']
    df['Pill'] = df['Pill'].replace(True, 'Recent')
    df['Pill'] = df['Pill'].replace(False, 'Never')
    df = df.fillna('LongAgo')
    return df


def column_for_weight_missing(df, categorical_columns):
    df[new_column] = 'No'
    idx = df['Weight']*df['NumBMI'] == 0
    df.loc[idx, new_column] = 'Yes'
    categorical_columns.append(new_column)
    return df, categorical_columns


# We read and preprocess the data here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("Reading the data from {}".format(data_file))
data = pd.read_csv(data_file, skipinitialspace=True)
assert len(set(data) -
        set(numerical_columns) - set(categorical_columns) - set(fitting_columns)) == 0, 'Missing columns'
assert (data.shape[1] -
        len(numerical_columns) - len(categorical_columns) - len(fitting_columns)) == 0, 'Overlapping columns'

print('Doing a little preprocessing (fixing pill, noting when weight is missing)')
data = fix_pill(data)
print(categorical_columns)
data, categorical_columns = column_for_weight_missing(data, categorical_columns)
print(categorical_columns)

# Change categories into Integer classes
categorical_names = {}
for feature in categorical_columns:
    print(feature)
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[feature])
    data[feature] = le.transform(data[feature])
    categorical_names[feature] = list(le.classes_)

# Define a function for one hot encoding the categorical data of a dataframe
data_cat = data[categorical_columns].values.astype(int)
encoder = sklearn.preprocessing.OneHotEncoder()
encoder.fit(data_cat)


def encode(df):
    df_cat = encoder.transform(df[categorical_columns].values.astype(int))
    df_num = df[numerical_columns].astype(float)
    return scipy.sparse.hstack([df_cat, df_num])


print("Setting aside a test subsample (data_test) for later")
data_test = data.sample(frac=0.2, random_state=same_random_seed)
data = data.drop(data_test.index)


if __name__ == "__main__":

    print("Checking the data, starting with numerical data:")
    print(data.describe().round(1))
    # print(data.describe(include=object))

    print("Have a go for time to preg")
    data_preg = data[data['ExitStatus'] == categorical_names['ExitStatus'].index('Pregnant')]
    data_train = data_preg.sample(frac=0.75, random_state=same_random_seed)
    data_validate = data_preg.drop(data_train.index)

    # First attempt with Linear Regression
    X_train, Y_train = get_X_Y_from_df_simple(data_train)
    X_validate, Y_validate = get_X_Y_from_df_simple(data_validate)

    regr = sklearn.linear_model.Lasso()
    alphas = np.logspace(-6, 3, 20)
    scores = [regr.set_params(alpha=alpha, max_iter=10000, normalize=True
                              ).fit(X_train, Y_train
                                    ).score(X_validate, Y_validate)
              for alpha in alphas]
    rsq = max(scores)
    best_alpha = alphas[scores.index(rsq)]
    regr.set_params(alpha=best_alpha, max_iter=10000, normalize=True)
    regr.fit(X_train, Y_train)
    acc_score = sklearn.metrics.r2_score(Y_validate, regr.predict(X_validate))
    print("validation r2_score is {:.1f}%".format(acc_score * 100))

    new_shape = (2, len(numerical_columns))
    modelL1 = pd.DataFrame(data=regr.coef_.reshape(new_shape), columns=numerical_columns)

    print("Lasso regression with alpha = {:.2g}, coefficients:".format(best_alpha))
    print(modelL1)
    modelL1.to_csv(simple_lasso_model, index=False)

    # Now try a boosted tree
    print('And with a boosted tree and categorical data')
    X_train = encode(data_train)
    Y_train = data_train['CyclesTrying'].values.astype(np.int8)

    X_validate = encode(data_validate)
    Y_validate = data_validate['CyclesTrying'].values.astype(np.int8)

    gbtree = xgboost.XGBClassifier(n_estimators=50, max_depth=3)
    gbtree.fit(X_train, Y_train)

    acc_score = sklearn.metrics.r2_score(Y_validate, gbtree.predict(X_validate))
    print("validation r2_score is {:.1f}%".format(acc_score * 100))




# predict_fn = lambda x: gbtree.predict_proba(encode(x))
