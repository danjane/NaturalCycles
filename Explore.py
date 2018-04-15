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
fitting_columns = ['DaysTrying', 'CyclesTrying']

simple_lasso_model = './Data/simple_lasso_model.csv'

# Use the same seed each time for repeatable results
same_random_seed = int.from_bytes('nc'.encode('utf-8'), 'little')  # cutesy
random.seed(same_random_seed)


def get_X_Y_from_df(df):
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


# We read and preprocess the data here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("Reading the data from {}".format(data_file))
data = pd.read_csv(data_file, skipinitialspace=True)
assert len(set(data) -
        set(numerical_columns) - set(categorical_columns) - set(fitting_columns)) == 0, 'Missing columns'
assert (data.shape[1] -
        len(numerical_columns) - len(categorical_columns) - len(fitting_columns)) == 0, 'Overlapping columns'

print('Doing a little preprocessing (fixing pill, ')
data = fix_pill(data)

print("Setting aside a test subsample (data_test) for later")
data_test = data.sample(frac=0.2, random_state=same_random_seed)
data = data.drop(data_test.index)


if __name__ == "__main__":

    print("Checking the data, starting with numerical data:")
    print(data.describe().round(1))
    print(data.describe(include=object))

    print("Have a go for time to preg")
    data_preg = data[data['ExitStatus'] == 'Pregnant']
    data_train = data_preg.sample(frac=0.75, random_state=same_random_seed)
    data_validate = data_preg.drop(data_train.index)

    X_train, Y_train = get_X_Y_from_df(data_train)
    X_validate, Y_validate = get_X_Y_from_df(data_validate)

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
    print("validation r^2 is {:.1f}%".format(rsq * 100))

    new_shape = (2, len(numerical_columns))
    modelL1 = pd.DataFrame(data=regr.coef_.reshape(new_shape), columns=numerical_columns)

    print("Lasso regression with alpha = {:.2g}, coefficients:".format(best_alpha))
    print(modelL1)
    modelL1.to_csv(simple_lasso_model, index=False)


# I used the code from here: https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html
print('Trying a full tree')
data2 = data.copy()

data2['Pill'] = data2['Pill'].replace(True, 'Recent')
data2['Pill'] = data2['Pill'].replace(False, 'Never')
data2.fillna('LongAgo', inplace=True)

# Change categories into one hots
categorical_names = {}
for feature in categorical_columns:
    print(feature)
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data2[feature])
    data2[feature] = le.transform(data2[feature])
    categorical_names[feature] = le.classes_
data_cat = data2[categorical_columns].values.astype(int)
encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_names)
encoder.fit(data_cat)
data_cat = encoder.transform(data_cat)

data_num = data2[numerical_columns].astype(float)

labels = data['CyclesTrying'].values.astype(np.int8)

data_processed = scipy.sparse.hstack([data_cat, data_num])

np.random.seed(same_random_seed)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data_processed, labels, train_size=0.80)

gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(train, labels_train)

sklearn.metrics.accuracy_score(labels_test, gbtree.predict(test))
# 0.31742399476953254

predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)
