import random
import pandas as pd
import numpy as np
import sklearn.linear_model

data_file = './Data/anafile_challenge_170522.csv'

numerical_columns = ['Age', 'NumBMI', 'TempLogFreq', 'SexLogFreq', 'AnovCycles ']

def get_X_Y_from_df(df):
    Y = df['CyclesTrying'].as_matrix()
    X = df[numerical_columns].as_matrix()
    X = np.concatenate([X, X ** 2], axis=1)  # basic non-linear model
    return X, Y

# Use the same seed each time for repeatable results
same_random_seed = int.from_bytes('nc'.encode('utf-8'), 'little')  # cutesy
random.seed(same_random_seed)

if __name__ == "__main__":
    print("Reading the data from {}".format(data_file))
    data = pd.read_csv(data_file, skipinitialspace=True)

    print("Setting aside a test subsample for later")
    data_test = data.sample(frac=0.2, random_state=same_random_seed)
    data = data.drop(data_test.index)

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
    print("validation r^2 is {:.1f}%".format(rsq*100))
    print(regr.coef_)

    new_shape = (2, len(numerical_columns))
    modelL1 = pd.DataFrame(data=regr.coef_.reshape(new_shape), columns=numerical_columns)

    print(modelL1)
