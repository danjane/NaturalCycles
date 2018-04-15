import random
import pandas as pd

data_file = './Data/anafile_challenge_170522.csv'

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