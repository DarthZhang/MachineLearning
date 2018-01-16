import config
import sys
import split_types
import data_cleaner
import pandas as pd
import numpy as np
import importlib
# importlib.reload(config)
path = config.corpus
sys.path.append(path)

full_or_partial_label = True # True = full, False = Partial

# Choose df
##======================================================================
#Load data
if full_or_partial_label:
    # FULL label
    df = pd.read_csv(path) #full label (e.g., INTP)
    df = df.sample(frac=1, random_state=123)  # should randomize whole dataframe)
    Y = df['type'].tolist()
else:
    # Partial label
    df = split_types.new_df  # partial label (e.g., I)
    df = df.sample(frac=1, random_state=123) # should randomize whole dataframe)


# Clean data
##============================================================
# data_cleaner = data_cleaner.data_cleaner()

## First time you run:
# X = df['posts'].tolist()
# print('preprocessing corpus...')
# X1 = data_cleaner.preprocess1(X)
# print('Done preprocessing')
# np.savez_compressed(config.path+'X2',a=X1)

# After or load Xtrain, Xvalidation and Xtest below:
X1 = np.load(config.path+'X2.npz')['a'].tolist()

# split 0.8-0.2
##============================================================
split_point = int(0.8 * len(X1))

# Create train and test sets
# Xtrain,Xtest = np.split(X1, [split_point])
# if full_or_partial_label:
#     Ytrain, Ytest = np.split(Y, [split_point])
#
# np.savez_compressed(config.path+'train_test_split', a=Xtrain, b=Xtest,c=Ytrain, d=Ytest) #Save them

# Load
loaded = np.load(config.path+'train_test_split.npz')
Xtrain = loaded['a']
Xtest = loaded['b']
if full_or_partial_label:
    Ytrain = loaded['c']
    Ytest = loaded['d']