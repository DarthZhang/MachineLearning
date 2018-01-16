import config
import sys
import split_types
import data_cleaner
import numpy as np
import pandas as pd

path = config.corpus
sys.path.append(path)

# Choose df
##============================================================================================
# FULL label

df = split_types.new_df
df = df.sample(frac=1, random_state=123) # randomizes whole dataframe
X = df['posts']

# Clean data
##============================================================
data_cleaner = data_cleaner.data_cleaner()

#X1 = data_cleaner.preprocess1(X)
#np.save(config.path+'X1',X1)

#print(X1)
#exit(0)

X1 = np.load(config.path+'X1.npy')
