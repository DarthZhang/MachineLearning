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
#
#df = split_types.new_df
#df_full_label = pd.read_csv(path)
#
#df = df.sample(frac=1, random_state=123) # randomizes whole dataframe
#
#df_full_label = df_full_label.sample(frac=1, random_state=123)
# #
#Y = df_full_label['type']
#X = df_full_label['posts']
#
# # Clean data
# ##============================================================
#data_cleaner = data_cleaner.data_cleaner()
#
#X1 = data_cleaner.preprocess2(X,Y,n_posts)

#
# X1 = pd.DataFrame(np.load(config.path+'X_half_post_full.npy'))
# df_full_label = X1.sample(frac=1, random_state=123).values
#
# X2 = []
# Y = []
# for element in df_full_label:
#       X2.append(element[1])
#       Y.append(element[0])
# np.save(config.path+'Y__half_post.npy', Y)
# np.save(config.path+'X_half_post.npy',X2)
#
# exit(0)
Y = np.load(config.path+'Y__half_post.npy')
X1 = np.load(config.path+'X_half_post.npy')