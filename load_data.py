import config
import sys
import split_types
import data_cleaner
import numpy as np

path = config.corpus
sys.path.append(path)

# Choose df
##============================================================================================
# FULL label
# df = pd.read_csv(path) #full label (e.g., INTP)
# X = df['posts'].tolist()
# Y = df['type'].tolist()

# Partial label
df = split_types.new_df  # partial label (e.g., I)


df = df.sample(frac=1, random_state=123) # randomizes whole dataframe

#Save it in order to skip loading time
# df.to_csv(config.path+'df.csv')

X = df['posts'].tolist()

# Clean data
##============================================================
data_cleaner = data_cleaner.data_cleaner()

# X1 = data_cleaner.preprocess1(X)
# np.save(config.path+'X1',X1)

X1 = np.load(config.path+'X1.npy')
# X1 = np.load(config.path+'X1.npy')
