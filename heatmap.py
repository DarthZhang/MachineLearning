import pandas as pd

# Get the liwc features from the train and test csv's into data frames
path_train = 'liwc_train.csv'
path_test = 'liwc_test.csv'
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

# Concatenate the 2 data frames because we want global characteristics
df_total = pd.concat([df_train, df_test], ignore_index=True)

# Remove useless columns
df_total = df_total.drop(['Filename', 'Segment'], axis=1)

print(df_total.shape)
print(df_total)

# df_total.to_csv('tmp.csv')


# Create the dataframe that will hold the statistics
features = df_total.columns.values
personalities = ['I', 'E', 'S', 'N', 'T', 'F', 'J', 'P']
df_final = pd.DataFrame(index=features, columns = personalities)
df_final = df_final.fillna(0)
print(df_final)

#Calculate average values for each feature
feature_means = []
for i in features:
    feature_means.append(df_total[i].mean())

print(feature_means)