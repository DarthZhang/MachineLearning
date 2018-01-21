import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import config
import load_data

# Get the liwc features from the train and test csv's into data frames
path_train = config.path + 'liwc_train.csv'
path_test = config.path + 'liwc_test.csv'
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

if not load_data.full_or_partial_label:
    print('Use full label!')
    exit(0)

# Get personality labels on the same position in the list as the features
train_labels = load_data.Ytrain
test_labels = load_data.Ytest
total_labels = pd.Series(train_labels).tolist() + pd.Series(test_labels).tolist()

# Concatenate the 2 data frames because we want global characteristics
df_total = pd.concat([df_train, df_test], ignore_index=True)

# Remove useless columns
df_total = df_total.drop(['Filename', 'Segment'], axis=1)

# print(df_total.shape)
# print(df_total)
# print(len(total_labels))
# print(total_labels)
# df_total.to_csv('tmp.csv')


# Create the dataframe that will hold the statistics
features = df_total.columns.values
personalities = ['I', 'E', 'S', 'N', 'T', 'F', 'J', 'P']
df_final = pd.DataFrame(index=features, columns=personalities)
df_final = df_final.fillna(0)
print(df_final)

# Calculate average values for each feature
feature_means = []
for i in features:
    feature_means.append(df_total[i].mean())

print(feature_means)


# Function that returns the mean of all values of a personality type for a feature
# feature = name of the feature
# type = letter of personality e.g. I, N, etc.
# position = position of the letter inside the label i.e. 0, 1, 2, 3
def mean_feature_type(feature, type, position):
    if position < 0 or position > 3:
        print('Position must be between 0 and 3')
        exit(0)

    values_list = pd.Series(df_total[feature]).tolist()
    sum = 0.0
    count = 0
    for i in range(0, len(values_list)):
        label = total_labels[i]
        if type == label[position]:
            sum = sum + values_list[i]
            count = count + 1
    return sum/count


# The actual computation of the values inside the heatmap. The value from row i, column j will be the ratio
# between the mean of all values of personality j for feature i and the mean of all values for feature i.
x = 1
i_numeral = 0
for i in features:
    j_numeral = 0
    for j in personalities:
        denominator = feature_means[i_numeral]
        nominator = mean_feature_type(i, j, j_numeral//2)
        df_final.loc[i, j] = nominator/denominator
        j_numeral = j_numeral + 1
    i_numeral = i_numeral + 1

# print(df_final)


# Plot heatmap
plt.figure(0)
sns.heatmap(df_final)
plt.show()

plt.figure(1)
sns.heatmap(df_final.iloc[0:30])
plt.show()

plt.figure(2)
sns.heatmap(df_final[30:60])
plt.show()

plt.figure(3)
sns.heatmap(df_final[60:93])
plt.show()
