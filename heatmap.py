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
# df_total.to_csv('tmp.csv')


features = df_total.columns.values
# Create the dataframe that will hold the statistics for the means heatmap
columns_means = ['I', 'E', 'S', 'N', 'T', 'F', 'J', 'P']
df_final_mean = pd.DataFrame(index=features, columns=columns_means)
df_final_mean = df_final_mean.fillna(0)
# print(df_final_mean)

# Create the dataframe that will hold the statistics for the differences heatmap
columns_differences = ['E-I', 'N-S', 'F-T', 'P-J']
df_final_difference = pd.DataFrame(index=features, columns=columns_differences)
df_final_difference = df_final_difference.fillna(0)
# print(df_final_difference)

# Create the dataframe that will hold the statistics for the ratios heatmap
columns_ratios = ['E/I', 'N/S', 'F/T', 'P/J']
df_final_ratios = pd.DataFrame(index=features, columns=columns_ratios)
df_final_ratios = df_final_ratios.fillna(0)
# print(df_final_ratios)

# Calculate average values for each feature
feature_means = []
for i in features:
    feature_means.append(df_total[i].mean())

print(feature_means)

# Calculate range for each feature
feature_ranges = []
for i in features:
    max = df_total[i].max()
    min = df_total[i].min()
    feature_ranges.append(max-min)

print(feature_ranges)


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


# The actual computation of the values inside the means heatmap. The value from row i, column j will be the ratio
# between the mean of all values of personality j for feature i and the mean of all values for feature i.
i_numeral = 0
for i in features:
    j_numeral = 0
    for j in columns_means:
        denominator = feature_means[i_numeral]
        nominator = mean_feature_type(i, j, j_numeral//2)
        df_final_mean.loc[i, j] = nominator / denominator
        j_numeral = j_numeral + 1
    i_numeral = i_numeral + 1

# print(df_final_means)

# The actual computation of the values inside the differences heatmap. The value from row i, column j will be the
# difference between the means of the 2 opposite personalities in column j, normalized by the range of the feature
i_numeral = 0
for i in features:
    j_numeral = 0
    for j in columns_differences:
        denominator = feature_ranges[i_numeral]
        nominator = mean_feature_type(i, j[0], j_numeral) - mean_feature_type(i, j[2], j_numeral)
        df_final_difference.loc[i, j] = nominator / denominator
        j_numeral = j_numeral + 1
    i_numeral = i_numeral + 1

# The actual computation of the values inside the ratios heatmap. The value from row i, column j will be the
# ratio between the means of the 2 opposite personalities in column j
i_numeral = 0
for i in features:
    j_numeral = 0
    for j in columns_ratios:
        mean1 = mean_feature_type(i, j[0], j_numeral)
        mean2 = mean_feature_type(i, j[2], j_numeral)
        df_final_ratios.loc[i, j] = mean1 / mean2
        j_numeral = j_numeral + 1
    i_numeral = i_numeral + 1

# plt.figure(0)
# current_palette = sns.color_palette()
# palette = sns.color_palette("RdBu_r", 5)
# sns.heatmap(df_final_mean, cmap=palette, center=1)
# plt.show()
#
# plt.figure(1)
# current_palette = sns.color_palette()
# palette = sns.color_palette("RdBu_r", 5)
# sns.heatmap(df_final_difference, cmap=palette, center=0)
# plt.show()

plt.figure(2, figsize=(7,8.5))
current_palette = sns.color_palette()
palette = sns.color_palette("RdBu_r", 5)
sns.set(font_scale=0.7)
sns.heatmap(df_final_ratios, cmap=palette, center=1, yticklabels=True)
# plt.ylabel('Semantic-Linguistic Features')
# plt.title('Language Patterns in different Personalities')
plt.savefig(config.path+'heatmap_full.png')
plt.show()

df_final_ratios.to_csv('tmp.csv')

