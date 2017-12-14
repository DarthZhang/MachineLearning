import pandas as pd


import config


path = config.path
dict_types = {}


df = pd.read_csv(path)

for type in df['type']:
    if type in dict_types:
        count = dict_types[type]
        count = count + 1
        dict_types[type] = count
    else:
        dict_types[type] = 1


# Average should be 8675/16 = 542.1875
for key, value in dict_types.items():
    print key
    print value