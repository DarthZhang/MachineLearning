import pandas as pd


import config


path = config.path
type_1 = []  # I or E
type_2 = []  # S or N
type_3 = []  # T or F
type_4 = []  # J or P


df = pd.read_csv(path)


def split_logic(composite_type):
    global type_1
    global type_2
    global type_3
    global type_4
    types = list(composite_type)
    type_1.append(types[0])
    type_2.append(types[1])
    type_3.append(types[2])
    type_4.append(types[3])


for composite_type in df['type']:
    split_logic(composite_type)

new_dict_df = {'type_1': type_1,
               'type_2': type_2,
               'type_3': type_3,
               'type_4': type_4}

new_df = pd.DataFrame(new_dict_df)
new_df['posts'] = df['posts']


# print new_df

new_df.to_csv(config.output_path)