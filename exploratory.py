import pandas as pd


import config


path = config.path
dict_types = {}
count_I = 0
count_E = 0
count_S = 0
count_N = 0
count_T = 0
count_F = 0
count_J = 0
count_P = 0
dict_type1 = {}  # I or E
dict_type2 = {}  # S or N
dict_type3 = {}  # T or F
dict_type4 = {}  # J or P

df = pd.read_csv(path)

for type in df['type']:
    if type in dict_types:
        count = dict_types[type]
        count = count + 1
        dict_types[type] = count
    else:
        dict_types[type] = 1




# Average should be 8675/16 = 542.1875
print 'Composite types counts:'
for key, value in dict_types.items():
    print key + ' : ' + str(value)


print '\n\n\nIndividual types count:'
for key, value in dict_types.items():
    letters = list(key)
    if letters[0] == 'I':
        count_I = count_I + value
    else:
        count_E = count_E + value
    if letters[1] == 'S':
        count_S = count_S + value
    else:
        count_N = count_N + value
    if letters[2] == 'T':
        count_T = count_T + value
    else:
        count_F = count_F + value
    if letters[3] == 'J':
        count_J = count_J + value
    else:
        count_P = count_P + value

print 'I : ' + str(count_I) + ' , E : ' + str(count_E)
print 'S : ' + str(count_S) + ' , N : ' + str(count_N)
print 'T : ' + str(count_T) + ' , F : ' + str(count_F)
print 'J : ' + str(count_J) + ' , P : ' + str(count_P)