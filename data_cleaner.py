from nltk.tokenize.casual import TweetTokenizer
import pandas as pd
import string
import itertools

punctuation = string.punctuation+'``'+'--'+"''"+'|'+'...'
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
translator = str.maketrans('', '', string.punctuation)


class data_cleaner():

    def flatten(x):
        if isinstance(x, list):
            return [a for i in x for a in data_cleaner.flatten(i)]
        else:
            return [x]


    #Normal Preprocess
    def preprocess1(self, listOfLists):
        # not removing URLs
        X1 = []
        i =0
        for sentence in listOfLists:
            list = sentence.split()
            sublist = []
            for word in list:
                if word.startswith('http'):
                    continue
                if '|||' in word:
                    word.replace('.','')
                    tuples = word.split('|||')
                    for tuple in tuples:
                        token1 = tokenizer.tokenize(tuple)
                        sublist.append(token1)
                elif '.' in word:
                    word.replace('.', '')
                    token = tokenizer.tokenize(word.translate(translator))
                    sublist.append(token)
                else:
                    token = tokenizer.tokenize(word.translate(translator))
                    sublist.append(token)
            output = [item for item in data_cleaner.flatten(sublist) if item not in punctuation ]
            X1.append(' '.join(output))
            print(i)
            i +=1
        return X1


    #Split the posts in more posts with respect to n_posts
    def preprocess2(self, listOfLists, labels, n_posts):
        # not removing URLs
        X1 = pd.DataFrame(columns=['type', 'post'])
        i =0
        rows_index = 0
        for sentences in listOfLists:
            #First post
            sentences = sentences.split('|||')
            #50 post
            tmp = []
            for sentence in sentences:
                sublist = ''
                for word in sentence.translate(translator).split():
                    if word.startswith('http'):
                        continue
                    x = tokenizer.tokenize(word)
                    sublist = sublist + ' '+ x[0]
                tmp.append(sublist)

            index = 0
            while index < len(tmp):
                counter = 0
                row = labels[i] + '.'
                while counter < n_posts and index < len(tmp):
                    row = row + tmp[index]
                    index +=1
                    counter +=1
                X1.loc[rows_index] = row.split('.')
                rows_index +=1
                print(rows_index)
            i += 1
            print(i)
        return X1
