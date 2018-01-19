from nltk.tokenize.casual import TweetTokenizer
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
