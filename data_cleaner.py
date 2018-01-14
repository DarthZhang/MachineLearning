from nltk.tokenize.casual import TweetTokenizer
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

class data_cleaner():

    def clean(self, sentence):
        if sentence.startswith('http'):
            return ""
        else:
            return sentence

    def preprocess1(self, listOfLists):
        # not removing URLs
        X1 = []
        for i in listOfLists:
            i = i.replace('|||', ' ')
            tokens = tokenizer.tokenize(i)
            X1.append(' '.join(tokens))
        return X1

    def preprocess2(self, listOfLists): #CRASHES!
        # removing URLs
        X1 = []
        for i in listOfLists:
            i = i.replace('|||', ' ')
            tmp_tokens = []
            tokens = tokenizer.tokenize(i)
            for j in tokens:
                tmp = self.clean(j)
                tmp_tokens.append(tmp)
                X1.append(' '.join(tmp_tokens).replace('  ', ' '))
        return X1

#data = data_cleaner()
#a = ["my first post as a travel blogger in ohlala magazine!! It is one of the biggest - if not the biggest and most read magazine by women in argentina, cannot believe i am collaborating for them from sweden! great way to close the year. if you read some spanish check it out, if not you can use facebook translate! more in my travel instagram. my first post as a travel blogger in ohlala magazine! it is one of the biggest - if not the biggest and most read magazine by women in argentina, cannot believe i am collaborating for"]

#a = data.preprocess1(a)
#print(a)