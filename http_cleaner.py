class Mycleaner():

    def clean(self, sentence):
        if sentence.startswith('http'):
            return ""
        if sentence == '|':
            return ""
        else :
            return sentence

