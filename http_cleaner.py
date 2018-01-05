class Mycleaner():

    def clean(self, sentence):
        if sentence.startswith('http'):
            return ""
        else :
            return sentence

