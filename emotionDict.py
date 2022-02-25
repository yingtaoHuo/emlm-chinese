import pandas as pd
from ltp import LTP

class emotionDict(object):
    def __init__(self):
        super(emotionDict, self).__init__()
        self.dictionary = pd.read_csv('emtion_dict.csv')

    def evaluate(self, word):
        try:
            target = self.dictionary[self.dictionary.词语 == word].index[0]
            target = self.dictionary.loc[[target]]
        except IndexError:
            return None

        return (target.词语.values[0], target.情感分类.values[0], target.强度.values[0], target.极性.values[0])

def clean():
    ltp = LTP()
    dictionary = pd.read_excel('情感词汇本体.xlsx')
    num = 0 
    for i in dictionary.itertuples():
        tokens_ = []
        tokens_.append(i[1])
        try:
            tokens = ltp.seg(tokens_)
            # toknes = tokens[0]
            if len(tokens[0][0]) != 1:
                #　print(tokens[0], i[1])
                dictionary.drop([i[0]], inplace=True)
                num += 1
        except:
            print(tokens_)
            # print(i[0], i[1], tokens[0][0])
            # input()
        # else:
        #     dictionary.drop([i[0]], inplace=True)
        #     print(i[0], i[1])
        #     print(tokens[0][0], len(tokens[0][0]))
    print(num)
    # for i in dictionary.itertuples():
    #     print(i[1])``
    #     input()
    dictionary.to_csv('emtion_dict.csv')


if __name__ == '__main__':
    # arguments = docopt(__doc__, version = __version__)
    # clean()
    handler = emotionDict()
    ltp = LTP()
    tokens = ltp.seg(['手脚不干净'])
    result = handler.evaluate('手脚不干净')
    print(tokens[0], result)