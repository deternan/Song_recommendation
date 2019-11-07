# coding=utf8

#
# get keywords
#


import jieba
import jieba.analyse

#jieba.set_dictionary("../jieba/extra_dict/dict.txt.tra.big")
jieba.set_dictionary("../jieba/extra_dict/dict.txt.big")


with open("../data/single_song.txt", "r") as f1:
    for line in f1:
        words = jieba.analyse.extract_tags(line,10, withWeight=True)        
        for tag, weight in words:
             print(tag + "    " + str(int(weight * 10000)))
            #print(",".join(words))
f1.close()