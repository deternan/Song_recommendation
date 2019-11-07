# coding=utf8

#
# LSI (Latent Semantic Indexing)
#
# Reference
# https://github.com/youngmihuang/lyrics_application/blob/master/lyrics_2.ipynb
#

import matplotlib.pyplot as plt
import numpy as np
import jieba.analyse
import jieba
import codecs
import os
from gensim import corpora, models, similarities

jieba.set_dictionary("../jieba/extra_dict/dict.txt.big")

        

# 載入同義字
# word_net = []
# with open("../jieba/extra_dict/lyrics_word_net.dataset", "r", encoding = 'utf8') as f1:
#     for line in f1:
#         word_net.append(line)
# 
# word_net = sorted(set(word_net))
# word_net_dic = {}
# 
# for word in word_net:
#     word_s = word.split()
#     word_net_dic[word_s[0]] = word_s[1]
# 
# # 將資料處理後的檔案存檔
# wf = open("../data/lyrics_word_net_mayday.txt", "w", encoding = 'utf8')
# 
# with open("../jieba/extra_dict/lyrics_word_net.dataset", "r", encoding = 'utf8') as f2:
#     for line in f2:
#         line_words = line.split()
#         line_lyrics = ""
#         for line_word in line_words:
#             if line_word in word_net_dic:
#                 line_lyrics = line_lyrics + word_net_dic[line_word] + ' '
#             else:
#                 line_lyrics = line_lyrics + line_word + ' '        
#         wf.write(line_lyrics+"\n")
# 
# wf.close()

#################################### LSI

# 移除常見字
with open("../jieba/extra_dict/stop_words.txt", encoding = 'utf8') as f:
    stop_word_content = f.readlines()
stop_word_content = [x.strip() for x in stop_word_content]
stop_word_content = " ".join(stop_word_content)

# 建立本次文檔的語料庫(字典)
# 將文檔裡的詞袋給予編號
dictionary = corpora.Dictionary(document.split() for document in open("../data/lyrics_word_net_mayday.txt", encoding = 'utf8'))
stoplist = set(stop_word_content.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]                          #dictionary.token2id: 代表什麼字詞對應到什麼id，有幾個id就代表有幾維向量空間
dictionary.filter_tokens(stop_ids)                                       # 移除停用字
dictionary.compactify() #remove faps in id sequence after worfs that were removed
dictionary.save("../data/lyrics_mayday.dict")

# 查看：序列化的結果
# for word,index in dictionary.token2id.items(): 
#     print(word +" id:"+ str(index))

texts = [[word for word in document.split() if word not in stoplist]
         for document in open("../data/lyrics_word_net_mayday.txt", encoding = 'utf8')]

# 移除只出現一次的字詞
# from collections import defaultdict
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1

# texts = [[token for token in text if frequency[token] > 1]
#          for text in texts]

# 將 corpus 序列化
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize("../data/model/lyrics_mayday.mm", corpus) # Corpus in Matrix Market format 

# 載入語料庫
if (os.path.exists("../data/lyrics_mayday.dict")):
    dictionary = corpora.Dictionary.load("../data/lyrics_mayday.dict")
    corpus = corpora.MmCorpus("../data/model/lyrics_mayday.mm") # 將數據流的語料變為內容流的語料
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")

# 創建 tfidf model
tfidf = models.TfidfModel(corpus)
# 轉為向量表示
corpus_tfidf = tfidf[corpus]

# 創建 LSI model
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
corpus_lsi = lsi[corpus_tfidf] # LSI潛在語義索引
lsi.save('../data/model/lyrics_mayday.lsi')
corpora.MmCorpus.serialize('../data/model/lsi_corpus_mayday.mm', corpus_lsi)
print("LSI topics:")
lsi.print_topics(5)

# 查看：每一首歌在各主題的佔比計算
for doc in corpus_lsi:
    print(doc)
