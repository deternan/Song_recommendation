# coding=utf8

import numpy as np
import jieba.analyse
import jieba
import codecs

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
from gensim import corpora, models, similarities

jiebadataSource = './jieba/extra_dict/'
dataSource = './data/'


# 移除常見字
with open(jiebadataSource + "stop_words.txt", encoding = 'utf8') as f:
    stop_word_content = f.readlines()
stop_word_content = [x.strip() for x in stop_word_content] #strip: 移除頭尾空格、中間不會
stop_word_content = " ".join(stop_word_content)

# 建立本次文檔的語料庫(字典)
dictionary = corpora.Dictionary(document.split() for document in open(dataSource + "lyrics_word_net_mayday.dataset", encoding = 'utf8'))
stoplist = set(stop_word_content.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id] 
dictionary.filter_tokens(stop_ids) 
dictionary.compactify() 
dictionary.save(dataSource + "lyrics_mayday.dict")


texts = [[word for word in document.split() if word not in stoplist]
         for document in open(dataSource + "lyrics_word_net_mayday.dataset", encoding = 'utf8')]
         
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize(dataSource + "model/lyrics_mayday.mm", corpus) 

# 載入語料庫
if (os.path.exists(dataSource + "lyrics_mayday.dict")):
    dictionary = corpora.Dictionary.load(dataSource + "lyrics_mayday.dict")
    corpus = corpora.MmCorpus(dataSource + "model/lyrics_mayday.mm")
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")
    

# 創建 LSI model
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
corpus_lsi = lsi[corpus_tfidf] 
lsi.save(dataSource + 'model/lyrics_mayday.lsi')
corpora.MmCorpus.serialize(dataSource + 'model/lsi_corpus_mayday.mm', corpus_lsi)

# print("LSI topics:")
# lsi.print_topics(5)

# Similarity 文本相似度分析
doc = '志明 真正 不知道 要安 怎麼 為 什麼 情人 不願閣 再 相偎 春嬌 已經 早就 無 在 聽 講這多 其實 攏總 攏 無卡 抓 走 到 淡水 的 海岸 兩個 人 的 愛情 已經 無人 看 已經 無人 聽 我 跟 你 最好 就 到 這 你 對 我 已經 沒 感覺 到 這凍止 你 也 免愛我 我 跟 你 最好 就 到 這 你 對 我 已經 沒 感覺 麥閣 傷感 麥閣 我 這愛你 你 沒愛我 志明 心情 真正 有 影寒 風 這大 你 也 真正 攏 沒心肝 春嬌 你 哪無要 和 我播 這 齣 電影 咱 就 走 到 這位 準抵 煞 走 到 淡水 的 海岸 兩個 人 的 愛情 已經 無人 看 已經 無人 聽 我 跟 你 最好 就 到 這 你 對 我 已經 沒 感覺 到 這凍止 你 也 免愛我 我 跟 你 最好 就 到 這 你 對 我 已經 沒 感覺 麥閣 傷感 麥閣 我 這愛你 你 沒愛我 我 跟 你 最好 就 到 這 你 對 我 已經 沒 感覺 到 這凍止 你 也 免愛我 我 跟 你 最好 就 到 這 你 對 我 已經 沒 感覺 麥閣 傷感 麥閣 我 這愛你 你 沒愛我 我 跟 你 最好 就 到 這'
vec_bow = dictionary.doc2bow(doc.split()) 
vec_lsi = lsi[vec_bow] 
print(vec_lsi)

# 建立索引
index = similarities.MatrixSimilarity(lsi[corpus]) 
index.save(dataSource + "model/lyrics_mayday.index")

# 計算相似度（前五名）
sims = index[vec_lsi] 
sims = sorted(enumerate(sims), key=lambda item: -item[1])
#print(sims[:11])

lyrics = [];
#fp = open(dataSource + "lyrics_word_net_mayday.dataset", encoding = 'utf8')
fp = open(dataSource + "lyrics_mayday.txt", encoding = 'utf8')
for i, line in enumerate(fp):
    lyrics.append(line)

# 結果輸出
for lyric in sims[:11]:
    print("\n相似歌詞：",  lyrics[lyric[0]])
    print("相似度：",  lyric[1])