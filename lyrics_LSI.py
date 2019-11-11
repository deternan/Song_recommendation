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
doc = '想 把 你 寫成 一首歌 想養 一隻 貓 想要 回到 每個 場景 撥慢 每 隻 錶 我倆 在 小孩 和 大人 的 轉角 蓋 一座 城堡 我倆 好好 好 到 瘋 掉 像 找回 失散多年 雙胞 生命 再長 不過 煙火 落下 了 眼角 世界 再大 不過 你 我 凝視 的 微笑 在 所有 流逝 風景 與 人群 中 你 對 我 最好 一切 好好 是否 太好 沒有 人 知道 你 和 我 背著 空蕩 的 書包 逃避 名為 日常 的 監牢 忘記 了 要 長大 忘記 了 要 變老 忘記 了 時間 有腳 最 安靜 的 時刻 回憶 總是 最 喧囂 最 喧囂 的 狂歡 孤單 包圍 著 孤島 還以 為 馴服 想 能 陪伴 我 像 一隻 家貓 它 就 窩 在 沙發 一角 卻 不肯 睡著 你 和 我 曾經 有 滿滿的 羽毛 跳 著名 為 青鳥 的 舞蹈 不 知道 未來 不 知道 煩惱 不知道 那些 日子 會 是 那麼 少 時間 的 電影 結果 才 知道 原來 大人 已經 沒有 童謠 最後 的 叮嚀 最後 的 擁抱 我倆 紅著 眼笑 我倆 都 要 把 自己 照顧 好 好 到 遺憾 無法 打擾 好好 的 生活 好好 的 變老 好好 假裝 我 已經 把 你 忘記 '
vec_bow = dictionary.doc2bow(doc.split()) 
vec_lsi = lsi[vec_bow] 
print(vec_lsi)

# 建立索引
index = similarities.MatrixSimilarity(lsi[corpus]) 
index.save(dataSource + "model/lyrics_mayday.index")

# 計算相似度（前五名）
sims = index[vec_lsi] 
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims[:5])

lyrics = [];
fp = open(dataSource + "lyrics_word_net_mayday.dataset", encoding = 'utf8')
for i, line in enumerate(fp):
    lyrics.append(line)

# 結果輸出
for lyric in sims[:5]:
    print("\n相似歌詞：",  lyrics[lyric[0]])
    print("相似度：",  lyric[1])