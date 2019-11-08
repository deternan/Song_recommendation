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
doc = '如果我們不曾相遇 我會是在哪裡 如果我們從不曾相識 不存在這首歌曲 每秒都活著 每秒都死去 每秒都問著自己 誰不曾找尋 誰不曾懷疑 茫茫人生奔向何地 那一天 那一刻 那個場景 你出現在我生命 從此後 從人生 重新定義 從我故事裡甦醒 如果我們不曾相遇 你又會在哪裡 如果我們從不曾相識 人間又如何運行 曬傷的脫皮 意外的雪景 與你相依的四季 蒼狗又白雲 身旁有了你 匆匆輪迴又有何懼 那一天 那一刻 那個場景 你出現在我生命 每一分 每一秒 每個表情 故事都充滿驚奇 偶然與巧合 舞動了蝶翼 誰的心頭風起 前仆而後繼 萬千人追尋 荒漠唯一菩提 是擦身相遇 或擦肩而去 命運猶如險棋 無數時間線 無盡可能性 終於交織向你 那一天 那一刻 那個場景 你出現在我生命 未知的 未來裡 未定機率 然而此刻擁有你 某一天 某一刻 某次呼吸 我們終將再分離 而我的 自傳裡 曾經有你 沒有遺憾的詩句 詩句裡 充滿感激 如果我們不曾相遇 我會是在哪裡 如果我們從不曾相識 不存在這首歌曲'
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