from sklearn.datasets import fetch_20newsgroups
data = news.data*10


'''
https://towardsdatascience.com/a-beginners-introduction-into-mapreduce-2c912bb5e6ac

For each text in the dataset, we want to tokenize it, clean it, remove stop words and finally count the words:

'''

# stage 1, without map reduce
def clean_word(word):
    return re.sub(r'[^\w\s]','',word).lower()


def word_not_in_stopwords(word):
    return word not in ENGLISH_STOP_WORDS and word and word.isalpha()
    
    
def find_top_words(data):
    cnt = Counter()
    for text in data:
        tokens_in_text = text.split()
        tokens_in_text = map(clean_word, tokens_in_text)
        tokens_in_text = filter(word_not_in_stopwords, tokens_in_text)
        cnt.update(tokens_in_text)
        
    return cnt.most_common(10)


# soln2, real map reduce
'''
The mapper gets a text, splits it into tokens, cleans them and filters stop words and non-words, 
finally, it counts the words within this single text document. 

The reducer function gets 2 counters and merges them. 

The chunk_mapper gets a chunk and does a MapReduce on it. 

'''
def mapper(text):
    tokens_in_text = text.split()
    tokens_in_text = map(clean_word, tokens_in_text)
    tokens_in_text = filter(word_not_in_stopwords, tokens_in_text)
    return Counter(tokens_in_text)


def reducer(cnt1, cnt2):
    cnt1.update(cnt2)
    return cnt1


def chunk_mapper(chunk):
    mapped = map(mapper, chunk)
    reduced = reduce(reducer, mapped)
    return reduced


%%time
data_chunks = chunkify(data, number_of_chunks=36)
#step 1:
mapped = pool.map(chunk_mapper, data_chunks)
#step 2:
reduced = reduce(reducer, mapped)
print(reduced.most_common(10))


OUTPUT:
[('subject', 122520),
 ('lines', 118240),
 ('organization', 111850),
 ('writes', 78360),
 ('article', 67540),
 ('people', 58320),
 ('dont', 58130),
 ('like', 57570),
 ('just', 55790),
 ('university', 55440)]
 
CPU times: user 1.52 s, sys: 256 ms, total: 1.77 s
Wall time: 4.67 s



