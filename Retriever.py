import sys, time
from math import log10, pow, sqrt
from Posting import Posting
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from Indexer import computeWordFrequencies
from heapq import nsmallest


def get_docid_map(filename):
    docidmap = dict()
    with open(filename, 'r', encoding='utf-8') as d:
        for line in d:
            curline = line.strip().split()
            docidmap[int(curline[0])] = (float(curline[1]), curline[2])
    return docidmap


def get_byte_offsets(filename):
    offset = dict()
    with open(filename, 'r', encoding='utf-8') as o:
        for line in o:
            curline = line.strip().split()
            offset[curline[0]] = int(curline[1])

    return offset


def get_posting_list(word, index, offset):
    postings = []
    with open(index, 'r', encoding='utf-8') as d:
        d.seek(offset[word])
        curline = d.readline().strip().split()
        for post in curline[1:]:
            pair = post.split("-")
            postings.append(Posting(int(pair[0]), int(pair[1]), int(pair[2])))
    return postings


def get_tf_weight(tf):
    return 1 + log10(tf)


def get_idf_weight(N, df):
    idf = N/df
    return log10(idf)


def get_tfidf_weight(tf, N, df):
    return get_tf_weight(tf) * get_idf_weight(N, df)


def get_length(query_weights):
    length = 0
    for term in query_weights:
        length += pow(query_weights[term], 2)
    return sqrt(length)


def retrieval(query, index, f, limit, offset, docidmap):
    score = dict()
    query_weights = dict()
    posting_dict = dict()
    tag_weights = dict()
    N = len(docidmap)
    query_freq = computeWordFrequencies(query)
    for word in query_freq:
        postings = get_posting_list(word, index, offset)
        df = len(postings)
        query_weight = f(query_freq[word], N, df)
        posting_dict[word] = postings
        query_weights[word] = query_weight
    query_length = get_length(query_weights)
    for term in posting_dict:
        for post in posting_dict[term]:
            if post.docid not in score:
                score[post.docid] = 0
            score[post.docid] += (query_weights[term]/query_length) * (get_tf_weight(post.tfidf)/docidmap[post.docid][0])
            if post.docid not in tag_weights:
                tag_weights[post.docid] = 0
            tag_weights[post.docid] += post.tier

    scale = len(query_freq) * 6
    for d in score:
        score[d] = (tag_weights[d]/scale) + score[d]

    heap = [(-value, key) for key, value in score.items()]
    result = nsmallest(limit, heap)
    result = [(key, -value) for value, key in result]

    return result


def main():
    docidmap = get_docid_map("docidmap.txt")
    offset = get_byte_offsets("offset.txt")
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    while True:
        user_input = input("Enter your query: ")
        if user_input == "!q":
            break
        start_time = time.time()
        query = tokenizer.tokenize(user_input.lower())
        stemmed_query = []
        for word in query:
            stemmed_query.append(stemmer.stem(word))
        try:
            result = retrieval(stemmed_query, "index.txt", get_tfidf_weight, 5, offset, docidmap)
            print("--- %s seconds ---" % (time.time() - start_time))
            for (docid, score) in result:
                print(f"{docidmap[docid][1]} {score}")
        except KeyError:
            print("Query is not searchable. Try another query.")


if __name__ == '__main__':
    main()
