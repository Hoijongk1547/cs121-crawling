import os, math
import re
import shelve
from json import load
from urllib.parse import urldefrag
from nltk.tokenize import RegexpTokenizer  # TOKENIZE WORDS EXCEPT PUNCTUATIONS
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from Posting import Posting
from contextlib import ExitStack
from heapq import merge


count = 1


def walks_dirs(file_path, batch_size):
    for dirpath, dirnames, files in os.walk(file_path):
        print(f'Found directory: {dirpath}')
        for i in range(0,len(files), batch_size):
            yield [os.path.join(dirpath, file_name) for file_name in files[i:i+batch_size]]


def write_doc_id(urldict):
    with open('docidmap.txt', 'w', encoding='utf-8') as r:
        for docid in sorted(urldict.keys()):
            r.write(f"{docid} {urldict[docid][0]} {urldict[docid][1]}\n")


def sort_and_write(index):
    global count
    save_path = f"Chunks/chunk_{count}"
    with open(save_path, 'w', encoding='utf-8') as r:
        for word in sorted(index.keys()):
            line = word
            for post in index[word]:
                line += f" {post.docid}-{post.tfidf}-{post.tier}"
            r.write(line + '\n')
        count += 1


def get_tokens(soup):
    content = soup.get_text(separator=" ")  # RAW CONTENT
    tokenizer = RegexpTokenizer(r'\w+')
    clearContent = tokenizer.tokenize(str(content).lower())  # TOKENIZE WORDS ONLY  #https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
    stemmer = PorterStemmer()
    stemmedText = []
    for t in clearContent:
        stemmedText.append(stemmer.stem(t))

    freq = computeWordFrequencies(stemmedText)
    length = get_length(freq)

    bold = set()
    h1 = set()
    h2 = set()
    h3 = set()
    titles = set()

    for content in soup.find_all(["b", "strong"]):
        for word in tokenizer.tokenize(content.get_text(separator=" ").lower()):
            word = stemmer.stem(word)
            bold.add(word)
    for content in soup.find_all("h1"):
        for word in tokenizer.tokenize(content.get_text(separator=" ").lower()):
            word = stemmer.stem(word)
            h1.add(word)
    for content in soup.find_all("h2"):
        for word in tokenizer.tokenize(content.get_text(separator=" ").lower()):
            word = stemmer.stem(word)
            h2.add(word)
    for content in soup.find_all("h3"):
        for word in tokenizer.tokenize(content.get_text(separator=" ").lower()):
            word = stemmer.stem(word)
            h3.add(word)
    for content in soup.find_all("title"):
        for word in tokenizer.tokenize(content.get_text(separator=" ").lower()):
            word = stemmer.stem(word)
            titles.add(word)

    for term in freq:
        tier = 1
        tf = freq[term]
        if term in bold:
            tier = 2
        if term in h3:
            tier = 3
        if term in h2:
            tier = 4
        if term in h1:
            tier = 5
        if term in titles:
            tier = 6
        freq[term] = (tf, tier)
    return freq, length


def computeWordFrequencies(Token: list) -> dict:
    # The time complexity is O(N)
    # This is due to going through the list once and counting the frequency of the word.
    frequencies = dict()  # O(1) For basic construction
    for word in Token:  # O(N) For word in list
        if word in frequencies:  # O(1)
            frequencies[word] += 1  # O(1)
        else:  # O(1) Worst case scenario should be the same
            frequencies[word] = 1  # O(1)
    return frequencies


def get_tf_weight(tf):
    return 1 + math.log10(tf)


def get_length(freq):
    length = 0
    for term in freq:
        length += math.pow(get_tf_weight(freq[term]), 2)
    return math.sqrt(length)


def get_index(file_path):
    index = dict()
    docid = 0
    urldict = dict()

    for batch in walks_dirs(file_path, 10000):
        for file_name in batch:
            with open(file_name) as json_file:
                print(file_name)
                data = load(json_file)  # 'url: https--- ' + 'content(html)'
                soup = BeautifulSoup(data["content"], "lxml")  # extract content(html) in data
                url = urldefrag(data["url"])[0]  # extract urls in data
                if any(url in pair for pair in urldict.values()):
                    continue
                docid += 1
                freq, length = get_tokens(soup)
                urldict[docid] = (length, url)
                for token in freq:
                    if token not in index:
                        index[token] = []
                    index[token].append(Posting(docid, freq[token][0], freq[token][1]))
        sort_and_write(index)
        index.clear()
    write_doc_id(urldict)


def merge_files(write_file, file_path):
    filelist = []
    with ExitStack() as stack, open(write_file, 'w', encoding='utf-8') as final_file:
        for dirpath, dirnames, files in os.walk(file_path):
            for file_name in files:
                chunk = os.path.join(dirpath, file_name)
                filelist.append(stack.enter_context(open(chunk, encoding='utf-8')))
        final_file.writelines(merge(*filelist))


def compress_file(read_file, write_file):
    current = ''
    postlist = []
    with open(read_file, 'r', encoding='utf-8') as partial, open(write_file, 'w', encoding='utf-8') as i:
        for line in partial:
            currline = line.strip().split()
            if current == '':
                current = currline[0]
                for post in currline[1:]:
                    pair = post.split("-")
                    postlist.append(Posting(int(pair[0]), int(pair[1]), int(pair[2])))
            elif currline[0] == current:
                second = []
                for post in currline[1:]:
                    pair = post.split("-")
                    second.append(Posting(int(pair[0]), int(pair[1]), int(pair[2])))
                postlist = list(merge(postlist,second))
            else:
                towrite = current
                for post in postlist:
                    towrite += f" {post.docid}-{post.tfidf}-{post.tier}"
                i.write(towrite + '\n')
                current = currline[0]
                postlist = []
                for post in currline[1:]:
                    pair = post.split("-")
                    postlist.append(Posting(int(pair[0]), int(pair[1]), int(pair[2])))
        towrite = current
        for post in postlist:
            towrite += f" {post.docid}-{post.tfidf}-{post.tier}"
        i.write(towrite + '\n')


def create_index_offset(read_file, write_file):
    with open(read_file, 'r', encoding='utf-8') as d, open(write_file, 'w', encoding='utf-8') as o:
        while True:
            index = d.tell()
            line = d.readline()
            if not line:
                break
            line = line.strip().split()
            o.write(f"{line[0]} {index}\n")


def main():
    get_index('DEV')
    print("Merging files into Index File")
    merge_files('combined.txt', 'Chunks')
    print("Compressing Index File")
    compress_file('combined.txt', 'index.txt')
    print("Creating Index Offset File")
    create_index_offset('index.txt', 'offset.txt')
    print("Finished creating Index File and it's Auxiliary Files")


if __name__ == '__main__':
    main()
