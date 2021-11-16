class Posting:
    def __init__(self, docid, tfidf, tier):
        self.docid = docid
        self.tfidf = tfidf
        self.tier = tier

    def __repr__(self):
        return "(%s, %s)" % (self.docid, self.tfidf)

    def __str__(self):
        return "(%s, %s)" % (self.docid, self.tfidf)

    def __lt__(self, other):
        return self.docid < other.docid
