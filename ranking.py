import math
import string
import csv

file = 'wine.csv'

#functions for both formulas

def terms(d):
    #get number of terms in doc d
    return len(d["description"].split())

def frequency(d,t):
    #get number of times term t occurs in document d
    count = 0
    for term in d['description'].split():
        if term == t:
            count += 1
    return count

def containing(t):
    #get number of docs that contain term t
    count = 0
    corpus = csv.DictReader(open(file))
    for doc in corpus:
        if t in doc['description'].split():
            count += 1
    return count

def sortFunc(e):
    return e[-1]


#functions for TF-IDF

def tf(d, t):
    return math.log(1+(frequency(d,t)/terms(d)))

def relevance(d, Q, T):
    terms = Q.split()
    sum = 0
    for term in terms:
        docs_containing = T[term]
        if docs_containing > 0:
            sum += (tf(d,term)/docs_containing)

    return sum


def tf_idf(query, k):
    for c in string.punctuation:
        query = query.replace(c,'')

    terms = {}
    for term in query.split():
        terms[term] = containing(term)

    corpus = csv.DictReader(open(file))
    results = []
    for doc in corpus:
        entry = (doc['id'], relevance(doc, query, terms))
        #print(entry)
        if entry[-1] > 0:
            results.append(entry)


    results.sort(reverse = True, key = sortFunc)
    results = results[:k]
    return results

#functions for BM25

#constants
k1 = 1.2
k2 = 500
b = 0.75

def total_docs():
    #return number of docs in corpus
    corpus = csv.DictReader(open(file))
    count = 0
    for doc in corpus:
        count += 1
    return count

def ave_length():
    #get average document length
    corpus = csv.DictReader(open(file))
    total = 0
    count = 0
    for doc in corpus:
        count += 1
        total += len(doc['description'].split())
        
    return total/count

def query_frequency(Q,t):
    Q = Q.split()
    count = 0
    for term in Q:
        if term == t:
            count += 1
    return count

def bm25_idf(t, c, tD):
    return math.log((tD - c + .5)/(c + .5))

def bm25_tf(d, t, aL):
    return ((k1+1)*frequency(d,t))/((k1*(1-b) + b*(terms(d)/aL)) + frequency(d,t))

def bm25_qtf(Q, t):
    return (((k2+1)*query_frequency(Q,t)) / (k2 + query_frequency(Q,t)))

def bm25_score(d, Q, T, aL, tD):
    terms = Q.split()
    sum = 0
    for term in terms:
        c = T[term]
        sum += bm25_idf(term, c, tD) * bm25_tf(d, term, aL) * bm25_qtf(Q, term)
    return sum

def bm25(query, k):
    for c in string.punctuation:
        query = query.replace(c,'')

    terms = {}
    for term in query.split():
        terms[term] = containing(term)

    aveLen = ave_length()
    totalDocs = total_docs()

    corpus = csv.DictReader(open(file))
    results = []
    for doc in corpus:
        entry = (doc['id'], bm25_score(doc, query, terms, aveLen, totalDocs))
        #print(entry)
        if entry[-1] > 0:
            results.append(entry)


    results.sort(reverse = True, key = sortFunc)
    results = results[:k]
    return results