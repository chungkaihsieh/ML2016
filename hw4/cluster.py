#!/usr/bin/env python

import numpy as np
import scipy
import pandas as pd
import math
import time
# import theano
# import keras
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster, datasets
from sklearn import decomposition
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import csv


'''
Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python hw4.py
'''


def test(label, check_index, file_name):
    nb_check = len(check_index)
    writer = open(file_name, "w")
    writer.write("ID,Ans\n")
    for i in range(nb_check):
        tmp = check_index[i]
        title_1 = tmp[1]
        title_2 = tmp[2]
        if(label[title_1] == label[title_2]):
            str_tmp = str(tmp[0]) + "," + str(1) + "\n"
        else:
            str_tmp = str(tmp[0]) + "," + str(0) + "\n"

        writer.write(str_tmp)
    writer.close()


'''
read data
'''


if __name__ == '__main__':

    path = sys.argv[1]
    file_name = sys.argv[2]

    title_file_name = path + "title_StackOverflow.txt"
    f = open(title_file_name)
    title_data = f.readlines()
    f.close()

    title_data = np.array(title_data)
    # print("title_data.shape:" + str(title_data.shape))
    # print(title_data[0])
    # print(title_data[19999])

    check_file_name = path +  "check_index.csv"
    check_index_tmp = pd.read_csv(check_file_name, sep=',', encoding='latin1')
    column_name = ['ID', 'x_ID', 'y_ID']
    check_index = check_index_tmp.as_matrix(columns=column_name)

    '''
	Bag of Word
	'''

    # #bi-grams vectorize : make text into vector

    # bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
    # 					token_pattern=r'\b\w+\b', min_df = 1)

    # print("test bi-gram")
    # analyze = bigram_vectorizer.build_analyzer()
    # value = analyze('Bi-grams are cool!') == ( ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])
    # print("analyze('Bi-grams are cool!') == ( ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool']) :" + str(value))

    # title_2gram_vector = bigram_vectorizer.fit_transform(title_data).toarray()
    # print("title_2gram_vector.shape :" + str(title_2gram_vector.shape) )

    # feature_index = bigram_vectorizer.vocabulary_.get('is this')
    # print("bi-gram to 'is this' token :" + str(feature_index))

    print("sentence to vector")
    # vectorizer = TfidfVectorizer(stop_words = 'english',  lowercase= True, min_df = 2, max_df=0.8)
    vectorizer = TfidfVectorizer(
        stop_words='english', lowercase=True, min_df=2, max_df=0.7)

    # vectorizer = TfidfVectorizer(ngram_range=(1, 2),
    # 				token_pattern=r'\b\w+\b',min_df = 1)

    # print("test bi-gram")
    # analyze = vectorizer.build_analyzer()
    # value = analyze('Bi-grams are cool!') == ( ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])
    # print("analyze('Bi-grams are cool!') == ( ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool']) :" + str(value))

    # title_vector =  vectorizer.fit_transform(title_data).toarray()
    title_vector = vectorizer.fit_transform(title_data)

    print("title_vector.shape:" + str(title_vector.shape))

    '''
	SVDreduction
	'''
    dim_reduction = 20
    svd = TruncatedSVD(dim_reduction)

    title_vector_reduction = svd.fit_transform(title_vector)

    # normalizer
    normalizer = Normalizer(norm='l2', copy=True)
    title_vector_reduction = normalizer.fit_transform(title_vector_reduction)

    print("after dim reuduction title_vector.shape : " +
          str(title_vector_reduction.shape))

    '''
	k-means
	'''

    tStart = time.time()

    print("K-means cluster")

    # K-means cluster
    k_means = cluster.KMeans(n_clusters=40)
    k_means.fit(title_vector_reduction)
    print("k_means.labels_.shape :" + str(k_means.labels_.shape))

    print("predict")
    test(k_means.labels_, check_index, file_name)

    tEnd = time.time()

    print("total cost :" + str(tEnd - tStart) + " sec")
