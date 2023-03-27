import os
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
from collections import Counter
import pickle
import pandas as pd
import numpy as np

input_sav_file = r"C:\BTTH\CS419\exercise\term_idf.sav"
input_csv_file = r"C:\BTTH\CS419\exercise\document-term_matrix_tfidf_normalized.csv"
input_query_path = r"C:\BTTH\CS336\Cranfield\queries"
input_query_res_file = r"C:\BTTH\CS336\Cranfield\cranqrel"

def get_text_from_file(filename):
    with open(filename, encoding='UTF-8', mode='r') as f:
        text = f.read()
    return text

# preprocess the query

def preprocess_text(text):
    processed_text = text.lower()
    processed_text = processed_text.replace("’", "'")
    processed_text = processed_text.replace("“", '"')
    processed_text = processed_text.replace("”", '"')
    processed_text = re.sub(r'[^\w\s]',' ',processed_text)
    return processed_text

def get_words_from_text(text):
    ps = PorterStemmer()
    processed_text = preprocess_text(text)
    filtered_words = [word for word in processed_text.split() if word not in stopwords.words('english')]
    processed_words = []
    for i in range(len(filtered_words)):
        processed_words.append(ps.stem(filtered_words[i]))
    return processed_words

##

def get_sorted_res(query, input_sav_file, input_csv_file):

    # process the query

    query_term_frequency_dict = dict(Counter(get_words_from_text(query)))

    file = open(input_sav_file, 'rb')
    terms_idf = pickle.load(file)

    query_vector_dict = {}

    for i in list(query_term_frequency_dict.keys()):
        if i in list(terms_idf.keys()):
            query_vector_dict[i] = query_term_frequency_dict[i] * terms_idf[i]

    l = np.linalg.norm(list(query_vector_dict.values()))

    for i in list(query_vector_dict.keys()):
        query_vector_dict[i] = query_vector_dict[i] / l

    ##

    ## print results

    document_term_matrix = pd.read_csv(input_csv_file, index_col=0)

    temp = document_term_matrix.to_dict(orient='index')

    res = []

    for i in list(temp.keys()):
        keys = set(list(query_vector_dict.keys())).intersection(list(temp[i].keys()))
        dot_product = sum(query_vector_dict[k] * temp[i][k] for k in keys)
        if dot_product != 0:
            res.append((i, dot_product))
    
    res = sorted(res, key=lambda tup: tup[1], reverse=True)

    res_only = [i[0] for i in res]

    return res, res_only

def calculate_AP(res_only, res_gold):

    precision = 0
    precision_list = []

    for i in range(len(res_only)):
        if res_only[i] in res_gold[:i+1]:
            precision += 1
        precision_list.append(float(precision)/(i+1))

    temp = 0
    for i in range(len(precision_list)-1, -1, -1):
        if temp >= precision_list[i]:
            precision_list[i] = temp
        elif temp < precision_list[i]:
            temp = precision_list[i]

    precision_list_11 = []
    for i in range(11):
        precision_list_11.append(precision_list[int((len(precision_list)-1)*(i/10))])

    # temp = 0
    # for i in range(len(precision_list_11)-1, -1, -1):
    #     if temp >= precision_list_11[i]:
    #         precision_list_11[i] = temp
    #     elif temp < precision_list_11[i]:
    #         temp = precision_list_11[i]

    # print(precision_list, precision_list_11, res_only, res_gold)

    average_precision = sum(precision_list_11)/len(precision_list_11)

    return average_precision

def calculate_MAP(input_query_path,input_query_res_file,input_sav_file,input_csv_file):

    AP_list = []

    with open(input_query_res_file, 'r', encoding='UTF-8') as f:
        lines = f.read().splitlines()

    j = '1'

    res_gold = []
    res_gold_list = []

    for i in lines:
        temp = i.split()
        if temp[0] == j:
            res_gold.append(int(temp[1]))
        elif temp[0] == '.':
            res_gold_list.append(res_gold)
            break
        else:
            j = temp[0]
            res_gold_list.append(res_gold)
            res_gold = []
            res_gold.append(int(temp[1]))

    res_only_list = []

    for query_file in sorted(os.listdir(input_query_path),key=lambda x: int(os.path.splitext(x)[0])):
        print(query_file)
        filename = os.path.join(input_query_path, query_file)
        print(get_text_from_file(filename))
        res_only_list.append(get_sorted_res(get_text_from_file(filename),input_sav_file,input_csv_file)[1])

    for i in range(len(res_gold_list)):
        AP_list.append(calculate_AP(res_only_list[i],res_gold_list[i]))
    
    print(AP_list, sum(AP_list), len(AP_list))

    return sum(AP_list)/len(AP_list)

print(calculate_MAP(input_query_path,input_query_res_file,input_sav_file,input_csv_file))

