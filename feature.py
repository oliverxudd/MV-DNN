# TODO
# 1 feature: remove {'A', 'the'...} from unigram

import os
import os.path as osp
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.linalg import blas


DATA_DIR = "/home/xudd/testbed/MV-DNN/arnetminer-data"

domain_names = ['Database', 'Medical Informatics', 'Visualization', 'Data Mining', 'Theory']
four_pair_of_domains = [['Data Mining', 'Theory'], ['Medical Informatics', 'Data Mining'],
                        ['Medical Informatics', 'Data Mining'], ['Visualization', 'Data Mining']]

FIELD_DEMILITER = '\t'
TITLE_FIELD_IDX = 1
NAME_INDEX = 2
YEAR_FILED_IDX = 3
ABSTRACT_FILED_IDX = 4
TRAIN_YEAR_UPPERLIMIT = 2000


"""获取训练时间段内，两个域内的作者名单、作者-论文矩阵、论文-作者矩阵、作者-特征矩阵、全部token构成的字典"""
def get_train_features(src_domain, tgt_domain):
    srcfile_loc = osp.join(DATA_DIR, src_domain+'.txt')
    tgtfile_loc = osp.join(DATA_DIR, tgt_domain+'.txt')

    # 首先收集字典，用于生成特征向量
    vocab_all = create_vocab_all(srcfile_loc, tgtfile_loc)

    src_authors, src_author_paper = get_train_author_paper(srcfile_loc)
    tgt_authors, tgt_author_paper = get_train_author_paper(tgtfile_loc)

    src_paper_token = get_train_paper_token(srcfile_loc, vocab_all)
    tgt_paper_token = get_train_paper_token(tgtfile_loc, vocab_all)

    src_features = blas.sgemm(alpha=1.0, a=src_author_paper, b=src_paper_token.T, trans_b=True)
    tgt_features = blas.sgemm(alpha=1.0, a=tgt_author_paper, b=tgt_paper_token.T, trans_b=True)

    feature_dict = {}
    feature_dict['src_authors'] = src_authors
    feature_dict['tgt_authors'] = tgt_authors
    feature_dict['src_paper_author'] = src_author_paper.T
    feature_dict['tgt_paper_author'] = tgt_author_paper.T
    feature_dict['src_features'] = src_features
    feature_dict['tgt_features'] = tgt_features
    feature_dict['vocab_all'] = vocab_all

    return feature_dict


def create_vocab_all(srcfile, tgtfile):
    src_tokens = get_tokens(srcfile)
    tgt_tokens = get_tokens(tgtfile)
    common_tokens_duplicate = src_tokens + tgt_tokens
    common_tokens = sorted(list(set(common_tokens_duplicate)))
    return common_tokens


def get_tokens(file_loc):
    corpus = get_corpus(file_loc, [TITLE_FIELD_IDX, ABSTRACT_FILED_IDX])

    vectorizer = CountVectorizer(lowercase=False)
    X = vectorizer.fit_transform(corpus)
    tokens = vectorizer.get_feature_names()
    return tokens


def get_train_author_paper(file_loc):
    corpus = get_corpus(file_loc, [NAME_INDEX])
    strain_ind, stest_ind = get_train_test_index(file_loc)
    train_corpus = list(np.array(corpus)[strain_ind])

    vectorizer = CountVectorizer(lowercase=False, analyzer='word', token_pattern=r"[\w\.\ \-()]*\w+")
    X = vectorizer.fit_transform(train_corpus)
    authors = vectorizer.get_feature_names()
    paper_author = X.toarray()
    author_paper = paper_author.T

    return authors, author_paper


def get_corpus(file_loc, field_inds):
    corpus = []
    lines = open(file_loc).readlines()
    for line in lines:
        line_content = ''
        for i, ind in enumerate(field_inds):
            field = line.split(FIELD_DEMILITER)[ind]
            line_content += field
            if i < len(field_inds)-1:
                line_content += ' '

        corpus.append(line_content)

    return corpus


def get_train_test_index(fileloc):
    train_ind = []
    test_ind = []
    lines = open(fileloc).readlines()
    for i, line in enumerate(lines):
        year = int(line.split(FIELD_DEMILITER)[YEAR_FILED_IDX])
        if  year <= TRAIN_YEAR_UPPERLIMIT:
            train_ind.append(i)
        else:
            test_ind.append(i)

    return train_ind, test_ind


def get_train_paper_token(file_loc, vocab_all):
    corpus = get_corpus(file_loc, [TITLE_FIELD_IDX, ABSTRACT_FILED_IDX])
    strain_ind, stest_ind = get_train_test_index(file_loc)
    train_corpus = list(np.array(corpus)[strain_ind])

    vectorizer = CountVectorizer(lowercase=False, vocabulary=vocab_all)
    X = vectorizer.fit_transform(train_corpus)
    paper_token = X.toarray()

    return paper_token

