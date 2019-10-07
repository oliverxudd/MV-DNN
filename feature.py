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


def get_features(src_domain, tgt_domain):
    srcfile_loc = osp.join(DATA_DIR, src_domain+'.txt')
    tgtfile_loc = osp.join(DATA_DIR, tgt_domain+'.txt')

    src_names, src_name_line_count  = get_names_counts(srcfile_loc)
    tgt_names, tgt_name_line_count  = get_names_counts(tgtfile_loc)
    common_tokens = get_common_tokens(srcfile_loc, tgtfile_loc)

    src_line_token_count = get_tokens_counts(srcfile_loc, common_tokens)
    tgt_line_token_count = get_tokens_counts(tgtfile_loc, common_tokens)

    src_features = blas.sgemm(alpha=1.0, a=src_name_line_count, b=src_line_token_count.T, trans_b=True)
    tgt_features = blas.sgemm(alpha=1.0, a=tgt_name_line_count, b=tgt_line_token_count.T, trans_b=True)

    feature_dict = {}
    feature_dict['src_domain_authors'] = src_names
    feature_dict['tgt_domain_authors'] = tgt_names
    feature_dict['src_feature_matrix'] = src_features
    feature_dict['tgt_feature_matrix'] = tgt_features
    feature_dict['vocab_tokens'] = common_tokens
    feature_dict['src_paper_author_matrix'] = src_name_line_count.T
    feature_dict['tgt_paper_author_matrix'] = tgt_name_line_count.T

    return feature_dict


def get_names_counts(file_loc):
    FIELD_DEMILITER = '\t'
    NAME_INDEX = 2
    corpus = get_corpus(file_loc, [NAME_INDEX])

    vectorizer = CountVectorizer(lowercase=False, analyzer='word', token_pattern=r"[\w\.\ \-()]*\w+")
    X = vectorizer.fit_transform(corpus)
    names = vectorizer.get_feature_names()
    line_name_counts = X.toarray()
    name_line_counts = line_name_counts.T

    return (names, name_line_counts)


def get_corpus(file_loc, field_inds):
    FIELD_DEMILITER = '\t'
    corpus = []
    with open(file_loc) as f:
        for line in f:
            line_content = ''
            for iter, ind in enumerate(field_inds):
                field = line.split(FIELD_DEMILITER)[ind]
                if iter < len(field_inds)-1:
                    line_content += ' '
                line_content += field
            corpus.append(line_content)

    return corpus


def get_common_tokens(srcfile, tgtfile):
    src_tokens = get_tokens(srcfile)
    tgt_tokens = get_tokens(tgtfile)
    common_tokens_duplicate = src_tokens + tgt_tokens
    common_tokens = sorted(list(set(common_tokens_duplicate)))
    return common_tokens


def get_tokens(file_loc):
    TITLE_FILED_IDX = 1
    ABSTRACT_FILED_IDX = 4
    corpus = get_corpus(file_loc, [TITLE_FILED_IDX, ABSTRACT_FILED_IDX])

    vectorizer = CountVectorizer(lowercase=False)
    X = vectorizer.fit_transform(corpus)
    tokens = vectorizer.get_feature_names()
    return tokens


def get_tokens_counts(file_loc, common_tokens):
    TITLE_FILED_IDX = 1
    ABSTRACT_FILED_IDX = 4
    corpus = get_corpus(file_loc, [TITLE_FILED_IDX, ABSTRACT_FILED_IDX])

    vectorizer = CountVectorizer(lowercase=False, vocabulary=common_tokens)
    X = vectorizer.fit_transform(corpus)
    line_token_counts = X.toarray()

    return line_token_counts
