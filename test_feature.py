import os
import os.path as osp
import random
import numpy as np
from scipy.linalg import blas
from feature import *


def test_scipy_sgemm():
    pair_domain = four_pair_of_domains[0]
    src_domain, tgt_domain = pair_domain[0], pair_domain[1]
    srcfile_loc = osp.join(DATA_DIR, src_domain+'.txt')
    tgtfile_loc = osp.join(DATA_DIR, tgt_domain+'.txt')

    src_names, src_name_line_count  = get_names_counts(srcfile_loc)
    tgt_names, tgt_name_line_count  = get_names_counts(tgtfile_loc)
    common_tokens = get_common_tokens(srcfile_loc, tgtfile_loc)

    print('src', src_name_line_count.shape)
    print('tgt', tgt_name_line_count.shape)
    print('common' , len(common_tokens))

    src_line_token_count = get_tokens_counts(srcfile_loc, common_tokens)
    tgt_line_token_count = get_tokens_counts(tgtfile_loc, common_tokens)

    print('src token', src_line_token_count.shape)
    print('tgt token', tgt_line_token_count.shape)

    res = blas.sgemm(alpha=1.0, a=src_name_line_count, b=src_line_token_count.T, trans_b=True)

    print('we need <dot result> == <sgemm result>')
    for i in range(10):
        X, Y = np.nonzero(res)
        rand_ind = random.choice(range(len(X)))
        xx = X[rand_ind]
        yy = Y[rand_ind]
        print('dot res:', np.dot(src_name_line_count[xx, :], src_line_token_count[:, yy]))
        print('sgemm res:', res[xx, yy])
        print()


def test_get_train_features():
    pair_domain = four_pair_of_domains[0]
    src_domain, tgt_domain = pair_domain[0], pair_domain[1]
    feature_dict = get_train_features(src_domain, tgt_domain)

    src_authors = feature_dict['src_authors']
    src_features = feature_dict['src_features']
    vocab_all = feature_dict['vocab_all']

    rand_idx = random.choice(range(len(src_authors)))
    print('rand idx:', rand_idx)
    print('researcher name:', src_authors[rand_idx])

    X = np.nonzero(src_features[rand_idx, :])
    ocurred_words = np.asarray(vocab_all)[X]
    print('his feature words:')
    print(ocurred_words)
    print('my friend, go look for researcher ', src_authors[rand_idx], ' in ', src_domain, '.txt. Check his feature words.')


if __name__ == '__main__':
    test_get_train_features()

