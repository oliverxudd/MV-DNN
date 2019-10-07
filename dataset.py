import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from feature import domain_names, four_pair_of_domains, DATA_DIR, get_features

class TrainDataset(Dataset):
    neg_num_for_one_sample = 5

    def __init__(self, src_domain, tgt_domain):
        feature_dict = get_features(src_domain, tgt_domain)
        self.src_authors = feature_dict['src_domain_authors']
        print('We got {} authors from source domain'.format(len(self.src_authors)))
        self.tgt_authors = feature_dict['tgt_domain_authors']
        print('We got {} authors from target domain'.format(len(self.tgt_authors)))
        self.crossfield_authors = get_crossfield_authors(src_authors, tgt_authors)
        print('We got {} cross-field authors'.format(len(self.crossfield_authors)))
        self.src_only_authors = get_first_only_authors(src_authors, tgt_authors)
        self.tgt_only_authors = get_first_only_authors(tgt_authors, src_authors)
        self.src_paper_author_mat = feature_dict['src_paper_author_matrix']
        self.tgt_paper_author_mat = feature_dict['tgt_paper_author_matrix']

    def __getitem__(self, idx):
        pos_sample = get_one_pos_sample() # size: (channel=2, dim)
        N_neg_samples = getN_neg_samples() # size: (channel=2, neg_num_for_one_sample, dim)

        return torch.from_numpy(pos_sample), torch.from_numpy(N_neg_samples)

    def __len__(self):
        return len(self.crossfield_authors) # 注意，由于__getitem__返回的样本是临时随机构建的，所以数据集的长度是人工指定的

    def get_crossfield_authors(self, domain1_authors, domain2_authors):
        res_authors = domain1_authors

        for au in domain1_authors:
            if au not in domain2_authors:
                res_authors.remove(au)

        return res_authors

    def get_first_only_authors(self, domain1_authors, domain2_authors):
        res_authors = domain1_authors

        for au in domain1_authors:
            if au in domain2_authors:
                res_authors.remove(au)

        return res_authors

    def get_one_pos_sample(self):
        one_pos_sample = try_get_one_pos_example(self)
        null_result = one_pos_sample.shape[0] < 2
        while null_result:
            one_pos_sample = try_get_one_pos_example(self)
            null_result = one_pos_sample.shape[0] < 2

        return one_pos_sample

    def try_get_one_pos_example(self):
        author1 = random.choice(self.crossfield_authors)
        paper = find_paper_of_author(author1)
        author2 = find_another_author_of_paper(paper, author1)
        author1_featureVec = feature_of_author(author1)
        author2_featureVec = feature_of_author(author2)

        return np.vstack((author1_featureVec, author2_featureVec))

    def find_paper_of_author(self):

