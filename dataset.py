import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from feature import *
import matplotlib.pyplot as plt

class TrainDataset(Dataset):

    def __init__(self, src_domain, tgt_domain):
        self.NEG_NUM = 5

        feature_dict = get_train_features(src_domain, tgt_domain)
        self.src_features = feature_dict['src_features']
        self.tgt_features = feature_dict['tgt_features']

        self.src_authors = feature_dict['src_authors']
        self.tgt_authors = feature_dict['tgt_authors']
        self.src_author_paper = feature_dict['src_paper_author'].T
        self.tgt_author_paper = feature_dict['tgt_paper_author'].T
        crossfield_authors = self.get_crossfield_authors(self.src_authors, self.tgt_authors)
        self.author_collaborators = {}
        for au in crossfield_authors:
            collaborators_ind = self.get_collaborators_from_target(au)
            self.author_collaborators[au] = collaborators_ind

        self.sample_num = self.get_sample_num()
        self.sample_dim = self.src_features.shape[1]

        self.create_samples()

    def __len__(self):
        return self.sample_num

    def __getitem__(self, item):
        src_pos = torch.from_numpy(self.source_pos_samples[item])
        tgt_pos = torch.from_numpy(self.target_pos_samples[item])
        tgt_negs = torch.from_numpy(self.target_negs_samples[item])
        return src_pos, tgt_pos, tgt_negs

    def get_crossfield_authors(self, domain1_authors, domain2_authors):
        # 强调：避开list的浅拷贝
        res_authors = []
        for au in domain1_authors:
            if au in domain2_authors:
                res_authors.append(au)

        return res_authors

    def get_collaborators_from_target(self, author):
        author_ind = self.tgt_authors.index(author)
        papers = np.nonzero(self.tgt_author_paper[author_ind, :])[0]
        related_authors = np.sum(self.tgt_author_paper[:, papers], axis=1)
        collaborators_ind = np.nonzero(related_authors)[0].tolist()
        collaborators_ind.remove(author_ind)
        return np.array(collaborators_ind)

    def get_sample_num(self):
        collaborator_matrix = np.hstack(self.author_collaborators.values())
        sample_num = np.nonzero(collaborator_matrix)[0].shape[0]
        return sample_num

    def create_samples(self):
        self.source_pos_samples = []
        self.target_pos_samples = []
        self.target_negs_samples = []

        for au in self.author_collaborators.keys():
            for collab_au_ind in self.author_collaborators[au]:
                au_ind = self.src_authors.index(au)
                self.source_pos_samples.append(self.src_features[au_ind, :])
                self.target_pos_samples.append(self.tgt_features[collab_au_ind, :])

                all_inds = np.array(range(len(self.tgt_authors)))
                candidates_inds = np.setdiff1d(all_inds, self.author_collaborators[au])
                random.shuffle(candidates_inds)
                neg_inds = candidates_inds[0:self.NEG_NUM]
                self.target_negs_samples.append(self.tgt_features[neg_inds, :])


def test_collabrators(dataset):
    authors = list(dataset.author_collaborators.keys())
    for i in range(10):
        author = random.choice(authors)
        print('author: {}'.format(author))
        collab_ind = dataset.author_collaborators[author]
        print('his collaborators:', np.array(dataset.tgt_authors)[collab_ind])
        print()

def test_sample_using_cosine_similarity(dataset):
    source_pos_samples = np.vstack(dataset.source_pos_samples)
    target_pos_samples = np.vstack(dataset.target_pos_samples)
    target_negs_samples = np.vstack(dataset.target_negs_samples)
    print('sample size: source, target_pos, target_negs')
    print(source_pos_samples.shape)
    print(target_pos_samples .shape)
    print(target_negs_samples .shape)
    corr_pos = []
    corr_neg = []
    for i in range(source_pos_samples.shape[0]):
        corr_pos.append(cos_sim(source_pos_samples[i, :], target_pos_samples[i, :]))
        for j in range(dataset.NEG_NUM):
            corr_neg.append( cos_sim(source_pos_samples[i, :], target_negs_samples[i*dataset.NEG_NUM+j, :]) )

    plt.figure()
    plt.title('corr_pos')
    plt.hist(np.array(corr_pos))

    plt.figure()
    plt.title('corr_neg')
    plt.hist(np.array(corr_neg))
    plt.show()

def cos_sim(a, b):
    dot_res = np.dot(a, b)
    dot_res = dot_res/ (np.linalg.norm(a)* np.linalg.norm(b))
    return dot_res

if __name__ == '__main__':
    pair_domains = four_pair_of_domains[0]
    src_domain = pair_domains[0]
    tgt_domain = pair_domains[1]
    dataset = TrainDataset(src_domain, tgt_domain)

    # test_collabrators(dataset)
    test_sample_using_cosine_similarity(dataset)
