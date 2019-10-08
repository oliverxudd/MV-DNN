import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.functional import cosine_similarity, softmax, nll_loss

from feature import *
from model import FeatureExtractor
from dataset import TrainDataset


class Trainer(object):
    def __init__(self, src_domain, tgt_domain):
        self.num_epoch = 10
        self.gamma = 1.0
        print('construct dataset and dataloader...')
        train_dataset = TrainDataset(src_domain, tgt_domain)
        self.NEG_NUM = train_dataset.NEG_NUM
        self.input_dim = train_dataset.sample_dim
        self.train_loader = DataLoader(train_dataset, batch_size=32)
        print('Done!')

        self.feature_extractor = FeatureExtractor(self.input_dim)
        self.optimizer = optim.SGD(self.feature_extractor.parameters(), lr=0.1, momentum=0.9)

    def train(self):
        for i in range(self.num_epoch):
            self.train_one_epoch(i)

    def train_one_epoch(self, epoch_ind):
        loss_item = 0
        for iter, (src_pos, tgt_pos, tgt_negs) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            src_pos_feature = self.feature_extractor(src_pos)
            tgt_pos_feature = self.feature_extractor(tgt_pos)
            tgt_negs_features = self.feature_extractor(tgt_negs.reshape(-1, self.input_dim))
            feature_dim = src_pos_feature.size()[1]
            tgt_negs_features = tgt_negs_features.reshape(-1, self.NEG_NUM, feature_dim)

            pos_sim = cosine_similarity(src_pos_feature, tgt_pos_feature)
            src_repeated_feature = src_pos_feature.unsqueeze(1).repeat(1, self.NEG_NUM, 1)
            neg_sims = cosine_similarity(src_repeated_feature, tgt_negs_features, dim=2)
            all_sims = torch.cat((pos_sim.unsqueeze(1), neg_sims), dim=1)

            PDQ = softmax(all_sims* self.gamma, dim=1)
            # neg_prob_sum = torch.sum(PDQ[:, 1:], 1)
            # prediction = torch.cat((PDQ[:, 0].unsqueeze(1), neg_prob_sum.unsqueeze(1)), dim=1)
            # batchsize = src_pos_feature.size()[0]
            # target = torch.zeros(batchsize).long() # 第一列是正解
            # loss = nll_loss(prediction, target)
            loss = -PDQ[:, 0].log().mean()

            loss.backward()
            self.optimizer.step()

            loss_item += loss.item()
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_ind, iter, len(self.train_loader), loss.item()))
