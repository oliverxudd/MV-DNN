from trainer import Trainer
from feature import *

if __name__ == '__main__':
    pair_domains = four_pair_of_domains[0]
    src_domain = pair_domains[0]
    tgt_domain = pair_domains[1]

    trainer = Trainer(src_domain, tgt_domain)
    trainer.train()
