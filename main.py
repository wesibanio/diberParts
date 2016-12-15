"""
Created on Tue Dec 13 17:40:42 20.16
"""

from parser2 import *
from POS_lib import *
from HMM_Model import HMM_Model

if __name__ == "__main__":
    sets, x, y = collect_sets('data_split.gz')
    # print(data['train'][0][0])
    train_set_x = np.array(sets['train'])[:, 0] # POS tags
    train_set_y = np.array(sets['train'])[:, 1] # words

    hmm_model = HMM_Model(x, y)
    hmm_model.mle(train_set_x, train_set_y)
    #e, t, s = mle(train_set_x, train_set_y, x, y)
    