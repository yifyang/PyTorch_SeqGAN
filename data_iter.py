import math
import random
import torch
import torch.utils.data
import numpy as np
from transformer import Constants


class GenDataIter:
    """ Toy data iter to load digits """

    def __init__(self, data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = math.ceil(self.data_num / self.batch_size)
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.tensor(d)

        # 0 is prepended to d as start symbol
        data = torch.cat([torch.zeros(len(index), 1, dtype=torch.int64), d], dim=1)
        target = torch.cat([d, torch.zeros(len(index), 1, dtype=torch.int64)], dim=1)
        
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            # l = [int(s) for s in list(line.strip())]
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class DisDataIter:
    """ Toy data iter to load digits """

    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = self.read_file(real_data_file)
        fake_data_lis = self.read_file(fake_data_file)
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] +\
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = math.ceil(self.data_num / self.batch_size)
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.tensor(data)
        label = torch.tensor(label)
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            # l = [int(s) for s in list(line.strip())]
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis

class myDataset(torch.utils.data.Dataset):
    def __init__(self, d_o, d_r):
        self.data = torch.cat([torch.zeros(len(d_r), 1, dtype=torch.int64), d_r], dim=1)
        self.target = torch.cat([d_o, torch.zeros(len(d_o), 1, dtype=torch.int64)], dim=1)
        # self.data = data_num
        # self.target = data_num

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)

def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)

def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''

    batch_seq = np.array([list(inst) for inst in insts])

    batch_pos = np.array([
        [pos_i+1 for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos


def prepare_dataloaders(data_file, batch_size, rand_file=None):
    with open(data_file, "r") as f:
        data_ori = f.readlines()
    data_o = []
    for line in data_ori:
        num_list = line.split(" ")
        data_o.append([int(item) for item in num_list])
    data_o = np.array(data_o)
    with open(rand_file, "r") as f:
        data_rand = f.readlines()
    data_r = []
    for line in data_rand:
        num_list = line.split(" ")
        data_r.append([int(item) for item in num_list])
    data_r = np.array(data_r)

    data_o = torch.LongTensor(data_o)
    data_r = torch.LongTensor(data_r)

    train_loader = torch.utils.data.DataLoader(
        myDataset(data_o, data_r),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    # valid_loader = torch.utils.data.DataLoader(
    #     myDataset(data[4000:]),
    #     num_workers=2,
    #     batch_size=opt.batch_size,
    #     collate_fn=paired_collate_fn)

    return train_loader
