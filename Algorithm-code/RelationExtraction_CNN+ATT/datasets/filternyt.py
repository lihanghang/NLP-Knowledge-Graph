# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
import numpy as np


class FilterNYTData(Dataset):

    def __init__(self, root_path, train=True):
        if train:
            path = os.path.join(root_path, 'train/')
            print('loading train data')
        else:
            path = os.path.join(root_path, 'test/')
            print('loading test data')

        self.labels = np.load(path + 'labels.npy')
        self.x = np.load(path + 'bags_feature.npy')
        self.x = zip(self.x, self.labels)

        print('loading finish')

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)


class FilterNYTLoad(object):
    '''
    load and preprocess data
    '''
    def __init__(self, root_path, max_len=80, limit=50, pos_dim=5, pad=1):

        self.max_len = max_len
        self.limit = limit
        self.root_path = root_path
        self.pos_dim = pos_dim
        self.pad = pad

        self.w2v_path = os.path.join(root_path, 'vector.txt')
        self.word_path = os.path.join(root_path, 'dict.txt')
        self.train_path = os.path.join(root_path, 'train', 'train.txt')
        self.test_path = os.path.join(root_path, 'test', 'test.txt')

        print('loading start....')
        self.w2v, self.word2id, self.id2word = self.load_w2v()
        self.p1_2v, self.p2_2v = self.load_p2v()

        np.save(os.path.join(self.root_path, 'w2v.npy'), self.w2v)
        np.save(os.path.join(self.root_path, 'p1_2v.npy'), self.p1_2v)
        np.save(os.path.join(self.root_path, 'p2_2v.npy'), self.p2_2v)

        print("parsing train text...")
        self.bags_feature, self.labels = self.parse_sen(self.train_path)
        np.save(os.path.join(self.root_path, 'train', 'bags_feature.npy'), self.bags_feature)
        np.save(os.path.join(self.root_path, 'train', 'labels.npy'), self.labels)

        print("parsing test text...")
        self.bags_feature, self.labels = self.parse_sen(self.test_path)
        np.save(os.path.join(self.root_path, 'test', 'bags_feature.npy'), self.bags_feature)
        np.save(os.path.join(self.root_path, 'test', 'labels.npy'), self.labels)
        print('save finish!')

    def load_p2v(self):
        pos1_vec = np.asarray(np.random.uniform(low=-1.0, high=1.0, size=(self.limit * 2 + 1, self.pos_dim)), dtype=np.float32)
        pos1_vec = np.vstack((np.zeros((1, self.pos_dim)), pos1_vec))
        pos2_vec = np.asarray(np.random.uniform(low=-1.0, high=1.0, size=(self.limit * 2 + 1, self.pos_dim)), dtype=np.float32)
        pos2_vec = np.vstack((np.zeros((1, self.pos_dim)), pos2_vec))

        return pos1_vec, pos2_vec

    def load_w2v(self):
        '''
        reading from vec.bin
        add two extra tokens:
            : UNK for unkown tokens
            : BLANK for the max len sentence
        '''
        wordlist = []
        vecs = []

        wordlist.append('BLANK')
        wordlist.extend([word.strip('\n') for word in open(self.word_path)])

        for line in open(self.w2v_path):
            line = line.strip('\n').split()
            vec = map(float, line)
            vecs.append(vec)

        dim = len(vecs[0])
        vecs.insert(0, np.zeros(dim))
        wordlist.append('UNK')

        vecs.append(np.random.uniform(low=-1.0, high=1.0, size=dim))
        # rng = np.random.RandomState(3435)
        # vecs.append(rng.uniform(low=-0.5, high=0.5, size=dim))
        word2id = {j: i for i, j in enumerate(wordlist)}
        id2word = {i: j for i, j in enumerate(wordlist)}

        return np.array(vecs, dtype=np.float32), word2id, id2word

    def parse_sen(self, path):
        '''
        parse the records in data
        '''
        all_sens =[]
        all_labels =[]
        f = open(path)
        while 1:
            line = f.readline()
            if not line:
                break
            entities = map(int, line.split(' '))
            line = f.readline()
            bagLabel = line.split(' ')

            rel = map(int, bagLabel[0:-1])
            num = int(bagLabel[-1])
            positions = []
            sentences = []
            entitiesPos = []
            masks = []
            for i in range(0, num):
                sent = f.readline().split(' ')
                positions.append(map(int, sent[0:2]))
                epos = map(lambda x: int(x) + 1, sent[0:2])
                epos.sort()
                mask = [1] * (epos[0] + 1)
                mask += [2] * (epos[1] - epos[0])
                mask += [3] * (len(sent[2:-1]) - epos[1])
                entitiesPos.append(epos)
                sentences.append(map(int, sent[2:-1]))
                masks.append(mask)
            bag = [entities, num, sentences, positions, entitiesPos, masks]
            all_labels.append(rel)
            all_sens += [bag]

        f.close()
        bags_feature = self.get_sentence_feature(all_sens)

        return bags_feature, all_labels

    def get_sentence_feature(self, bags):
        '''
        : word embedding
        : postion embedding
        return:
        sen list
        pos_left
        pos_right
        '''
        update_bags = []

        for bag in bags:
            es, num, sens, pos, enPos, masks = bag
            new_sen = []
            new_pos = []
            new_masks = []
            for idx, sen in enumerate(sens):
                _pos, _mask = self.get_pos_feature(len(sen), pos[idx], masks[idx])
                new_pos.append(_pos)
                new_masks.append(_mask)
                new_sen.append(self.get_pad_sen(sen))
            update_bags.append([es, num, new_sen, new_pos, enPos, new_masks])

        return update_bags

    def get_pad_sen(self, sen):
        '''
        padding the sentences
        '''
        # sen.insert(0, self.word2id['BLANK'])
        if len(sen) < self.max_len + 2 * self.pad:
            sen += [self.word2id['BLANK']] * (self.max_len +2 * self.pad - len(sen))
        else:
            sen = sen[: self.max_len + 2 * self.pad]

        return sen

    def get_pos_feature(self, sen_len, ent_pos, mask):
        '''
        clip the postion range:
        : -limit ~ limit => 0 ~ limit * 2+2
        : -51 => 1
        : -50 => 1
        : 50 => 101
        : >50: 102
        '''

        def padding(x):
            if x < 1:
                return 1
            if x > self.limit * 2 + 1:
                return self.limit * 2 + 1
            return x

        if sen_len < self.max_len:
            index = np.arange(sen_len)
        else:
            index = np.arange(self.max_len)

        pf1 = []
        pf2 = []
        pf1 += map(padding, index - ent_pos[0] + 2 + self.limit)
        pf2 += map(padding, index - ent_pos[1] + 2 + self.limit)

        if len(pf1) < self.max_len + 2 * self.pad:
            pf1 += [0] * (self.max_len + 2 * self.pad - len(pf1))
            pf2 += [0] * (self.max_len + 2 * self.pad - len(pf2))
            mask += [0] * (self.max_len + 2 * self.pad - len(mask))
        return [pf1, pf2], mask


if __name__ == "__main__":
    data = FilterNYTLoad('./datasets/FilterNYT/')
