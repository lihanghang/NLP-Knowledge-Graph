# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
import numpy as np


class NYTData(Dataset):

    def __init__(self, root_path, train=True):
        if train:
            path = os.path.join(root_path, 'train/')
            print('loading train data')
        else:
            path = os.path.join(root_path, 'test/')
            print('loading test data')

        self.labels = np.load(path + 'labels.npy')
        self.x = np.load(path + 'bags_feature.npy')
        self.x = list(zip(self.x, self.labels))

        print('loading finish')

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)


class NYTLoad(object):
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
        self.train_path = os.path.join(root_path, 'bags_train.txt')
        self.test_path = os.path.join(root_path, 'bags_test.txt')

        print('loading start....')
        self.w2v, self.word2id, self.id2word = self.load_w2v()
        self.p1_2v, self.p2_2v = self.load_p2v()

        np.save(os.path.join(self.root_path, 'w2v.npy'), self.w2v)
        np.save(os.path.join(self.root_path, 'p1_2v.npy'), self.p1_2v)
        np.save(os.path.join(self.root_path, 'p2_2v.npy'), self.p2_2v)

        print("parsing train text...")
        self.bags_feature, self.labels = self.parse_sen(self.train_path, 'train')
        np.save(os.path.join(self.root_path, 'train', 'bags_feature.npy'), self.bags_feature)
        np.save(os.path.join(self.root_path, 'train', 'labels.npy'), self.labels)

        print("parsing test text...")
        self.bags_feature, self.labels = self.parse_sen(self.test_path, 'test')
        np.save(os.path.join(self.root_path, 'test', 'bags_feature.npy'), self.bags_feature)
        np.save(os.path.join(self.root_path, 'test', 'labels.npy'), self.labels)
        print('save finish!')

    def load_p2v(self):
        pos1_vec = [np.zeros(self.pos_dim)]
        pos1_vec.extend([np.random.uniform(low=-1.0, high=1.0, size=self.pos_dim) for _ in range(self.limit * 2 + 1)])
        pos2_vec = [np.zeros(self.pos_dim)]
        pos2_vec.extend([np.random.uniform(low=-1.0, high=1.0, size=self.pos_dim) for _ in range(self.limit * 2 + 1)])
        return np.array(pos1_vec, dtype=np.float32), np.array(pos2_vec, dtype=np.float32)

    def load_w2v(self):
        '''
        reading from vec.bin
        add two extra tokens:
            : UNK for unkown tokens
        '''
        wordlist = []

        f = open(self.w2v_path)
        # dim = int(f.readline().split()[1])
        # f = f.readlines()

        vecs = []
        for line in f:
            line = line.strip('\n').split()
            vec = list(map(float, line[1].split(',')[:-1]))
            vecs.append(vec)
            wordlist.append(line[0])

        #  wordlist.append('UNK')
        #  vecs.append(np.random.uniform(low=-0.5, high=0.5, size=dim))
        word2id = {j: i for i, j in enumerate(wordlist)}
        id2word = {i: j for i, j in enumerate(wordlist)}

        return np.array(vecs, dtype=np.float32), word2id, id2word

    def parse_sen(self, path, flag):
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
            if flag == 'train':
                line = line.split('\t')
                num = line[3].strip().split(',')
                num = len(num)
            else:
                line = line.split('\t')
                num = line[2].strip().split(',')
                num = len(num)

            ldists = []
            rdists = []
            sentences = []
            entitiesPos = []
            pos = []
            masks = []
            rels = []

            for i in range(num):
                ent_pair_line = f.readline().strip().split(',')
                #  entities = ent_pair_line[:2]
                # ignore the entities index in vocab
                entities = [0, 0]
                epos = list(map(lambda x: int(x) + 1, ent_pair_line[2:4]))
                pos.append(epos)
                epos.sort()
                entitiesPos.append(epos)

                rel = int(ent_pair_line[4])
                rels.append(rel)
                sent = f.readline().strip().split(',')
                sentences.append(list(map(lambda x: int(x), sent)))
                ldist = f.readline().strip().split(',')
                rdist = f.readline().strip().split(',')
                mask = f.readline().strip().split(",")
                ldists.append(list(map(int, ldist)))
                rdists.append(list(map(int, rdist)))
                masks.append(list(map(int, mask)))

            rels = list(set(rels))
            if len(rels) < 4:
                rels.extend([-1] * (4 - len(rels)))
            else:
                rels = rels[:4]
            bag = [entities, num, sentences, ldists, rdists, pos, entitiesPos, masks]

            all_labels.append(rels)
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
            es, num, sens, ldists, rdists, pos, enPos, masks = bag
            new_sen = []
            new_pos = []
            new_entPos = []
            new_masks= []

            for idx, sen in enumerate(sens):
                sen, pf1, pf2, pos, mask = self.get_pad_sen_pos(sen, ldists[idx], rdists[idx], enPos[idx], masks[idx])
                new_sen.append(sen)
                new_pos.append([pf1, pf2])
                new_entPos.append(pos)
                new_masks.append(mask)
            update_bags.append([es, num, new_sen, new_pos, new_entPos, new_masks])

        return update_bags

    def get_pad_sen_pos(self, sen, ldist, rdist, pos, mask):
        '''
        refer: github.com/SharmisthaJat/RE-DS-Word-Attention-Models
        '''
        x = []
        pf1 = []
        pf2 = []
        masks = []

        # shorter than max_len
        if len(sen) <= self.max_len:
            for i, ind in enumerate(sen):
                x.append(ind)
                pf1.append(ldist[i] + 1)
                pf2.append(rdist[i] + 1)
                masks.append(mask[i])
        # longer than max_len, expand between two entities
        else:
            idx = [i for i in range(pos[0], pos[1] + 1)]
            if len(idx) > self.max_len:
                idx = idx[:self.max_len]
                for i in idx:
                    x.append(sen[i])
                    pf1.append(ldist[i] + 1)
                    pf2.append(rdist[i] + 1)
                    masks.append(mask[i])
                pos[0] = 1
                pos[1] = len(idx) - 1
            else:
                for i in idx:
                    x.append(sen[i])
                    pf1.append(ldist[i] + 1)
                    pf2.append(rdist[i] + 1)
                    masks.append(mask[i])

                before = pos[0] - 1
                after = pos[1] + 1
                pos[0] = 1
                pos[1] = len(idx) - 1
                numAdded = 0
                while True:
                    added = 0
                    if before >= 0 and len(x) + 1 <= self.max_len + self.pad:
                        x.append(sen[before])
                        pf1.append(ldist[before] + 1)
                        pf2.append(rdist[before] + 1)
                        masks.append(mask[before])
                        added = 1
                        numAdded += 1

                    if after < len(sen) and len(x) + 1 <= self.max_len + self.pad:
                        x.append(sen[after])
                        pf1.append(ldist[after] + 1)
                        pf2.append(rdist[after] + 1)
                        masks.append(mask[after])
                        added = 1

                    if added == 0:
                        break

                    before -= 1
                    after += 1

                pos[0] = pos[0] + numAdded
                pos[1] = pos[1] + numAdded

        while len(x) < self.max_len + 2 * self.pad:
            x.append(0)
            pf1.append(0)
            pf2.append(0)
            masks.append(0)

        if pos[0] == pos[1]:
            if pos[1] + 1 < len(sen):
                pos[1] += 1
            else:
                if pos[0] - 1 >= 1:
                    pos[0] = pos[0] - 1
                else:
                    raise Exception('pos= {},{}'.format(pos[0], pos[1]))

        return [x, pf1, pf2, pos, masks]

    def get_pad_sen(self, sen):
        '''
        padding the sentences
        '''
        sen.insert(0, self.word2id['BLANK'])
        if len(sen) < self.max_len + 2 * self.pad:
            sen += [self.word2id['BLANK']] * (self.max_len +2 * self.pad - len(sen))
        else:
            sen = sen[: self.max_len + 2 * self.pad]

        return sen

    def get_pos_feature(self, sen_len, ent_pos):
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

        pf1 = [0]
        pf2 = [0]
        pf1 += list(map(padding, index - ent_pos[0] + 2 + self.limit))
        pf2 += list(map(padding, index - ent_pos[1] + 2 + self.limit))

        if len(pf1) < self.max_len + 2 * self.pad:
            pf1 += [0] * (self.max_len + 2 * self.pad - len(pf1))
            pf2 += [0] * (self.max_len + 2 * self.pad - len(pf2))
        return [pf1, pf2]


if __name__ == "__main__":
    data = NYTLoad('./datasets/NYT/')