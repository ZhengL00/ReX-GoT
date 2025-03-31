import csv, pickle, re
import torch


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
task_id = 1
tasks =['sentiment category','sentiment degree','intention detection','offensiveness detection','metaphor occurrence','metaphor category']

def load_dataset(path, pad_size=35,lang = None,mode=None,meta = None):
    contents = []
    with open(img_pkl + 'all_' + mode + '_' + lang + '.pkl', 'rb') as f:
        id_vgg = pickle.load(f)
    with open(root_pkl+'all_text_'+lang+'.pkl', 'rb') as f:
        id_textST = pickle.load(f)
    with open(root_pkl+'all_metaphor_'+lang+'_text_meta.pkl', 'rb') as f:
        id_text_meta = pickle.load(f)
    with open(root_pkl+'all_metaphor_'+lang+'.pkl', 'rb') as f:
        id_metaphor = pickle.load(f)
    with open(root_pkl+'all_metaphor_'+lang+'_source.pkl', 'rb') as f:
        id_source = pickle.load(f)
    with open(root_pkl+'all_metaphor_'+lang+'_target.pkl', 'rb') as f:
        id_target = pickle.load(f)
    with open(root_pkl+'all_metaphor_'+lang+'_pad.pkl', 'rb') as f:
        id_metaphor_pad = pickle.load(f)

    with open(path, encoding='utf-8') as f:
        m_cate_name = dict()
        m_cate_name['image dominant'] = 1
        m_cate_name['text dominant'] = 2
        m_cate_name['complementary'] = 3
        num=0
        for line in f:
            if num==0:
                num=1
                continue
            line = line.split(',')
            id = line[0]
            label_sentiment = line[1].split('(')[0]
            label_intention = line[3].split('(')[0]
            label_offensiveness = line[4].split('(')[0]
            label_m_occurrence = line[5].split('(')[0]
            if lang == 'all':
                id = line[1]
                label_sentiment = line[6].split('(')[0]
                label_intention = line[2].split('(')[0]
                label_offensiveness = line[5].split('(')[0]
                label_m_occurrence = line[4].split('(')[0]
            if label_m_occurrence == '0':
                label_m_category = '4'
            else:
                if lang == 'all':
                    label_m_category = m_cate_name[line[3]]
                else:
                    label_m_category = m_cate_name[line[6]]
            imgfeature = id_vgg[id]
            textfeature = torch.FloatTensor(id_textST[id])
            metafeature = torch.FloatTensor(id_metaphor[id])
            sourcefeature = torch.FloatTensor(id_source[id])
            targetfeature = torch.FloatTensor(id_target[id])
            text_metafeature = torch.FloatTensor(id_text_meta[id])
            padfeature = torch.FloatTensor(id_metaphor_pad[id])

            contents.append([textfeature, metafeature,padfeature,imgfeature, int(label_sentiment) - 1, int(label_intention) - 1, int(label_offensiveness), int(label_m_occurrence), int(label_m_category) - 1,id,sourcefeature,targetfeature,text_metafeature])

    return contents

def build_dataset(pad_size = 15,lang=None,mode=None,meta=None):

    train_path = root_csv+'avg_train_label_'+lang+'.csv'
    val_path = root_csv+'avg_val_label_'+lang+'.csv'
    test_path = root_csv+'avg_test_label_'+lang+'.csv'

    train = load_dataset(train_path, pad_size=pad_size,lang = lang,mode=mode,meta=meta)
    val = load_dataset(val_path, pad_size=pad_size,lang = lang,mode=mode,meta=meta)
    test = load_dataset(test_path, pad_size=pad_size,lang = lang,mode=mode,meta=meta)

    return train, val, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device,lang):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.lang = lang

    def _to_tensor(self, datas):
        text = torch.FloatTensor([_[0].tolist() for _ in datas]).to(self.device)
        meta = torch.FloatTensor([_[1].tolist() for _ in datas]).to(self.device)
        image = torch.FloatTensor([_[3] for _ in datas]).to(self.device)
        meta_pad = torch.FloatTensor([_[2].tolist() for _ in datas]).to(self.device)

        y_sentiment = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        y_intention = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        y_offensiveness = torch.LongTensor([_[6] for _ in datas]).to(self.device)
        y_m_occurrence = torch.LongTensor([_[7] for _ in datas]).to(self.device)
        y_m_category = torch.LongTensor([_[8] for _ in datas]).to(self.device)

        source = torch.FloatTensor([_[10].tolist() for _ in datas]).to(self.device)
        target = torch.FloatTensor([_[11].tolist() for _ in datas]).to(self.device)
        text_meta = torch.FloatTensor([_[12].tolist() for _ in datas]).to(self.device)

        return (text, meta, meta_pad,image, id, source, target, text_meta), (y_sentiment,y_intention,y_offensiveness,y_m_occurrence,y_m_category)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, batch_size, device,lang):
    iter = DatasetIterater(dataset, batch_size, device,lang)
    return iter
