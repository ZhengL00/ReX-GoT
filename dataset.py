import torch
import json
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CICERO2Dataset(Dataset):
    def __init__(self, f, shuffle):
        content, labels, answer = [], [], []
        x = open(f).readlines()
        if shuffle:
            random.shuffle(x)

        for line in x:

            instance = json.loads(line)
            choices = instance["Choices"]
            context = sep_token.join([instance["Question"], "target: " + instance["Target"],
                                      "context: " + " <utt> ".join(instance["Dialogue"])])
            l = instance["Correct Answers"]
            for k, c in enumerate(choices):
                content.append("{} \\n choice: {}".format(context, c))
                content.append("{} \\n let's think step by step,and answer the two questions:, is the answer {} right and why?".format(context, c))
                if k == l:
                    labels.append(1)
                else:
                    labels.append(0)

        self.content, self.labels, self.choices = content, labels, choices

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        s1, s2, s3 = self.content[index], self.labels[index], self.choices[index]

        return s1, s2, s3

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


def configure_dataloaders(train_batch_size=16, eval_batch_size=16, shuffle=False):
    "Prepare dataloaders"
    train_dataset = CICERO2Dataset("data/cicero/train.json", True)
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=train_batch_size,
                              collate_fn=train_dataset.collate_fn)

    val_dataset = CICERO2Dataset("data/cicero/val.json", False)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)

    test_dataset = CICERO2Dataset("data/cicero/test.json", False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)

    return train_loader, val_loader, test_loader

