import json
import time
import random
import gc, os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Model
from sklearn.metrics import accuracy_score, f1_score
from transformers import T5Tokenizer, T5ForConditionalGeneration



def train_or_eval_model(model, dataloader, optimizer=None, split="Train"):
    losses, preds, preds_cls, labels_cls= [], [], [], []
    if split == "Train":
        model.train()
    else:
        model.eval()

    for batch in tqdm(dataloader, leave=False):
        if split == "Train":
            optimizer.zero_grad()

        content, l_cls = batch
        loss, p, p_cls= model(batch)

        preds.append(p)
        preds_cls.append(p_cls)
        labels_cls.append(l_cls)

        if split == "Train":
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)

    all_preds_cls = [item for sublist in preds_cls for item in sublist]
    all_labels_cls = [item for sublist in labels_cls for item in sublist]
    acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
    f1 = round(f1_score(all_labels_cls, all_preds_cls, average="macro"), 4)


    instance_preds = [item for sublist in preds for item in sublist]
    instance_labels = np.array(all_labels_cls).reshape(-1, args.num_choices).argmax(1)
    instance_acc = round(accuracy_score(instance_labels, instance_preds), 4)


    return avg_loss, acc, instance_acc, f1

def configure_optimizer(model, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=args.adam_epsilon)
    return optimizer


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate for transformers.")
    parser.add_argument("--wd", default=0.0, type=float, help="Weight decay for transformers.")
    parser.add_argument("--warm-up-steps", type=int, default=0, help="Warm up steps.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--eval-bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs.")
    parser.add_argument("--name", default="", help="Which model.")
    parser.add_argument('--shuffle', action='store_true', default=False, help="Shuffle train data such that positive and negative \
        sequences of the same question are not necessarily in the same batch.")

    global args
    args = parser.parse_args()
    print(args)


    train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    epochs = args.epochs
    name = args.name
    shuffle = args.shuffle

    num_choices = 4
    vars(args)["num_choices"] = num_choices
    assert eval_batch_size % num_choices == 0, "Eval batch size should be a multiple of num choices, which is 4 for CICERO2"

    model = Model(
        name=name,
        num_choices=num_choices
    ).cuda()

    sep_token = model.tokenizer.sep_token

    optimizer = configure_optimizer(model, args)

    for e in range(epochs):
        train_loader, val_loader, test_loader = configure_dataloaders(
            train_batch_size, eval_batch_size, shuffle
        )

        train_loss, train_acc, _, train_f1 = train_or_eval_model(model, train_loader, optimizer, "Train")
        val_loss, val_acc, val_ins_acc, val_f1 = train_or_eval_model(model, val_loader, split="Val")
        test_loss, test_acc, test_ins_acc, test_f1 = train_or_eval_model(model, test_loader, split="Test")

        x = "Epoch {}: Loss: Train {}; Val {}; Test {}".format(e + 1, train_loss, val_loss, test_loss)
        y1 = "Classification Acc: Train {}; Val {}; Test {}".format(train_acc, val_acc, test_acc)
        y2 = "Classification Macro F1: Train {}; Val {}; Test {}".format(train_f1, val_f1, test_f1)
        z = "Instance Acc: Val {}; Test {}".format(val_ins_acc, test_ins_acc)

        lf = open(lf_name, "a")
        lf.write(x + "\n" + y1 + "\n" + y2 + "\n" + z + "\n\n")
        lf.close()

        f = open(fname, "a")
        f.write(x + "\n" + y1 + "\n" + y2 + "\n" + z + "\n\n")
        f.close()

    lf = open(lf_name, "a")
    lf.close()
