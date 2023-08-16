import os
import random
import torch
from torch.optim import AdamW
from model import Model
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import Adafactor, get_scheduler



def prompt(context):
    new_context = f'Given the context "{context}", '
    prompt = new_context + f'which options are correct?'
    return new_context, prompt


def cot(context):
    new_context = f'Given the context "{context}", '
    prompt = new_context + f' let\'s think step by step, which options are correct and why?'
    return new_context, prompt


def got_step1(context):
    new_context = f'Given the context "{context}", '
    prompt = new_context + f'Based on common sense, which options are unreasonable and why?'
    return new_context, prompt


def got_step2(context, text1,choice):
    new_context = context +text1
    prompt = new_context + f' Analysis based on the incorrect options, if the answer is {choice}, is it reasonable and why?'
    return new_context, prompt


def got_step3(context, text1, text2):
    new_context = context + text1 +text2
    prompt = new_context + f' Analysis based on the previous steps, which options are reasonable?'
    return new_context, prompt


def configure_optimizer(model, args):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    return optimizer


def configure_scheduler(optimizer, num_training_steps, args):

    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler