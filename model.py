import json
import random
import numpy as np
import torch
import transformers
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import prompt,cot,got_step1,got_step2,got_step3

class Model(nn.Module):
    def __init__(
            self,
            name: str,
            num_choices: int
    ):
        super().__init__()

        self.name = name
        self.num_choices = num_choices
        self.tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.model_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        self.max_length = 512
        self.hidden_size = 1024
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.classify = nn.Linear(self.hidden_size, 2)

    def score_input(self, content,labels,choices):
        outputs = []
        for text in content:
            context_1, got1 = got_step1(text)
            input_ids1 = self.tokenizer_t5(got1, return_tensors="pt").input_ids
            output1 = self.model_t5.generate(input_ids1.to(self.model.device))
            input_reconstructed = self.model_t5.generate(input_ids=output1)
            input_reconstructed_decoded = self.tokenizer_t5.decode(input_reconstructed[0], skip_special_tokens=True)
            input_ids_reconstructed = self.tokenizer_t5.encode(input_reconstructed_decoded, return_tensors="pt")
            if input_ids_reconstructed.shape[1] < input_ids1.shape[1]:
                input_ids_reconstructed = torch.nn.functional.pad(
                    input_ids_reconstructed,
                    (0, input_ids1.shape[1] - input_ids_reconstructed.shape[1]),
                    value=self.tokenizer_t5.pad_token_id
                )
            out1 = self.tokenizer_t5.decode(output1[0])
            for i in range(choices):
                context_2, got2 = got_step2(text, out1,choices[i])
                input_ids2 = self.tokenizer_t5(got2, return_tensors="pt").input_ids
                output2 = self.model_t5.generate(input_ids2.to(self.model.device))
                input_reconstructed2 = self.model_t5.generate(input_ids=output2)
                input_reconstructed_decoded2 = self.tokenizer_t5.decode(input_reconstructed2[0], skip_special_tokens=True)
                input_ids_reconstructed2 = self.tokenizer_t5.encode(input_reconstructed_decoded2, return_tensors="pt")
                if input_ids_reconstructed2.shape[1] < input_ids2.shape[1]:
                    input_ids_reconstructed2 = torch.nn.functional.pad(
                        input_ids_reconstructed2,
                        (0, input_ids2.shape[1] - input_ids_reconstructed2.shape[1]),
                        value=self.tokenizer_t5.pad_token_id
                    )
                out2 = self.tokenizer_t5.decode(output2[0])

            context_3, got3 = got_step3(text, out1, out2)
            num_answers = 3
            answers = []
            for _ in range(num_answers):
                input_ids3 = self.tokenizer_t5(got3, return_tensors="pt").input_ids
                output3 = self.model_t5.generate(input_ids3.to(self.model.device))
                input_reconstructed3 = self.model_t5.generate(input_ids=output3)
                input_reconstructed_decoded3 = self.tokenizer_t5.decode(input_reconstructed3[0],skip_special_tokens=True)
                input_ids_reconstructed3 = self.tokenizer_t5.encode(input_reconstructed_decoded3, return_tensors="pt")
                if input_ids_reconstructed3.shape[1] < input_ids3.shape[1]:
                    input_ids_reconstructed3 = torch.nn.functional.pad(
                        input_ids_reconstructed3,
                        (0, input_ids3.shape[1] - input_ids_reconstructed3.shape[1]),
                        value=self.tokenizer_t5.pad_token_id
                    )
                answer_start = torch.argmax(output3.start_logits)
                answer_end = torch.argmax(output3.end_logits)
                answer = self.tokenizer_t5.convert_tokens_to_string(
                    self.tokenizer_t5.convert_ids_to_tokens(input_ids3["input_ids"][0][answer_start:answer_end + 1])
                )
                answers.append(answer)
            voting_machine = VotingMachine()
            for answer in answers:
                voting_machine.vote(answer)
            out3 = voting_machine.get_results()
            outputs.append(out3)
        batch = self.tokenizer_t5(
            outputs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        out = self.model_t5(
            batch["input_ids"].to(self.model.device), batch["attention_mask"].to(self.model.device),
            output_hidden_states=True
        )
        loss = self.ce_loss_func(out["logits"], labels)

        return out["logits"], loss

    def forward(self, batch):
        content, labels, answer = batch
        labels = torch.tensor(labels, dtype=torch.long).to(self.model.device)
        logits, loss = self.score_input(content,labels)
        preds_cls = list(torch.argmax(logits, 1).cpu().numpy())
        positive_logits = logits[:, 1]
        preds = torch.argmax(positive_logits.reshape(-1, self.num_choices), 1)
        preds = list(preds.cpu().numpy())
        return loss, preds, preds_cls

