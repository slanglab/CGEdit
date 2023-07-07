import transformers
import torch
import torch.nn as nn
import numpy as np
import dataclasses
from torch.utils.data import Dataset
from typing import List, Union, Dict
import sys


gen_method = sys.argv[1]    # 'CGEdit' or 'CGEdit-ManualGen'
lang = sys.argv[2]          # 'AAE' or 'IndE'
if lang == 'AAE':
    model_name = 'bert-base-cased'
elif lang == 'IndE':
    model_name = 'bert-base-uncased'
out_dir = model_name + "_" + gen_method + "_" + lang

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__(transformers.PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        """
        Creates each single-feature model (where task == feature), and 
        has them share the same encoder transformer.
        """
        taskmodels_dict = {}
        shared_encoder = transformers.AutoModel.from_pretrained(
                model_name,
                config=transformers.AutoConfig.from_pretrained(model_name) )

        for task_name in head_type_list:
            head = torch.nn.Linear(768, 2)
            taskmodels_dict[task_name] = head
        
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    def forward(self, inputs, **kwargs):
        x = self.encoder(inputs)                # pass thru encoder once
        x = x.last_hidden_state[:,0,:]          # get CLS
        out_list = []
        for task_name,head in self.taskmodels_dict.items(): # pass thru each head
            out_list.append(self.taskmodels_dict[task_name](x))
        return torch.vstack(out_list)

class MultitaskTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = torch.transpose(inputs["labels"], 0, 1)
        labels = torch.flatten(labels)
        outputs = model(inputs["input_ids"])     # Forward pass
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, labels)

        return (loss, outputs) if return_outputs else loss

class CustomDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        sample = {"input_ids": text, "labels": label}
        return sample


def trainM(tokenizer, train_f):
    ## Load data
    features_dict = { "input_ids": [], "labels": [] }

    with open(train_f) as r:
        for line in r:
            lineSplit = line.strip().split("\t")
            tokenized = tokenizer.encode(lineSplit[0], max_length=64, pad_to_max_length=True, truncation=True)
            features_dict["input_ids"].append(torch.LongTensor(tokenized))
            features_dict["labels"].append(torch.tensor([float(x) for x in lineSplit[1:]], dtype=torch.long))

    features_dict["input_ids"] = torch.stack(features_dict["input_ids"])
    features_dict["labels"] = torch.stack(features_dict["labels"])
    features_dict = CustomDataset(features_dict["input_ids"], features_dict["labels"])

    ## Train
    trainer = MultitaskTrainer(
            model = multitask_model,
            args=transformers.TrainingArguments(
                output_dir="./models/"+out_dir,
                overwrite_output_dir=True,
                learning_rate=1e-4,
                do_train=True,
                warmup_steps=300,  # 2 steps per epoch when batch_size=64
                num_train_epochs=500,
                per_device_train_batch_size=64,
                save_steps=500,
            ),
            train_dataset=features_dict,
    )
    trainer.train()
    torch.save({'model_state_dict': multitask_model.state_dict()}, 
            "./models/"+out_dir+"/final.pt")


if __name__ == "__main__":
    
    train_file = "./data/"+gen_method+"/"+lang+".tsv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if lang == 'AAE':
        head_type_list=[
                "zero-poss",
                "zero-copula",
                "double-tense",
                "be-construction","resultant-done",
                "finna","come","double-modal",
                "multiple-neg","neg-inversion","n-inv-neg-concord","aint",
                "zero-3sg-pres-s","is-was-gen",
                "zero-pl-s",
                "double-object",
                "wh-qu" ]
    elif lang == 'IndE':
        head_type_list=[
                "foc_self", "foc_only", "left_dis", "non_init_exis", 
                "obj_front", "inv_tag", "cop_omis", "res_obj_pron",
                "res_sub_pron", "top_non_arg_con" ]
    
    multitask_model = MultitaskModel.create(
            model_name=model_name,
            head_type_list=head_type_list
        )
    multitask_model.to(device)

    ## Train on contrast set
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    trainM(tokenizer, train_file)

    





