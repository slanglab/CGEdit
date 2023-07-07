import transformers 
import torch
import torch.nn as nn
import numpy as np
import dataclasses
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from typing import List, Union, Dict
import re
import sys


gen_method = sys.argv[1]    # 'CGEdit' or 'CGEdit-ManualGen'
lang = sys.argv[2]          # 'AAE' or 'IndE'
test_set = sys.argv[3]
if lang == 'AAE':
    model_name = 'bert-base-cased'
elif lang == 'IndE':
    model_name = 'bert-base-uncased'
model_dir = model_name + "_" + gen_method + "_" + lang
out_dir = model_dir + "_" + test_set + ".tsv"

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__(transformers.PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        """
        Creates each single-task model (where task == feature), and 
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
        x = self.encoder(inputs)
        x = x.last_hidden_state[:,0,:]
        out_list = []
        for task_name, head in self.taskmodels_dict.items():
            out_list.append(self.taskmodels_dict[task_name](x))
        return torch.vstack(out_list)

def eval_dataloader(eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)

    data_loader = DataLoader(eval_dataset,
                batch_size=64,
                sampler=eval_sampler
                )

    return data_loader

class CustomDataset(Dataset):
    def __init__(self, text):
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {"input_ids": self.text[idx] }


def testM(tokenizer, model, test_f):
    features_dict = {"input_ids": []}

    with open(test_f) as r: 
        for line in r:
            if len(line.split()) < 2: continue
            tokenized = tokenizer.encode(line.strip(), max_length=64, pad_to_max_length=True, truncation=True)
            features_dict["input_ids"].append(torch.LongTensor(tokenized))

    features_dict["input_ids"] = torch.stack(features_dict["input_ids"])
    features_dict = CustomDataset(features_dict["input_ids"])

    # For each head/feature, predict on all sentences if feature is present
    dataloader = eval_dataloader(features_dict)
    with open("./data/results/"+out_dir,'a') as f:
        for steps, inputs in enumerate(dataloader):
            for ex in inputs["input_ids"]:
                with torch.no_grad():
                    output = model(ex.unsqueeze(0).to(device))
                output = torch.nn.functional.softmax(output, dim=1)
                output = [str(float(x[1])) for x in output]
                sent = tokenizer.decode(ex).split()
                sent = [e for e in sent if e != '[PAD]' and e != '[CLS]' and e != '[SEP]']
                f.write(" ".join(sent)+"\t"+"\t".join(output)+"\n")



if __name__ == "__main__":
    
    test_file = "./data/" + test_set + ".csv"
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
                
    ## Test
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    checkpoint = torch.load(model_dir+"/final.pt", map_location=device)
    multitask_model.load_state_dict(checkpoint['model_state_dict'])
    multitask_model.eval()
    multitask_model.to(device)
    testM(tokenizer, multitask_model, test_file)






