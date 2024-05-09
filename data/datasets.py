"""
Decoder-only Cognate Prediction
Lasse van den Berg, Adnan Bseisu
CPSC 477, Spring 2024

This file contains functions to create torch Tensor Datasets and batched Dataloaders.
There are two different classes.
TranslationData was used while we were experimenting with Autoregressive Decoders.
Direct simply process strings such that inputs are String_A and outputs are String_B.
"""


import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Optional, Tuple, Dict, List, Union

def TranslationData(tokenizer,
                    cognate_pair_list: List[list[str, str]],
                    batch_size: int,
                    val_size: int,
                    test_size: int,
                    copy: bool = False):
    
    assert len(cognate_pair_list) > val_size + test_size
    data_a = []
    data_b = []
    data_c = []
    for a, b in cognate_pair_list:
        
        input_string = "<" + a + "|" + b
        target_string = a + "|" + b + ">"
        if copy:
            input_string = "<" + a + "|" + a
            target_string = a + "|" + a + ">"
        sep_index = input_string.index("|")

        for i in range(len(b)+2):
            data_a.append(tokenizer(input_string[:sep_index+i]))
            data_b.append(tokenizer(target_string[:sep_index+i]))
            data_c.append(torch.tensor(sep_index))

    data_a = torch.stack(data_a)
    data_b = torch.stack(data_b)
    data_c = torch.stack(data_c)
    
    dataset = TensorDataset(data_a, data_b, data_c)
    dataset_size = len(dataset)
    
    train, val, test = random_split(dataset, [dataset_size - (val_size + test_size), val_size, test_size])
    
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)
    
    print(f"""
    Created Dataset
    
    Size: {dataset_size}
    Batch: {batch_size}
    Train: {dataset_size - (val_size + test_size)} Instances
    Val:   {val_size} Instances
    Test:  {test_size} Instances
    
    INPUT TENSOR:\t\tTARGET TENSOR:\t\tSEP INDEX\n
    Tensor({tokenizer.decode(data_a[0])})\t\tTensor({tokenizer.decode(data_b[0])})\t{data_c[0]}

    """)
    
    return train_dataloader, val_dataloader, test_dataloader


def Direct(tokenizer,
                    cognate_pair_list: List[list[str, str]],
                    batch_size: int,
                    val_size: int,
                    test_size: int,
                    copy: bool = False):
    
    assert len(cognate_pair_list) > val_size + test_size
    
    data_a = []
    data_b = []

    for a, b in cognate_pair_list:
        
        input_string = a
        target_string = b
        data_a.append(tokenizer(input_string))
        data_b.append(tokenizer(target_string))

    data_a = torch.stack(data_a)
    data_b = torch.stack(data_b)
    
    dataset = TensorDataset(data_a, data_b)
    dataset_size = len(dataset)
    
    train, val, test = random_split(dataset, [dataset_size - (val_size + test_size), val_size, test_size])
    
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)
    
    print(f"""
    Created Dataset
    
    Size: {dataset_size}
    Batch: {batch_size}
    Train: {dataset_size - (val_size + test_size)} Instances
    Val:   {val_size} Instances
    Test:  {test_size} Instances
    
    INPUT TENSOR:\t\tTARGET TENSOR:\t\tSEP INDEX\n
    Tensor({tokenizer.decode(data_a[0])})\t\tTensor({tokenizer.decode(data_b[0])})\t

    """)
    
    return train_dataloader, val_dataloader, test_dataloader