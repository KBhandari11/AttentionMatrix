import torch
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2Model,GPT2LMHeadModel, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch.nn as nn
from sklearn.decomposition import PCA, TruncatedSVD
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio.v2 as imageio
import os
from plot import plot
import random
import seaborn as sns
from numpy import linalg as LA
import pandas as pd

from utils.get_embedding import generate_embeddings
from utils.get_network_details import get_unweighted,compute_network_distribution_property 


def get_trajectory(model,tokenizer,model_name,input_number_list, few_shot=0, svd=True, threshold=0.1, random =False):
    data = []
    few_shot_example_prompts = ["Question: What is the sum of 5 and 6?\n Answer: 11","Question: What is the sum of 1 and 9?\n Answer: 10", "Question: What is the sum of 0 and 0?\n Answer: 0"]
    for idx, (x, y) in enumerate(input_number_list):
            if few_shot != 0:
                text = "\n".join(few_shot_example_prompts[0:few_shot])+f"\nQuestion: What is the sum of {x} and {y}?\nAnswer: "
            else:
                text = f"Question: What is the sum of {x} and {y}?\nAnswer: "
            inputs = tokenizer(text, return_tensors='pt')
            torch.manual_seed(idx*123123)
            if random:
                vocab_size = len(tokenizer)
                random_token_ids = torch.randint(0,vocab_size,inputs.input_ids.shape)
                random_tokens = tokenizer.convert_ids_to_tokens(random_token_ids.flatten().tolist())
                inputs = {"input_ids":random_token_ids, "attention_mask":inputs.attention_mask}
                text = ' '.join(random_tokens)
            token_ids = inputs.input_ids 
            #print("Input Tokens",inputs.input_ids.shape,text)
            mapping = {i:token for i, (token,id) in enumerate(zip(tokenizer.convert_ids_to_tokens(token_ids.flatten().tolist()),token_ids.flatten().tolist() ))}
            _,attention, _, generated_text = generate_embeddings(inputs,model,tokenizer)
            num_layers = attention.shape[0] 
            layers = []
            heterogenity_value = []
            gamma_value = []
            eigen_values = []
            for i  in range(num_layers):
                adj_matrix =attention[i] 
                G = nx.from_numpy_array(adj_matrix)
                G = nx.relabel_nodes(G, mapping)
                G_unweighted = get_unweighted(G, threshold)
                hetero, gamma = compute_network_distribution_property(G_unweighted)

                layers.append(i) 
                heterogenity_value.append(hetero)
                gamma_value.append(gamma)
                eigenvalues, _ = LA.eig(adj_matrix)
                eigen_values.append(eigenvalues)
            data.append({"data":text,
                         "model":model_name,
                         "few_shot":few_shot,
                         "threshold":threshold,
                         "prompt":text,
                         "token_ids":token_ids.flatten().tolist(),
                         "layers":layers,
                         "gamma":gamma_value,
                         "heterogenity":heterogenity_value,
                        "eigen_values":eigen_values,
                        "random":random,
                        "generated_text":generated_text 
                         })     
    return data

def get_model(model_name):
    if model_name == 'gpt2-xl': 
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_name == 'meta-llama/Llama-3.1-8B':
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
def run_data():
    few_shot_example_prompts = ["Question: What is the sum of 5 and 6?\nAnswer: 11","Question: What is the sum of 1 and 9?\nAnswer: 10", "Question: What is the sum of 0 and 0?\nAnswer: 0"]
    data = []
    for model_name in ['gpt2-xl','meta-llama/Llama-3.1-8B']:
        model, tokenizer = get_model(model_name)
        for few_shot in [0,1,2,3]:
            for threshold in [0.1,0.2,0.3]:
                for random in [False, True]:
                    value = get_trajectory(model,tokenizer,model_name,[(1,2),(10,13),("a", "b"),(12315645616,1231564516816184)],few_shot_example_prompts,few_shot=few_shot, svd=False, threshold=threshold, random=random)
                    data += value
                    df = pd.DataFrame(data)
                    df.to_csv("./data/distribution_statistics.csv",index=False)