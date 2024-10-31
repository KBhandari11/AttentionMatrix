import torch
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2Model,GPT2LMHeadModel, AutoModelForCausalLM
import networkx as nx
import random 

import seaborn as sns
import pandas as pd

from utils.get_embedding import generate_embeddings
from utils.get_network_details import get_unweighted,compute_network_distribution_property 

def get_trajectory(model,tokenizer,model_name,few_shot_example,input_number_list,equation,equation_function,difficulty, few_shot=0, svd=True, threshold=0.1):
    data = []
    for idx, numbers in enumerate(input_number_list):
            text = "Find the missing value(?):\n"+",".join([f"({x},{y},{equation_function(x,y)})" for x,y in few_shot_example[:few_shot]]) +"," if few_shot>0 else "" + f"({numbers[0]},{numbers[1]},?)"
            inputs = tokenizer(text, return_tensors='pt')
            torch.manual_seed(idx*123123)
            if False:
                vocab_size = len(tokenizer)
                random_token_ids = torch.randint(0,vocab_size,inputs.input_ids.shape)
                random_tokens = tokenizer.convert_ids_to_tokens(random_token_ids.flatten().tolist())
                inputs = {"input_ids":random_token_ids, "attention_mask":inputs.attention_mask}
                text = ' '.join(random_tokens)
            token_ids = inputs["input_ids"] 
            #print("Input Tokens",inputs.input_ids.shape,text)
            mapping = {i:token for i, (token,id) in enumerate(zip(tokenizer.convert_ids_to_tokens(token_ids.flatten().tolist()),token_ids.flatten().tolist() ))}
            _,attention, _, generated_text = generate_embeddings(inputs,model,tokenizer)
            num_layers = attention.shape[0]
            for i  in range(num_layers):
                adj_matrix =attention[i] 
                G = nx.from_numpy_array(adj_matrix)
                G = nx.relabel_nodes(G, mapping)
                G_unweighted = get_unweighted(G, threshold)
                deg_weight, deg_unweight,hetero, gamma, eigen_values_Adj,eigen_values_Lap  = compute_network_distribution_property(G,G_unweighted,adj_matrix)

                data.append({"model":model_name,
                         "few_shot":few_shot,
                         "input_question": numbers,
                         "threshold":threshold,
                         "prompt":text,
                         "token_ids":token_ids.flatten().tolist(),
                         "layer":i,
                         "gamma":gamma,
                         "heterogenity":hetero,
                        "eigen_values_adj":eigen_values_Adj,
                        "eigen_values_lap":eigen_values_Lap,
                        "generated_text":generated_text,
                        "deg_weight":deg_weight,  
                        "deg_unweight":deg_unweight,  
                        "difficulty":difficulty,
                        "equation":equation
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
    random.seed(123123)
    complexity_functions = {
                            "x+y":{"equation":lambda x,y: x+y,"difficulty":1}, 
                            "x-y":{"equation":lambda x,y: x-y,"difficulty":1}, 
                            "x+2y":{"equation":lambda x,y: x+2*y,"difficulty":2}, 
                            "2x+y":{"equation":lambda x,y: 2*x+y,"difficulty":2}, 
                            "2x+3y":{"equation":lambda x,y: 2*x+3*y,"difficulty":3}, 
                            "3x+2y":{"equation":lambda x,y: 3*x+2*y,"difficulty":3}, 
                            "xy":{"equation":lambda x,y: x*y,"difficulty":4}, 
                            "2xy":{"equation":lambda x,y: 2*x*y,"difficulty":4}, 
                            "3xy+2":{"equation":lambda x,y: 3*x*y+2,"difficulty":5}, 
                            "2xy-5":{"equation":lambda x,y: 2*x*y-5,"difficulty":5},
                            }
    few_shot_example = [(random.randint(0, 15), random.randint(0, 15)) for i in range(10)]
    print(few_shot_example)
    input_list = [(0,0),(1,10),(1,5),(2,9),(9,9),(12,3),('a',"b"),(11212312,1233123)]
    data = []
    for model_name in ['meta-llama/Llama-3.1-8B','gpt2-xl']:
        model, tokenizer = get_model(model_name)
        for few_shot in [0,1,2,3,4,5,6,7,8,9]:
            for threshold in [0.1,0.2,0.3]:
                for equation, detail in complexity_functions.items():
                    print(model_name,few_shot,threshold,equation)
                    value = get_trajectory(model,tokenizer,model_name,few_shot_example,input_list,equation,detail["equation"],difficulty=detail["difficulty"],few_shot=few_shot, svd=False, threshold=threshold)
                    data += value
                    df = pd.DataFrame(data)
                    df.to_csv("./data/distribution_statistics_degree_seq.csv",index=False)

if __name__ == '__main__':
    run_data()