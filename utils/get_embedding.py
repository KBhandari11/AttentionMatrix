import torch
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA, TruncatedSVD
import torch.nn.functional as F
from numpy import linalg as LA

def generate_embeddings(inputs, model,tokenizer, pca=False):
    # Get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        hidden_states = outputs.hidden_states  # A tuple of layers
        attentions = outputs.attentions
        hidden_states = hidden_states[-1]
        generated_tokens = model.generate(
            input_ids=inputs['input_ids'],  
            attention_mask=inputs["attention_mask"],
            max_length=50, 
            num_return_sequences=1
        )
        generated_text = tokenizer.batch_decode(sequences=generated_tokens, skip_special_tokens=False)  
    new_hidden_states_pca = []
    new_hidden_states =[]
    new_attention_states = []
    i = 0
    for layer_embeddings in hidden_states:
        i+=1
        new_hidden_states.append(layer_embeddings[0].numpy())
        if pca:
            pca = PCA(n_components=2)
            new_hidden_states_pca.append(pca.fit_transform(layer_embeddings[0].numpy()))
    i = 0
    for attention_value in attentions:
        new_attention_states.append(attention_value[0,0].numpy())#.mean(axis=0))
        i+=1
    org,attention = np.array(new_hidden_states),np.array(new_attention_states)  
    if pca:
        org_pca = np.array(new_hidden_states_pca)
        z = np.transpose(org_pca, (1,0,2))
        return (org,attention, org_pca,generated_text)
    else: 
        return (org,attention,None,generated_text)
    

