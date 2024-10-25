import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import numpy as np



from .get_embedding import generate_embeddings
from .get_network_details import plot_degree_distribution,compute_network_distribution_property 


def plot_trajectory(model,tokenizer,model_name,input_number_list, few_shot_example_prompts,few_shot=0, svd=True, threshold=0.1):
    for x, y in input_number_list:
        if few_shot != 0:
            text = "\n".join(few_shot_example_prompts[0:few_shot])+f"\nQuestion: What is the sum of {x} and {y}?\nAnswer: "
        else:
            text = f"Question: What is the sum of {x} and {y}?\nAnswer: "
        inputs = tokenizer(text, return_tensors='pt')
        token_ids = inputs.input_ids 
        print("Input Tokens",inputs.input_ids.shape,text)
        _,attention, _, generated_text = generate_embeddings(inputs, model)
        num_layers = attention.shape[0] 
        cols = 8
        rows = (num_layers + cols - 1) // cols  # Calculate the number of rows needed
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        fig_dist, axes_dist = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        fig_degree_dist, axes_degree_dist = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()
        axes_dist = axes_dist.flatten()
        axes_degree_dist = axes_degree_dist.flatten()
        distribution_value = []
        for i  in range(num_layers):
            adj_matrix =attention[i] 
            G = nx.from_numpy_array(adj_matrix)
            mapping = {i:token for i, (token,id) in enumerate(zip(tokenizer.convert_ids_to_tokens(token_ids.flatten().tolist()),token_ids.flatten().tolist() ))}
            G = nx.relabel_nodes(G, mapping)
            edges = G.edges(data=True)
            edge_weights = [d['weight'] for (u, v, d) in edges]
            min_weight = min(edge_weights)
            max_weight = max(edge_weights)
            normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
            norm_adj_matrix = (adj_matrix-np.mean(edge_weights)) / (max_weight - min_weight) 
            # Plot the graph on the corresponding subplot
            ax = axes[i]
            ax_dist = axes_dist[i]
            ax_degree_dist = axes_degree_dist[i]
            pos = nx.circular_layout(G)  # Get a layout for nodes
            # Draw the nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightblue", node_size=200)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color="black", font_weight="bold")
            
            # Draw the edges with varying transparency based on the normalized edge weights
            for (u, v, d), alpha in zip(edges, normalized_weights):
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, width=2, alpha=alpha, edge_color="black")
            ax.set_title(f'Layer {i+1}')
            sns.heatmap(norm_adj_matrix, annot=False,ax=ax_dist, cmap="crest")
            ax_dist.set_title(f'Layer {i+1}')
            G_unweighted, ax_degree_dist= plot_degree_distribution(G,threshold,ax_degree_dist)
            ax_degree_dist.set_title(f'Layer {i+1} | Threshold {threshold}') 
            hetero, gamma = compute_network_distribution_property(G_unweighted)
            distribution_value.append([i,hetero, gamma])
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            fig_dist.delaxes(axes_dist[j])
            fig_degree_dist.delaxes(axes_degree_dist[j]) 
        fig.suptitle(f"Model {model_name} | Few Show {few_shot} | threshold: {threshold}")
        fig_dist.suptitle(f"Model {model_name} | Few Show {few_shot} | threshold: {threshold}")
        fig_degree_dist.suptitle(f"Model {model_name} \n {text}")
        plt.tight_layout()
        plt.show()
        distribution_value = np.array(distribution_value)
        fig, [ax0,ax1] = plt.subplots(figsize=(10,5),ncols=2)
        ax0.set_title("heterogeneity")
        ax1.set_title("gamma")
        ax0.plot(distribution_value[:,0],distribution_value[:,1])
        ax1.plot(distribution_value[:,0],distribution_value[:,2])
        ax0.set_xlabel("Layers")
        ax0.set_ylabel("heterogeneity")
        ax1.set_ylabel("gamma") 
        plt.show()


def plot_trajectory_random(model,tokenizer,model_name,input_number_list,few_shot_example_prompts, few_shot=0, svd = True,threshold= 0.1):
    vocab_size = len(tokenizer)
    for idx, (x, y) in enumerate(input_number_list):
        input_default = tokenizer("\n".join(few_shot_example_prompts[0:few_shot])+f"\nWhat is the sum of {x} and {y}?", return_tensors='pt')#[1]
        torch.manual_seed(idx*123123)
        random_token_ids = torch.randint(0,vocab_size,input_default.input_ids.shape)
        random_tokens = tokenizer.convert_ids_to_tokens(random_token_ids.flatten().tolist())
        input_random = {"input_ids":random_token_ids, "attention_mask":input_default.attention_mask}
        print("Input Tokens",random_token_ids.shape,' '.join(random_tokens))
        _,attention, _, generated_text = generate_embeddings(input_random, model)
        num_layers = attention.shape[0] 
        cols = 8
        rows = (num_layers + cols - 1) // cols  # Calculate the number of rows needed
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        fig_dist, axes_dist = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        fig_degree_dist, axes_degree_dist = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()
        axes_dist = axes_dist.flatten()
        axes_degree_dist = axes_degree_dist.flatten()
        distribution_value = []
        for i  in range(num_layers):
            adj_matrix =attention[i] 
            G = nx.from_numpy_array(adj_matrix)
            mapping = {i:token for i, (token,id) in enumerate(zip(tokenizer.convert_ids_to_tokens(random_tokens.flatten().tolist()),random_tokens.flatten().tolist() ))}
            G = nx.relabel_nodes(G, mapping)
            edges = G.edges(data=True)
            edge_weights = [d['weight'] for (u, v, d) in edges]
            min_weight = min(edge_weights)
            max_weight = max(edge_weights)
            normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
            norm_adj_matrix = (adj_matrix-np.mean(edge_weights)) / (max_weight - min_weight) 
            # Plot the graph on the corresponding subplot
            ax = axes[i]
            ax_dist = axes_dist[i]
            ax_degree_dist = axes_degree_dist[i] 
            pos = nx.circular_layout(G)  # Get a layout for nodes
            
            # Draw the nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightblue", node_size=200)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color="black", font_weight="bold")
            
            # Draw the edges with varying transparency based on the normalized edge weights
            for (u, v, d), alpha in zip(edges, normalized_weights):
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, width=2, alpha=alpha, edge_color="black")
            ax.set_title(f'Layer {i+1}')
            sns.heatmap(norm_adj_matrix, annot=False,ax=ax_dist, cmap="crest")
            ax_dist.set_title(f'Layer {i+1}')
            G_unweighted, ax_degree_dist= plot_degree_distribution(G,threshold,ax_degree_dist)
            ax_degree_dist.set_title(f'Layer {i+1} | Threshold {threshold}')
            hetero, gamma = compute_network_distribution_property(G_unweighted)
            distribution_value.append([i,hetero, gamma]) 
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            fig_dist.delaxes(axes_dist[j])
            fig_degree_dist.delaxes(axes_degree_dist[j]) 
        fig.suptitle(f"Model {model_name} | Few Show {few_shot} | threshold: {threshold}")
        fig_dist.suptitle(f"Model {model_name} | Few Show {few_shot} | threshold: {threshold}")
        #fig_dist.suptitle(f"Model {model_name} \n {' '.join(random_tokens)}")
        plt.tight_layout()
        plt.show()
        distribution_value = np.array(distribution_value)
        fig, [ax0,ax1] = plt.subplots(figsize=(10,5),ncols=2)
        ax0.set_title("heterogeneity")
        ax1.set_title("gamma")
        ax0.plot(distribution_value[:,0],distribution_value[:,1])
        ax1.plot(distribution_value[:,0],distribution_value[:,2])
        ax0.set_xlabel("Layers")
        ax0.set_ylabel("heterogeneity")
        ax1.set_ylabel("gamma") 
        plt.show()