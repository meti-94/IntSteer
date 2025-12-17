# %%

import sys
sys.path.append('/g/data/hn98/Mehdi/test/IntSteer/src')
import argparse
import os
import torch
from transformer_lens import HookedTransformer
from sae_lens import HookedSAETransformer
from functools import partial
import json
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download
from sae_lens import SAE
from transformers import AutoTokenizer, AutoModelForCausalLM

from sae_ts.steering.evals_utils import multi_criterion_evaluation
from sae_ts.steering.utils import normalise_decoder
from sae_ts.steering.sae import JumpReLUSAE
from sae_ts.ft_effects.utils import LinearAdapter, steer_model_addition, steer_model_rotation
from sae_ts.baselines.activation_steering import get_activation_steering, load_act_steer
from sae_ts.steering.utils import normalise_decoder


# %%

def load_sae_model(config):
    sae_load_method = config.get('sae_load_method', 'saelens')
    model_size = config.get('repo_id', '8b')
    if '9b' in model_size:
        model_size = 'gemma-2-9b'
    if '2b' in model_size:
        model_size = 'gemma-2-2b'
    if sae_load_method == 'saelens':
        sae, _, _ = SAE.from_pretrained(
            release=config['sae'],
            sae_id=config['layer'],
            device='cpu'
        )
        normalise_decoder(sae)
    elif sae_load_method == 'gemmascope':
        
        # We did not have access to the internet on our computation node
        if model_size == 'gemma-2-9b':
            sae = JumpReLUSAE(3584, 16384)
            sae.load("/g/data/hn98/Mehdi/hf_home/sae/gemma-2-9b/layer12.pth", map_location="cpu")
        if model_size == 'gemma-2-2b':
            sae = JumpReLUSAE(2304, 16384)
            sae.load("/g/data/hn98/Mehdi/hf_home/sae/gemma-2-2b/layer12.pth", map_location="cpu")
    else:
        raise ValueError(f"Unknown sae_load_method: {sae_load_method}")
    return sae


def load_sae_steer(path):
    # Read the configuration for SAE steering
    with open(os.path.join(path, "feature_steer.json"), 'r') as f:
        config = json.load(f)

    # Load SAE model
    sae = load_sae_model(config)
    vectors = []
    for ft_id, ft_scale in config['features']:
        vectors.append(sae.W_dec[ft_id] * ft_scale)
    vectors = torch.stack(vectors, dim=0)
    vec = vectors.sum(dim=0)
    vec = vec / torch.norm(vec, dim=-1, keepdim=True)
    hp = config['hp']
    layer = config['layer']

    return vec, hp, layer


def single_step_steer(adapter, target, bias_scale=1):
    # used for optimised steering
    steer_vec = adapter.W @ target.to(device)
    steer_vec = steer_vec / torch.norm(steer_vec) ######
    bias_vec = adapter.W @ adapter.b
    bias_vec = bias_vec / torch.norm(bias_vec) ######
    bias_vec = bias_vec * bias_scale
    steer = steer_vec - bias_vec
    steer = steer / torch.norm(steer, dim=-1, keepdim=True)
    return steer

@torch.no_grad()
def pinverse_steer(adapter, target, target_scale=1):
    target = target / torch.norm(target)
    target = target * target_scale
    target = target.to(device)
    W_pinv = torch.linalg.pinv(adapter.W)
    x_optimal = (target - adapter.b) @ W_pinv
    return x_optimal
    
def get_adapter_path(big_model=False, layer=12):
    """Get the path to the adapter weights, downloading from HF hub if needed."""
    model_name = "9b" if big_model else "2b"
    adapter_name = f"adapter_{model_name}_layer_{layer}.pt"
    if not os.path.exists(adapter_name):
        path = hf_hub_download(
            repo_id="schalnev/sae-ts-effects", 
            filename=adapter_name,
            repo_type="dataset"
        )
        return path
    return adapter_name

def load_optimised_steer(path, big_model=False):
    """Load SAE-TS configuration and get steering vector."""
    with open(os.path.join(path, "optimised_steer.json"), 'r') as f:
        config = json.load(f)
    
    layer = config['layer']
    sae = load_sae_model(config)
    adapter = LinearAdapter(2304, 16384)
    # adapter_path = get_adapter_path(big_model=big_model, layer=layer)
    adapter_path = '/mnt/weka/unsw.genaisim/unsw.mahdi/artifacts/saes/adapter_2b_layer_12.pt'
    # print(sae.W_enc.shape[0], sae.W_enc.shape[1])
    adapter.load_state_dict(torch.load(adapter_path, weights_only=True))
    adapter.to(device)
    # sys.exit()
    
    target = torch.zeros(adapter.W.shape[1]).to(device)
    for ft_id, ft_scale in config['features']:
        target[ft_id] = ft_scale

    vec = single_step_steer(adapter, target, bias_scale=1)
    return vec, config['hp'], layer

def load_pinv_steer(path, big_model=False):
    """Load pinverse steering configuration and get steering vector."""
    with open(os.path.join(path, "optimised_steer.json"), 'r') as f:
        config = json.load(f)
    layer = config['layer']
    sae = load_sae_model(config)
    # adapter = LinearAdapter(sae.W_enc.shape[0], sae.W_enc.shape[1])
    adapter = LinearAdapter(2304, 16384)
    # print(sae.W_enc.shape[0], sae.W_enc.shape[1])
    
    # adapter_path = get_adapter_path(big_model=big_model, layer=layer)
    adapter_path = '/mnt/weka/unsw.genaisim/unsw.mahdi/artifacts/saes/adapter_2b_layer_12.pt'
    # print(adapter_path)
    adapter.load_state_dict(torch.load(adapter_path, weights_only=True))
    adapter.to(device)
    
    target = torch.zeros(adapter.W.shape[1]).to(device)
    for ft_id, ft_scale in config['features']:
        target[ft_id] = ft_scale

    vec = pinverse_steer(adapter, target, target_scale=1)
    return vec, config['hp'], layer

def get_rotation_matrix_path(layer, big_model=False):
    """Get the path to the rotation matrix, downloading from HF hub if needed."""
    model_name = "9b" if big_model else "2b"
    rotation_name = f"R_dec_{model_name}_layer_{layer}.pt"
    correction_name = f"correction_bias_{model_name}_layer_{layer}.pt"
    
    if not os.path.exists(rotation_name):
        rotation_path = hf_hub_download(
            repo_id="schalnev/sae-ts-effects", 
            filename=rotation_name,
            repo_type="dataset"
        )
    else:
        rotation_path = rotation_name
        
    if not os.path.exists(correction_name):
        correction_path = hf_hub_download(
            repo_id="schalnev/sae-ts-effects",
            filename=correction_name,
            repo_type="dataset"
        )
    else:
        correction_path = correction_name
        
    return rotation_path, correction_path

def load_rotation_steer(path, big_model=False):
    """Load rotation steering configuration and get steering vector."""
    with open(os.path.join(path, "optimised_steer.json"), 'r') as f:
        config = json.load(f)
    
    layer = config['layer']
    hp = config['hp']

    sae = load_sae_model(config)
    sae.to(device)
    
    # Get paths to rotation matrix and bias, downloading if needed
    # rotation_path, correction_path = get_rotation_matrix_path(layer, big_model)
    rotation_path = '/mnt/weka/unsw.genaisim/unsw.mahdi/artifacts/saes/R_dec_2b_layer_12.pt'
    correction_path = '/mnt/weka/unsw.genaisim/unsw.mahdi/artifacts/saes/correction_bias_2b_layer_12.pt'
    
    R = torch.load(rotation_path)
    R = R.to(device)
    b = torch.load(correction_path)
    b = b.to(device)

    vectors = []
    for ft_id, ft_scale in config['features']:
        vectors.append(sae.W_dec[ft_id] * ft_scale)
    vectors = torch.stack(vectors, dim=0)
    vec = vectors.sum(dim=0)
    vec = vec / torch.norm(vec, dim=-1, keepdim=True)

    steer = R.T @ vec
    steer = steer / torch.norm(steer)
    steer = steer - b
    steer = steer / torch.norm(steer)

    return steer, hp, layer


def plot(path, coherence, score, product, scales, method, steering_goal_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scales, y=coherence, mode='lines', name='Coherence'))
    fig.add_trace(go.Scatter(x=scales, y=score, mode='lines', name='Score'))
    fig.add_trace(go.Scatter(x=scales, y=product, mode='lines', name='Coherence * Score'))
    fig.update_layout(
        title=f'Steering Analysis for {steering_goal_name} ({method})',
        xaxis_title='Scale',
        yaxis_title='Value',
        legend_title='Metric',
        yaxis=dict(range=[0, 1])
    )
    # Create a filename-friendly version of the steering goal name
    safe_name = steering_goal_name.replace(" ", "_")
    fig.write_image(os.path.join(path, f"scores_{safe_name}_{method}.png"), scale=2)



    
def addition_steer(model, steer, hp, path, method='activation_steering'):
    scales = list(range(0, 320, 20))
    with open(os.path.join(path, "criteria.json"), 'r') as f:
        criteria = json.load(f)

    # Read the steering goal name from criteria.json
    steering_goal_name = criteria[0].get('name', 'Unknown')

    all_texts = []
    avg_score = []
    avg_coh = []
    individual_scores = []
    individual_coherences = []
    individual_products = []

    for scale in tqdm(scales):
        texts, storage = steer_model_addition(model, steer, hp, default_prompt, scale=scale, n_samples=256)
        all_texts.append((scale, texts, storage))

        # We did not have access to the internet on our computation node
    # Log or store these results
    result = {
        'path': path,
        'method': method,
        'steering_goal_name': steering_goal_name,
        'max_product': 0,
        'scale_at_max': 0
    }
    with open(os.path.join(path, f"addition_{steering_goal_name}_{method}.json"), 'w') as f:
        json.dump(all_texts, f, indent=2)

    # plot(path, avg_coh, avg_score, product, scales, method, steering_goal_name)

    # Save data used to make the graphs
    graph_data = {
        'path': path,
        'method': method,
        'steering_goal_name': steering_goal_name,
        'scales': scales,
        'avg_coherence': avg_coh,
        'avg_score': 0,
        'product': 0,
        'individual_scores': 0,
        'individual_coherences': 0,
        'individual_products': 0
    }
    return result, graph_data

def rotation_steer(model, steer, hp, path, method='activation_steering'):
    scales = list(range(0, 320, 20))
    with open(os.path.join(path, "criteria.json"), 'r') as f:
        criteria = json.load(f)

    # Read the steering goal name from criteria.json
    steering_goal_name = criteria[0].get('name', 'Unknown')

    all_texts = []
    avg_score = []
    avg_coh = []
    individual_scores = []
    individual_coherences = []
    individual_products = []

    for scale in tqdm(scales):
        # the difference is in the steering scale 
        texts, storage = steer_model_rotation(model, steer, hp, default_prompt, scale=scale/320, n_samples=256)
        all_texts.append((scale, texts, storage))

        # We did not have access to the internet on our computation node
    # Log or store these results
    result = {
        'path': path,
        'method': method,
        'steering_goal_name': steering_goal_name,
        'max_product': 0,
        'scale_at_max': 0
    }
    with open(os.path.join(path, f"rotation_{steering_goal_name}_{method}.json"), 'w') as f:
        json.dump(all_texts, f, indent=2)

    # plot(path, avg_coh, avg_score, product, scales, method, steering_goal_name)

    # Save data used to make the graphs
    graph_data = {
        'path': path,
        'method': method,
        'steering_goal_name': steering_goal_name,
        'scales': scales,
        'avg_coherence': avg_coh,
        'avg_score': 0,
        'product': 0,
        'individual_scores': 0,
        'individual_coherences': 0,
        'individual_products': 0
    }
    return result, graph_data


def analyse_steer(model, steer, hp, path, method='activation_steering'):
    scales = list(range(0, 320, 20))
    with open(os.path.join(path, "criteria.json"), 'r') as f:
        criteria = json.load(f)

    # Read the steering goal name from criteria.json
    steering_goal_name = criteria[0].get('name', 'Unknown')

    all_texts = []
    avg_score = []
    avg_coh = []
    individual_scores = []
    individual_coherences = []
    individual_products = []

    for scale in tqdm(scales):
        texts = steer_model(model, steer, hp, default_prompt, scale=scale, n_samples=256)
        all_texts.append((scale, texts))

        score, coherence = multi_criterion_evaluation(
            texts,
            [criteria[0]['score'], criteria[0]['coherence']],
            prompt=default_prompt,
            print_errors=True,
        )

        score = [item['score'] for item in score]
        score = [(item - 1) / 9 for item in score]
        coherence = [item['score'] for item in coherence]
        coherence = [(item - 1) / 9 for item in coherence]

        # Compute the product for each sample. This is for variance analysis.
        products = [s * c for s, c in zip(score, coherence)]

        individual_scores.append(score)
        individual_coherences.append(coherence)
        individual_products.append(products)

        avg_score.append(sum(score) / len(score))
        avg_coh.append(sum(coherence) / len(coherence))

    # Compute the product at each scale
    product = [c * s for c, s in zip(avg_coh, avg_score)]

    # Find the maximum product and the corresponding scale
    max_product = max(product)
    max_index = product.index(max_product)
    max_scale = scales[max_index]

    # Log or store these results
    result = {
        'path': path,
        'method': method,
        'steering_goal_name': steering_goal_name,
        'max_product': max_product,
        'scale_at_max': max_scale
    }

    with open(os.path.join(path, f"generated_texts_{method}.json"), 'w') as f:
        json.dump(all_texts, f, indent=2)

    plot(path, avg_coh, avg_score, product, scales, method, steering_goal_name)

    # Save data used to make the graphs
    graph_data = {
        'path': path,
        'method': method,
        'steering_goal_name': steering_goal_name,
        'scales': scales,
        'avg_coherence': avg_coh,
        'avg_score': avg_score,
        'product': product,
        'individual_scores': individual_scores,
        'individual_coherences': individual_coherences,
        'individual_products': individual_products
    }
    print(f"Max product: {max_product} at scale {max_scale}")
    return result, graph_data

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_big_model', dest='big_model', action='store_false', help="Disable big model")
    parser.add_argument('--default_prompt', default="I think", help="Set the prompt")
    # default_prompt = "Surprisingly," 
    # This parses the arguments and assigns them to your variables in one go
    args = parser.parse_args()
    big_model, default_prompt = args.big_model, args.default_prompt
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # We did not have access to the internet on our computation node
    model2path = {
                    'google/gemma-2-2b':'/g/data/hn98/Mehdi/hf_home/hub/gemma-2-2b',
                    'google/gemma-2-9b':'/g/data/hn98/Mehdi/hf_home/hub/gemma-2-9b',
                }
    
    if big_model:
        model_name = "google/gemma-2-9b"
        hf_tokenizer = AutoTokenizer.from_pretrained(model2path[model_name])
        hf_model = AutoModelForCausalLM.from_pretrained(model2path[model_name])
        hf_model.to(device)
        model = HookedSAETransformer.from_pretrained_no_processing(model_name = model_name,
                                                                          hf_model=hf_model,
                                                                          tokenizer=hf_tokenizer,
                                                                          center_unembed=True,
                                                                          device=device)
    else:
        model_name = 'google/gemma-2-2b'
        hf_tokenizer = AutoTokenizer.from_pretrained(model2path[model_name])
        hf_model = AutoModelForCausalLM.from_pretrained(model2path[model_name])
        hf_model.to(device)
        model = HookedSAETransformer.from_pretrained_no_processing(model_name = model_name, 
                                                                          hf_model=hf_model, 
                                                                          tokenizer=hf_tokenizer,
                                                                          center_unembed=True,
                                                                          device=device)
if __name__ == "__main__":
    cfgs_dir = "steer_cfgs"

    if big_model:
        paths = [
            f"{cfgs_dir}/gemma2-9b/anger",
            f"{cfgs_dir}/gemma2-9b/christian_evangelist",
            f"{cfgs_dir}/gemma2-9b/conspiracy", 
            f"{cfgs_dir}/gemma2-9b/french",
            f"{cfgs_dir}/gemma2-9b/london",
            f"{cfgs_dir}/gemma2-9b/love",
            f"{cfgs_dir}/gemma2-9b/praise",
            f"{cfgs_dir}/gemma2-9b/want_to_die",
            f"{cfgs_dir}/gemma2-9b/wedding",
        ]
    else:
        paths = [
            f"{cfgs_dir}/gemma2/anger",
            f"{cfgs_dir}/gemma2/christian_evangelist",
            f"{cfgs_dir}/gemma2/conspiracy",
            f"{cfgs_dir}/gemma2/french",
            f"{cfgs_dir}/gemma2/london",
            f"{cfgs_dir}/gemma2/love",
            f"{cfgs_dir}/gemma2/praise",
            f"{cfgs_dir}/gemma2/want_to_die",
            f"{cfgs_dir}/gemma2/wedding",
        ]

    results = []
    graph_data_list = []

    for path in paths:
        print(path)
        print("Activation Addition Steering")
        pos_examples, neg_examples, val_examples, layer = load_act_steer(path)
        steer = get_activation_steering(model, pos_examples, neg_examples, device=device, layer=layer)
        steer = steer / torch.norm(steer, dim=-1, keepdim=True)
        hp = f"blocks.{layer}.hook_resid_post"
        result, graph_data = addition_steer(model, steer, hp, path, method='ActSteer')
        results.append(result)
        graph_data_list.append(graph_data)

        print("Activation Rotational Steering")
        pos_examples, neg_examples, val_examples, layer = load_act_steer(path)
        steer = get_activation_steering(model, pos_examples, neg_examples, device=device, layer=layer)
        steer = steer / torch.norm(steer, dim=-1, keepdim=True)
        hp = f"blocks.{layer}.hook_resid_post"
        result, graph_data = rotation_steer(model, steer, hp, path, method='ActSteer')
        results.append(result)
        graph_data_list.append(graph_data)

        print("SAE Addition Steering")
        steer, hp, layer = load_sae_steer(path)
        steer = steer.to(device)
        result, graph_data = addition_steer(model, steer, hp, path, method='SAE')
        results.append(result)
        graph_data_list.append(graph_data)

        print("SAE Rotation Steering")
        steer, hp, layer = load_sae_steer(path)
        steer = steer.to(device)
        result, graph_data = rotation_steer(model, steer, hp, path, method='SAE')
        results.append(result)
        graph_data_list.append(graph_data)
