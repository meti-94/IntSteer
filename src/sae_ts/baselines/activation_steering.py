# %%
import os
import torch
from transformer_lens import HookedTransformer
from functools import partial
import json
from ..steering.patch import patch_resid

# %%

def load_act_steer(dir_path):
    data_path = os.path.join(dir_path, "act_steer.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    pos_examples = data['pos']
    neg_examples = data['neg']
    if 'validation' in data:
        val_examples = data['validation']
    else:
        val_examples = []
    layer = data['layer']
    return pos_examples, neg_examples, val_examples, layer

def chat_to_prompt(chat, model):
    # chat is a list of dictionaries with keys 'role' and 'content'. 'role' is either 'user' or 'model'.
    res = ""
    for c in chat:
        res += f"{c['role']}\n{c['content']}\n"
    print(res)
    return model.to_tokens(res, prepend_bos=True)

def get_acts(examples, model, device, use_chat):
    total_acts = torch.zeros(model.cfg.n_layers, model.cfg.d_model, device=device)
    for i, pos in enumerate(examples):
        if use_chat:
            tokens = model.tokenizer.apply_chat_template(pos, return_tensors='pt')
            tokens = tokens[:, :-2] # remove the last two tokens <end_of_turn> and newline.
        else:
            tokens = model.to_tokens(pos, prepend_bos=True)
        points = [f"blocks.{n}.hook_resid_post" for n in range(model.cfg.n_layers)]
        logits, cache = model.run_with_cache(tokens, names_filter=points)
        acts = [cache[p][0].mean(0) for p in points]
        acts = torch.stack(acts, dim=0)
        total_acts += acts
    return total_acts / len(examples)

def steer_model(model, steer, layer, text, use_chat, scale=5):
    if use_chat:
        toks = model.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors='pt', add_generation_prompt=True)
    else:
        toks = model.to_tokens(text, prepend_bos=True)
    toks = toks.expand(10, -1)

    hp = f"blocks.{layer}.hook_resid_post"
    with model.hooks([(hp, partial(patch_resid, steering=steer[layer], scale=scale))]):
        gen_toks = model.generate(toks, max_new_tokens=40, use_past_kv_cache=True)
    return model.to_string(gen_toks)

def evaluate_layers(model, steer, eval_examples, scale=20):
    losses = []
    for layer in range(model.cfg.n_layers):
        loss_total = 0
        hp = f"blocks.{layer}.hook_resid_post"
        for pos in eval_examples:
            tokens = model.tokenizer.apply_chat_template(pos, return_tensors='pt')
            tokens = tokens[:, :-2]
            with model.hooks([(hp, partial(patch_resid, steering=steer[layer], scale=scale))]):
                loss = model.forward(tokens, return_type="loss")
            loss_total += loss.item()
        losses.append(loss_total / len(eval_examples))
    return losses

def get_activation_steering(model, pos_examples, neg_examples, device, layer=None):
    use_chat = isinstance(pos_examples[0], dict) # if dict, then it's in chat format.
    pos_acts = get_acts(pos_examples, model, device, use_chat)
    neg_acts = get_acts(neg_examples, model, device, use_chat)
    steer = pos_acts - neg_acts # shape (n_layers, d_model)
    if layer is not None:
        return steer[layer]
    return steer
