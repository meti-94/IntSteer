import os
import signal
from .utils import gen, get_feature_acts, get_scale

from transformer_lens import utils as tutils

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from transformer_lens import HookedTransformer
from ..steering.sae import JumpReLUSAE
from huggingface_hub import hf_hub_download
import numpy as np
from queue import Empty
import time

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

@torch.no_grad()
def load_model_and_sae(rank, big_model=False):
    device = torch.device(f"cuda:{rank}")
    if big_model:
        model = HookedTransformer.from_pretrained_no_processing("google/gemma-2-9b", device=device, dtype=torch.float16)
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-9b-pt-res",
            filename="layer_12/width_16k/average_l0_130/params.npz",
            force_download=False)
        loader_batch_size = 4  # Lower batch size for big models
    else:
        model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device, dtype=torch.float16)
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename="layer_12/width_16k/average_l0_82/params.npz",
            force_download=False)
        loader_batch_size = 64  # Higher batch size for small models
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae.to(device)
    # prep data
    data = tutils.load_dataset("NeelNanda/c4-code-20k", split="train")
    tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=32)
    tokenized_data = tokenized_data.shuffle(42)
    loader = DataLoader(tokenized_data, batch_size=loader_batch_size)
    return model, sae, loader


@torch.no_grad()
def worker(rank, world_size, task_queue, features, save_dir, big_model, scale=None):
    setup(rank, world_size)
    model, sae, loader = load_model_and_sae(rank, big_model=big_model)

    if big_model:
        # Lower batch sizes for big models
        baseline_samples = gen(model=model, n_batches=20, batch_size=32)
    else:
        # Higher batch sizes for small models
        baseline_samples = gen(model=model, n_batches=10)
    baseline_dist = get_feature_acts(model=model, sae=sae, tokens=baseline_samples, batch_size=32)
    print('got baseline dist')

    results = []
    try:
        while True:
            try:
                feature_index = task_queue.get(timeout=1)  # 1 second timeout
            except Empty:
                break
            start_time = time.time()  # Start timing
            feature = features[feature_index]
            if scale is None:
                if big_model:
                    opt_scale = get_scale(model=model,
                                          steer=feature.to(sae.W_dec.device),
                                          loader=loader,
                                          scales=list(range(0, 220, 20)),
                                          n_batches=8,
                                          target_loss=3.5,
                                          )
                else:
                    opt_scale = get_scale(model=model,
                                          steer=feature.to(sae.W_dec.device),
                                          loader=loader,
                                          scales=list(range(0, 220, 20)),
                                          target_loss=4,
                                          )
            else:
                opt_scale = scale

            if big_model:
                ft_samples = gen(model=model, steer=feature.to(sae.W_dec.device), scale=opt_scale, n_batches=2, batch_size=32)
            else:
                ft_samples = gen(model=model, steer=feature.to(sae.W_dec.device), scale=opt_scale)
            ft_dist = get_feature_acts(model=model, sae=sae, tokens=ft_samples, batch_size=32)
            diff = ft_dist - baseline_dist
            results.append({
                'effect': diff.cpu(),
                'used_feature': (feature * opt_scale).cpu(),
                'feature_index': feature_index
            })
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"processed {feature_index} in {elapsed_time:.2f} seconds")
    finally:
        torch.save(results, os.path.join(save_dir, f"partial_results_rank_{rank}.pt"))
        cleanup()

@torch.no_grad()
def main(features, save_dir, big_model=False):
    world_size = torch.cuda.device_count()
    num_features = features.shape[0]
    
    mp.set_start_method('spawn')
    
    task_queue = mp.Queue()
    
    # Populate task queue
    for i in range(num_features):
        task_queue.put(i)
    
    processes = []
    try:
        for rank in range(world_size):
            p = mp.Process(
                target=worker,
                args=(rank, world_size, task_queue, features.detach().clone(), save_dir, big_model))
            p.start()
            processes.append(p)
        
        # Wait for all processes to finish
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("Main process interrupted. Waiting for workers to save and exit...")
        for p in processes:
            p.join(timeout=5)
        print("All workers have exited.")
    finally:
        # Force terminate any remaining processes
        for p in processes:
            if p.is_alive():
                print(f"Force terminating process {p.pid}")
                os.kill(p.pid, signal.SIGTERM)
        task_queue.close()

    # Collect and combine partial results
    all_results = []
    for rank in range(world_size):
        partial_results = torch.load(os.path.join(save_dir, f"partial_results_rank_{rank}.pt"))
        all_results.extend(partial_results)
        os.remove(os.path.join(save_dir, f"partial_results_rank_{rank}.pt"))

    # Sort results by feature index
    all_results.sort(key=lambda x: x['feature_index'])

    # Separate effects and used features
    all_effects = [r['effect'] for r in all_results]
    all_used_features = [r['used_feature'] for r in all_results]

    # Convert lists to tensors
    all_effects = torch.stack(all_effects)
    all_used_features = torch.stack(all_used_features)

    print("Saving final results...")
    # Save final results
    torch.save(all_effects, os.path.join(save_dir, "all_effects.pt"))
    torch.save(all_used_features, os.path.join(save_dir, "used_features.pt"))
    print('done save')


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # # 16k
    # path_to_params = hf_hub_download(
    # repo_id="google/gemma-scope-2b-pt-res",
    # filename="layer_12/width_16k/average_l0_82/params.npz",
    # force_download=False)
    # params = np.load(path_to_params)
    # pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
    # sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    # sae.load_state_dict(pt_params)
    # sae._requires_grad = False

    # save_dir = "effects/G2_2B_L12/multi_16k_from_0"
    # os.makedirs(save_dir)
    # main(sae.W_dec, save_dir)

    ###################################3


    # # 32k
    # # layer_12/width_32k/average_l0_76
    # path_to_params = hf_hub_download(
    # repo_id="google/gemma-scope-2b-pt-res",
    # filename="layer_12/width_32k/average_l0_76/params.npz",
    # force_download=False)
    # params = np.load(path_to_params)
    # pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
    # sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    # sae.load_state_dict(pt_params)
    # sae._requires_grad = False

    # save_dir = "effects/G2_2B_L12/multi_32k_from_0"
    # os.makedirs(save_dir)
    # main(sae.W_dec, save_dir)


    ###################################

    # # sample and combine
    # # 16k
    # path_to_params = hf_hub_download(
    # repo_id="google/gemma-scope-2b-pt-res",
    # filename="layer_12/width_16k/average_l0_82/params.npz",
    # force_download=False)
    # params = np.load(path_to_params)
    # pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
    # sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    # sae.load_state_dict(pt_params)
    # sae._requires_grad = False
    # initial_features = sae.W_dec # shape d_model x d_sae

    # save_dir = "effects/G2_2B_L12/sample_and_combine_16k"
    # os.makedirs(save_dir)
    # n_samples = 32768
    # steering_vectors = []
    # for i in range(n_samples):
    #     sampled_idxs = torch.randint(0, initial_features.shape[0], (2,))
    #     scales = torch.rand(2)
    #     vec_to_add = (initial_features[sampled_idxs[0]] * scales[0] +
    #                   initial_features[sampled_idxs[1]] * scales[1])
    #     # normalize
    #     vec_to_add = vec_to_add / vec_to_add.norm()
    #     steering_vectors.append(vec_to_add)
    # steering_vectors = torch.stack(steering_vectors)

    # main(steering_vectors, save_dir)

    ###################################
    # # random vectors
    # save_dir = "effects/G2_2B_L12/random"
    # os.makedirs(save_dir)
    # n_samples = 32768
    # steering_vectors = torch.randn(n_samples, 2304)
    # # normalize
    # steering_vectors = steering_vectors / steering_vectors.norm(dim=-1, keepdim=True)
    # main(steering_vectors, save_dir)
    
    ###################################
    # # 9b
    save_dir = "effects/G2_9B_L12/131k_from_0_fixloss"
    os.makedirs(save_dir)

    # 131k
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-9b-pt-res",
        filename="layer_12/width_131k/average_l0_96/params.npz",
        force_download=False)
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae._requires_grad = False

    input_features = sae.W_dec[:32000]

    main(input_features, save_dir, big_model=True)


    # # ###################################
    # # # 2b
    # # # load adapter. Then find the 16k features that have the biggest effects
    # # # steer with the optimised features.
    
    # save_dir = "effects/G2_2B_L12/optimised_16k"
    # os.makedirs(save_dir)

    # from ft_effects.utils import LinearAdapter, get_sae
    # sae = get_sae()

    # # load adapter
    # adapter = LinearAdapter(sae.W_enc.shape[0], sae.W_enc.shape[1])
    # adapter.load_state_dict(torch.load("linear_adapter.pt"))
    # adapter.cpu()

    # b = adapter.W @ adapter.b
    # b = b/torch.norm(b)
    # input_features = []
    # for f in adapter.W.T:
    #     f_to_add = f/torch.norm(f) - b
    #     f_to_add = f_to_add/torch.norm(f_to_add)
    #     input_features.append(f_to_add)
    # input_features = torch.stack(input_features)
    # print(input_features.shape)

    # main(input_features, save_dir)


