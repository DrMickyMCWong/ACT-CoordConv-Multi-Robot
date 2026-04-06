from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG # must import first

import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from training.utils import *

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1a')
args = parser.parse_args()
task = args.task

# configs
task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)

# device
device = os.environ['DEVICE']


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

def plot_history(train_history, validation_history, ckpt_dir, seed):
    if not train_history:
        return
    metric_keys = train_history[0]['metrics'].keys()
    for key in metric_keys:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_steps = [entry['step'] for entry in train_history]
        train_values = [entry['metrics'][key].item() for entry in train_history]
        plt.plot(train_steps, train_values, label='train')
        if validation_history:
            val_steps = [entry['step'] for entry in validation_history]
            val_values = [entry['metrics'][key].item() for entry in validation_history]
            plt.plot(val_steps, val_values, label='validation')
        plt.xlabel('training step')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f'Saved plots to {ckpt_dir}')


def run_validation(policy, val_dataloader):
    policy.eval()
    epoch_dicts = []
    with torch.inference_mode():
        for data in val_dataloader:
            forward_dict = forward_pass(data, policy)
            epoch_dicts.append(forward_dict)
    return compute_dict_mean(epoch_dicts)


def train_bc(train_dataloader, val_dataloader, policy_config):
    # load policy
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)

    # load optimizer
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    # create checkpoint dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    grad_clip = train_cfg.get('grad_clip_norm', 1.0)
    max_steps = train_cfg.get('max_steps')
    eval_freq = train_cfg.get('eval_freq')
    log_freq = train_cfg.get('log_freq')
    save_freq = train_cfg.get('save_freq')

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    global_step = 0

    for epoch in tqdm(range(train_cfg['num_epochs'])):
        if max_steps is not None and global_step >= max_steps:
            break
        print(f'\nEpoch {epoch}')
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            if max_steps is not None and global_step >= max_steps:
                break
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            metrics_detached = detach_dict(forward_dict)
            train_history.append({'step': global_step, 'metrics': metrics_detached})

            if log_freq and global_step % log_freq == 0:
                summary_string = ' '.join([f"{k}: {v.item():.3f}" for k, v in metrics_detached.items()])
                print(f'Step {global_step}: {summary_string}')

            if eval_freq and global_step % eval_freq == 0:
                val_summary = run_validation(policy, val_dataloader)
                validation_history.append({'step': global_step, 'metrics': val_summary})
                val_loss = val_summary['loss']
                summary_string = ' '.join([f"{k}: {v.item():.3f}" for k, v in val_summary.items()])
                print(f'[Eval @ step {global_step}] {summary_string}')
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_ckpt_info = (global_step, min_val_loss, deepcopy(policy.state_dict()))
                    print(f'New best checkpoint at step {global_step} (val loss {val_loss:.4f})')
                policy.train()

            if save_freq and global_step % save_freq == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"policy_step_{global_step}_seed_{train_cfg['seed']}.ckpt")
                torch.save(policy.state_dict(), ckpt_path)
                plot_history(train_history, validation_history, checkpoint_dir, train_cfg['seed'])

        if log_freq:
            print(f'Completed epoch {epoch} -- total steps {global_step}')

    ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    plot_history(train_history, validation_history, checkpoint_dir, train_cfg['seed'])

    if best_ckpt_info is not None:
        best_step, best_val_loss, best_state_dict = best_ckpt_info
        best_ckpt_path = os.path.join(checkpoint_dir, f"policy_best_step_{best_step}_val_{best_val_loss:.4f}.ckpt")
        torch.save(best_state_dict, best_ckpt_path)
        print(f"Best checkpoint saved at step {best_step} with validation loss {best_val_loss:.4f}")

if __name__ == '__main__':
    # set seed
    set_seed(train_cfg['seed'])
    # create ckpt dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)
   # number of training episodes
    data_dir = os.path.join(task_cfg['dataset_dir'], task)
    num_episodes = len([f for f in os.listdir(data_dir) if f.endswith('.hdf5')])

    # load data
    train_dataloader, val_dataloader, stats, _ = load_data(data_dir, num_episodes, task_cfg['camera_names'],
                                                            train_cfg['batch_size_train'], train_cfg['batch_size_val'])
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # train
    train_bc(train_dataloader, val_dataloader, policy_config)