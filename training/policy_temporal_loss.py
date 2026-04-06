import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        # Add temporal weight parameter
        self.temporal_weight = args_override.get('temporal_weight', 0.5)
        
        self.use_dynamic_temporal = args_override.get('use_dynamic_temporal', False)  # NEW FLAG
        
        print(f'KL Weight {self.kl_weight}')
        print(f'Temporal Weight {self.temporal_weight}')
        print(f'Dynamic Temporal Loss: {self.use_dynamic_temporal}')
        
    def compute_loss_with_temporal_weighting(self, a_hat, actions, is_pad):
        """
        Compute L1 loss with temporal derivative penalty to reduce over-smoothing
        """
        # Standard L1 loss with padding mask
        l1_loss = F.l1_loss(a_hat, actions, reduction='none')
        l1_loss = (l1_loss * ~is_pad.unsqueeze(-1)).mean()
        
        # Temporal derivative loss (penalize smoothing) - only if sequence length > 1
        if actions.shape[1] > 1:
            # Compute differences (velocities)
            action_diff = torch.diff(actions, dim=1)  # [batch, seq-1, action_dim]
            pred_diff = torch.diff(a_hat, dim=1)
            
            # Create mask for temporal differences (exclude last timestep from padding)
            temporal_mask = ~is_pad[:, 1:]  # [batch, seq-1]
            
            # Compute temporal loss with masking
            temporal_loss = F.l1_loss(pred_diff, action_diff, reduction='none')
            temporal_loss = (temporal_loss * temporal_mask.unsqueeze(-1)).mean()
        else:
            temporal_loss = torch.tensor(0.0, device=actions.device)
        
        return l1_loss, temporal_loss
    
    def compute_dynamic_temporal_loss(self, a_hat, actions, is_pad):
        """
        Compute dynamic temporal loss focusing on velocity consistency
        L_temp = Σ(v̂_t - v_t)² where v_t = x_t - x_{t-1}
        """
        # Standard L1 loss with padding mask
        l1_loss = F.l1_loss(a_hat, actions, reduction='none')
        l1_loss = (l1_loss * ~is_pad.unsqueeze(-1)).mean()
        
        if actions.shape[1] > 1:
            # Compute velocities
            action_velocity = torch.diff(actions, dim=1)      # v_t = x_t - x_{t-1}
            pred_velocity = torch.diff(a_hat, dim=1)          # v̂_t = x̂_t - x̂_{t-1}
            
            temporal_mask = ~is_pad[:, 1:]
            
            # Dynamic temporal loss: L2 on velocity differences
            velocity_diff = pred_velocity - action_velocity    # (v̂_t - v_t)
            dynamic_temporal_loss = (velocity_diff ** 2)       # (v̂_t - v_t)²
            dynamic_temporal_loss = (dynamic_temporal_loss * temporal_mask.unsqueeze(-1)).mean()
        else:
            dynamic_temporal_loss = torch.tensor(0.0, device=actions.device)
        
        return l1_loss, dynamic_temporal_loss

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            
            #all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            #l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            # Choose between basic and dynamic temporal loss
            if self.use_dynamic_temporal:
                l1, temporal_loss = self.compute_dynamic_temporal_loss(a_hat, actions, is_pad)
            else:
                l1, temporal_loss = self.compute_loss_with_temporal_weighting(a_hat, actions, is_pad)
                
            loss_dict['l1'] = l1
            loss_dict['temporal'] = temporal_loss
            loss_dict['kl'] = total_kld[0]
            # loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            
            loss_dict['loss'] = (loss_dict['l1'] + 
                               self.temporal_weight * loss_dict['temporal'] + 
                               loss_dict['kl'] * self.kl_weight)
            
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld