import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed
'''
Dynamic-matching approach: 
"Match the expert's pattern of acceleration,
braking, and steering" → Results in more natural, human-like driving that adapts to different situations
    'kl_weight': 10.0,
    'temporal_weight': 0.5,
    'dynamic_matching_weight': 0.3,  # New parameter
    # ... other args
'''
class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        # Add temporal weight parameter
        self.temporal_weight = args_override.get('temporal_weight', 0.5)
        # Add dynamic matching weight
        self.dynamic_matching_weight = args_override.get('dynamic_matching_weight', 0.3)
        
        print(f'KL Weight {self.kl_weight}')
        print(f'Temporal Weight {self.temporal_weight}')
        print(f'Dynamic Matching Weight {self.dynamic_matching_weight}')

    def compute_distribution_distance(self, pred_changes, demo_changes, mask):
        """
        Compute distance between distributions of changes using Wasserstein distance
        or MMD (Maximum Mean Discrepancy)
        """
        # Flatten and mask the changes
        pred_flat = pred_changes[mask].flatten()
        demo_flat = demo_changes[mask].flatten()
        
        if len(pred_flat) == 0 or len(demo_flat) == 0:
            return torch.tensor(0.0, device=pred_changes.device)
        
        # Simple approximation: compare statistical moments
        # Mean difference
        mean_diff = torch.abs(pred_flat.mean() - demo_flat.mean())
        
        # Variance difference  
        var_diff = torch.abs(pred_flat.var() - demo_flat.var())
        
        # Distribution distance (simplified Wasserstein-1)
        pred_sorted, _ = torch.sort(pred_flat)
        demo_sorted, _ = torch.sort(demo_flat)
        
        # Resample to same length for comparison
        min_len = min(len(pred_sorted), len(demo_sorted))
        if min_len > 1:
            pred_resampled = F.interpolate(pred_sorted.unsqueeze(0).unsqueeze(0), 
                                         size=min_len, mode='linear', align_corners=True).squeeze()
            demo_resampled = F.interpolate(demo_sorted.unsqueeze(0).unsqueeze(0), 
                                         size=min_len, mode='linear', align_corners=True).squeeze()
            wasserstein_dist = torch.abs(pred_resampled - demo_resampled).mean()
        else:
            wasserstein_dist = torch.tensor(0.0, device=pred_changes.device)
        
        return mean_diff + var_diff + wasserstein_dist

    def compute_dynamic_matching_loss(self, a_hat, actions, is_pad):
        """
        Compute dynamic-matching loss that compares distribution of changes
        between predicted and demonstration trajectories
        """
        if actions.shape[1] <= 1:
            return torch.tensor(0.0, device=actions.device)
        
        # Compute temporal differences (velocity/acceleration)
        demo_changes_1st = torch.diff(actions, dim=1)  # First-order differences
        pred_changes_1st = torch.diff(a_hat, dim=1)
        
        # Mask for valid temporal differences
        temporal_mask = ~is_pad[:, 1:]
        
        # First-order dynamic matching
        dynamic_loss_1st = self.compute_distribution_distance(
            pred_changes_1st, demo_changes_1st, temporal_mask
        )
        
        # Second-order differences (acceleration) if sequence is long enough
        if actions.shape[1] > 2:
            demo_changes_2nd = torch.diff(demo_changes_1st, dim=1)
            pred_changes_2nd = torch.diff(pred_changes_1st, dim=1)
            temporal_mask_2nd = ~is_pad[:, 2:]
            
            dynamic_loss_2nd = self.compute_distribution_distance(
                pred_changes_2nd, demo_changes_2nd, temporal_mask_2nd
            )
            
            return dynamic_loss_1st + 0.5 * dynamic_loss_2nd
        
        return dynamic_loss_1st

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
        
        # Dynamic matching loss
        dynamic_matching_loss = self.compute_dynamic_matching_loss(a_hat, actions, is_pad)
        
        return l1_loss, temporal_loss, dynamic_matching_loss

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
            
            l1, temporal_loss, dynamic_matching_loss = self.compute_loss_with_temporal_weighting(a_hat, actions, is_pad)
            loss_dict['l1'] = l1
            loss_dict['temporal'] = temporal_loss
            loss_dict['dynamic_matching'] = dynamic_matching_loss
            loss_dict['kl'] = total_kld[0]
            
            loss_dict['loss'] = (loss_dict['l1'] + 
                               self.temporal_weight * loss_dict['temporal'] + 
                               self.dynamic_matching_weight * loss_dict['dynamic_matching'] +
                               loss_dict['kl'] * self.kl_weight)
            
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
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