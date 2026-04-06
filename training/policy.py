import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

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
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.0, 0.0],
                                         std=[0.229, 0.224, 0.225, 1.0, 1.0])

        def add_coord_channels(img_tensor):
            b, c, h, w = img_tensor.shape
            device = img_tensor.device
            xs = torch.linspace(0.0, 1.0, w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
            ys = torch.linspace(0.0, 1.0, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
            return torch.cat([img_tensor, xs, ys], dim=1)

        if image.dim() == 4:
            image = add_coord_channels(image)
            image = normalize(image)
        elif image.dim() == 5:
            b, n, c, h, w = image.shape
            image = image.view(b * n, c, h, w)
            image = add_coord_channels(image)
            image = normalize(image)
            image = image.view(b, n, c + 2, h, w)
        else:
            raise ValueError(f"Unexpected image tensor shape: {image.shape}")
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar) # KL divergence loss, # Compute how "normal" the latent space is 
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            # Balance both objectives
            # total_loss = l1_loss + kl_weight * kl_loss
            #                ↑        ↑
            #             Learn     Keep latent
            #             actions   space organized
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
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.0, 0.0],
                                         std=[0.229, 0.224, 0.225, 1.0, 1.0])

        def add_coord_channels(img_tensor):
            b, c, h, w = img_tensor.shape
            device = img_tensor.device
            xs = torch.linspace(0.0, 1.0, w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
            ys = torch.linspace(0.0, 1.0, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
            return torch.cat([img_tensor, xs, ys], dim=1)

        if image.dim() == 4:
            image = add_coord_channels(image)
            image = normalize(image)
        elif image.dim() == 5:
            b, n, c, h, w = image.shape
            image = image.view(b * n, c, h, w)
            image = add_coord_channels(image)
            image = normalize(image)
            image = image.view(b, n, c + 2, h, w)
        else:
            raise ValueError(f"Unexpected image tensor shape: {image.shape}")
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
    # This formula computes: KL(learned_distribution || standard_normal)
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Returns how "far" the learned distribution is from standard normal
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld