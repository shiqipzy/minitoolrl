"""Current G0 objective: clipped GRPO, averaged over policy tokens per trajectory."""
import torch


def grpo_loss(logps, old_logps, advantages, mask, epsilon=0.2):
    if logps.shape != old_logps.shape or mask.shape != logps.shape:
        raise ValueError('token shapes differ')
    if advantages.shape != (logps.shape[0],):
        raise ValueError('row shapes differ')
    adv = advantages.detach().unsqueeze(1)
    # Mask before exp: prompts, observations and padding do not contribute.
    lp = torch.where(mask.bool(), logps, torch.zeros_like(logps))
    old = torch.where(mask.bool(), old_logps.detach(), torch.zeros_like(old_logps))
    ratio = (lp - old).exp()
    objective = torch.minimum(ratio * adv, ratio.clamp(1 - epsilon, 1 + epsilon) * adv)
    counts = mask.sum(1)
    if not bool((counts > 0).all()):
        raise ValueError('empty policy row')
    return -((objective * mask).sum(1) / counts).mean()
