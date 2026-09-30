"""Frozen CLIP with a tensor-only image path retaining gradients to the source."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .models import freeze


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{tuple(tensor.shape)}:{tensor.dtype}'.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class CLIPReward(nn.Module):
    """Raw cosine similarity (no learned logit scale, softmax, or batch scaling).

    Decoder RGB [-1,1] -> clamp to [0,1] -> tensor bicubic resize/center crop
    -> CLIP normalization. Clamping is piecewise differentiable; no PIL,
    quantization, detach, stochastic augmentation, or no_grad on image features.
    """
    def __init__(self, model, tokenizer, image_size=224,
                 mean=(.48145466,.4578275,.40821073), std=(.26862954,.26130258,.27577711)):
        super().__init__()
        self.model = freeze(model)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.register_buffer('mean', torch.tensor(mean).reshape(1,3,1,1))
        self.register_buffer('std', torch.tensor(std).reshape(1,3,1,1))
        self.register_buffer('text_feature', torch.empty(0), persistent=False)
        self.prompt = None

    @torch.no_grad()
    def set_prompt(self, prompt):
        tokens = self.tokenizer([prompt], padding=True, truncation=False, return_tensors='pt')
        if tokens['input_ids'].shape[-1] > self.model.config.text_config.max_position_embeddings:
            raise ValueError('Reward prompt exceeds CLIP context length; shorten it (no silent truncation)')
        tokens = {k:v.to(self.mean.device) for k,v in tokens.items()}
        pooled = self.model.text_model(**tokens)[1]
        self.text_feature = F.normalize(self.model.text_projection(pooled), dim=-1)
        self.prompt = prompt

    def preprocess(self, images):
        x = ((images+1)/2).clamp(0,1)
        h,w = x.shape[-2:]
        size = self.image_size
        shape = (size, round(w*size/h)) if h <= w else (round(h*size/w),size)
        x = F.interpolate(x, size=shape, mode='bicubic', align_corners=False, antialias=True)
        top,left = (shape[0]-size)//2,(shape[1]-size)//2
        return (x[...,top:top+size,left:left+size]-self.mean)/self.std

    def features(self, images):
        pooled = self.model.vision_model(pixel_values=self.preprocess(images))[1]
        return F.normalize(self.model.visual_projection(pooled), dim=-1)

    def forward(self, images):
        if self.prompt is None:
            raise RuntimeError('Set the fixed reward prompt first')
        return (self.features(images)*self.text_feature).sum(-1)


def load_reward(config, device):
    try:
        from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
    except ImportError as exc:
        raise ImportError('Install requirements-guidance.txt in your experiment environment') from exc
    kwargs = dict(revision=config['revision'], local_files_only=config['local_files_only'])
    model = CLIPModel.from_pretrained(config['model'], torch_dtype=torch.float32,
                                     attn_implementation='eager', **kwargs)
    tokenizer = CLIPTokenizer.from_pretrained(config['model'], **kwargs)
    processor = CLIPImageProcessor.from_pretrained(config['model'], **kwargs)
    reward = CLIPReward(model, tokenizer, model.config.vision_config.image_size,
                        processor.image_mean, processor.image_std)
    metadata = {'weights_sha256': state_hash(model), 'resolved_revision': getattr(model.config,'_commit_hash',None),
                'image_size': reward.image_size, 'mean': processor.image_mean, 'std': processor.image_std,
                'preprocessing': '[-1,1] to [0,1], clamp, tensor bicubic antialias shortest-side resize, center crop, normalize',
                'score': 'cosine of L2-normalized projected image and text features; no logit scaling'}
    return reward.to(device), metadata


class RewardPotential:
    def __init__(self, reward, strength):
        self.reward, self.strength = reward, float(strength)

    def __call__(self, images):
        # Existing MH/HMC kernels minimize Phi: higher reward must LOWER Phi.
        return -self.strength*self.reward(images).double()
