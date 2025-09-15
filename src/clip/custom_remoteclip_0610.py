# coding=utf-8
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.clip import load, tokenize
import open_clip

import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录
parent_dir = os.path.dirname(current_dir)

# 将上一级目录添加到sys.path
sys.path.append(parent_dir)

### Remote CLIP model checkpoint paths
REMOTE_CLIP_PATHS = {
    'ViT-L-14': '/data/caixinyi/TTA/rtdetrv2_pytorch_TTA-自训练-base-prompt/pretrained/RemoteCLIP-ViT-L-14.pt',
    'ViT-B-32': '/data/caixinyi/TTA/rtdetrv2_pytorch_TTA-自训练-base-prompt/pretrained/RemoteCLIP-ViT-B-32.pt',
    'RN50': '/data/caixinyi/TTA/rtdetrv2_pytorch_TTA-自训练-base-prompt/pretrained/RemoteCLIP-RN50.pt'
}


def load_remote_clip(model_name, device):
    """Load remote sensing CLIP model"""
    clip_model, _, transform = open_clip.create_model_and_transforms(model_name)

    # Load pretrained weights
    if model_name in REMOTE_CLIP_PATHS and os.path.exists(REMOTE_CLIP_PATHS[model_name]):
        ckpt = torch.load(REMOTE_CLIP_PATHS[model_name], map_location="cpu")
        message = clip_model.load_state_dict(ckpt)
        print(f"Loaded RemoteCLIP {model_name}: {message}")
    else:
        print(f"Warning: RemoteCLIP checkpoint not found for {model_name}, using default weights")

    clip_model = clip_model.to(device).eval()

    # Get embedding dimension
    if hasattr(clip_model, 'text_projection'):
        embed_dim = clip_model.text_projection.shape[0]
    else:
        embed_dim = clip_model.transformer.width

    return clip_model, embed_dim, transform


def tokenize_remote(texts, context_length=77, return_lengths=False):
    if isinstance(texts, str):
        texts = [texts]

    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    tokenized = tokenizer(texts, context_length=context_length)

    if return_lengths:
        lens = []
        for tokens in tokenized:
            tokens_list = tokens.tolist()
            try:
                length = tokens_list.index(0)
            except ValueError:
                length = context_length
            lens.append(length)
        return tokenized, lens

    return tokenized


class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L-14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        # Map arch names to open_clip format
        arch_mapping = {
            "ViT-L/14": "ViT-L-14",
            "ViT-B/32": "ViT-B-32",
            "ViT-B/16": "ViT-B-16",
            "RN50": "RN50"
        }
        clip_arch = arch_mapping.get(arch, arch)

        clip, embed_dim, _ = load_remote_clip(clip_arch, device)
        self.encoder = clip.visual
        # Clean up transformer to save memory
        if hasattr(clip, 'transformer'):
            del clip.transformer
        torch.cuda.empty_cache()

        self.cls_head = nn.Linear(embed_dim, n_class)

    @property
    def dtype(self):
        if hasattr(self.encoder, 'conv1'):
            return self.encoder.conv1.weight.dtype
        else:
            # For ViT models
            return next(self.encoder.parameters()).dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = next(clip_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.to(prompts.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(prompts.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end',
                 learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = next(clip_model.parameters()).dtype
        self.dtype = dtype

        # Get device from clip_model
        if hasattr(clip_model.visual, 'conv1'):
            self.device = clip_model.visual.conv1.weight.device
        else:
            self.device = next(clip_model.parameters()).device

        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # ----- 初始化上下文向量 -----
        if ctx_init:
            print("Initializing the context with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize_remote(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        if self.batch_size is not None:
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  # (N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # learnable context vectors

        # ----- 构造 prompts 并使用统一 tokenizer 获取 tokenized prompts 和 name_lens -----
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype)
            nn.init.normal_(cls_vectors, std=0.02)
            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)
            prompts = [prompt_prefix + " X." for _ in classnames]  # X is learnable cls token

        # 使用统一的 open_clip tokenizer
        tokenized_prompts, name_lens = tokenize_remote(prompts, return_lengths=True)
        tokenized_prompts = tokenized_prompts.to(self.device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # ----- 保存组成 prompt 的各部分 -----
        self.register_buffer("token_prefix", embedding[:, :1, :])  # <sos>
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., <eos>
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # cls + eos

        # ----- 保存其它属性 -----
        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        """reset the prompt to be the initial state"""
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)

        # 重新构造 prompts
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_token = "X"
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(cls_vectors, std=0.02)
            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)

        # 使用 open_clip tokenizer 重新编码
        tokenized_prompts, name_lens = tokenize_remote(prompts, return_lengths=True)
        tokenized_prompts = tokenized_prompts.to(self.device)

        # 重新计算 embedding
        clip_arch = {
            "ViT-L/14": "ViT-L-14",
            "ViT-B/32": "ViT-B-32",
            "ViT-B/16": "ViT-B-16",
            "RN50": "RN50"
        }.get(arch, arch)

        clip, _, _ = load_remote_clip(clip_arch, self.device)
        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        # 更新 token prefix/suffix
        self.token_prefix = embedding[:, :1, :]
        if self.learned_cls:
            self.token_suffix = embedding[:, 1 + self.n_ctx + 1:, :]
        else:
            self.token_suffix = embedding[:, 1 + self.n_ctx:, :]

        # 更新其他元信息
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).repeat(self.n_cls, 1, 1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(0).repeat(self.n_cls, 1, 1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None:
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        cls,  # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx  # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L-14",
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipTestTimeTuning, self).__init__()
        # Map arch names to open_clip format
        arch_mapping = {
            "ViT-L/14": "ViT-L-14",
            "ViT-B/32": "ViT-B-32",
            "ViT-B/16": "ViT-B-16",
            "RN50": "RN50"
        }
        clip_arch = arch_mapping.get(arch, arch)

        clip, _, _ = load_remote_clip(clip_arch, device)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    arch_mapping = {
        "ViT-L/14": "ViT-L-14",
        "ViT-B/32": "ViT-B-32",
        "ViT-B/16": "ViT-B-16",
        "RN50": "RN50"
    }
    mapped_arch = arch_mapping.get(clip_arch, clip_arch)
    classnames = ['plane','storage tank','ship']

    model = ClipTestTimeTuning(device,classnames , None, arch=mapped_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model


