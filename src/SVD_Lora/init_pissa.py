
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from typing import List, Optional


class PiSSALayer(nn.Module):
    """
    PiSSA层实现，通过SVD分解将预训练权重分解为残差矩阵和低秩矩阵
    """

    def __init__(self,
                 original_layer: nn.Module,
                 r: int = 64,
                 lora_alpha: float = 64,
                 lora_dropout: float = 0.05):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.original_layer = original_layer

        # 获取原始权重
        if hasattr(original_layer, 'weight'):
            self.weight_shape = original_layer.weight.shape
            self.perform_svd_decomposition(original_layer.weight)
        else:
            raise ValueError("Original layer must have weight attribute")

        # Dropout层
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def perform_svd_decomposition(self, weight: torch.Tensor):
        """
        执行SVD分解，将权重矩阵分解为残差矩阵和低秩矩阵
        只有低秩矩阵A和B是可训练的，残差矩阵保持冻结
        """
        # 确保权重在CPU上进行SVD分解
        weight_cpu = weight.detach().cpu().float()
        out_features, in_features = weight_cpu.shape

        # 执行SVD分解: W = U @ S @ V^T
        U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)  # U: [256, r], S: [r], Vh: [r, 1024]

        # 目标是还原为原始 W 的形状: [out_features, in_features]
        U_r = U[:, :self.r]  # [out_features, r]:[256,r]
        S_r = S[:self.r]  # [r]
        V_r = Vh[:self.r, :]  # [r, in_features]:[r,1024]

        # 构造低秩矩阵 W_lr = U_r @ diag(S_r) @ Vt_r
        W_lr = U_r @ torch.diag(S_r) @ V_r  # [256, r] x [r, r] x [r, 1024] = [256, 1024]
        if W_lr.shape != weight_cpu.shape:
            raise ValueError(f"SVD reconstruction shape mismatch: {W_lr.shape} vs {weight_cpu.shape}")

        # 计算残差矩阵 W_res = W - W_lr
        W_res = weight_cpu - W_lr

        # 将残差矩阵设置为不可训练的缓冲区（冻结状态）
        self.register_buffer('residual_weight', W_res.to(weight.device))

        # 设置可训练的低秩分解参数
        # A矩阵: [r, in_features]，对应Vt_r，可训练
        # B矩阵: [out_features, r]，对应U_r @ diag(sqrt(S_r))，可训练
        sqrt_S_r = torch.sqrt(S_r)

        self.lora_A = nn.Parameter(V_r.to(weight.device), requires_grad=True)  # 可训练
        self.lora_B = nn.Parameter((U_r @ torch.diag(sqrt_S_r)).to(weight.device), requires_grad=True)  # 可训练

        # 缩放因子
        self.scaling = self.lora_alpha / self.r

        # 确保残差矩阵不会被意外更新
        self.residual_weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：使用残差矩阵和低秩矩阵的组合
        """
        # 使用残差矩阵的计算
        residual_output = torch.nn.functional.linear(x, self.residual_weight,
                                                     self.original_layer.bias if hasattr(self.original_layer,
                                                                                         'bias') else None)

        # 使用低秩矩阵的计算: x @ A^T @ B^T
        # x: [B, *, in_features] → A.T: [in_features, r] → B.T: [r, out_features]
        lora_output = F.linear(
            F.linear(self.dropout(x), self.lora_A),  # [B, *, r]
            self.lora_B  # [r, out_features]
        ) * self.scaling

        # 加上 frozen residual
        residual_output = F.linear(x, self.residual_weight,
                                   bias=self.original_layer.bias if hasattr(self.original_layer, 'bias') else None)

        return residual_output + lora_output

    @property
    def weight(self):
        """模拟 nn.Linear 的 .weight 接口"""
        lora_part = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        return self.residual_weight + lora_part

    @property
    def bias(self):
        """模拟 nn.Linear 的 .bias 接口"""
        return self.original_layer.bias if hasattr(self.original_layer, "bias") else None


def apply_pissa_to_model(
    pissa_model: nn.Module,
    target_modules: List[str],
    r: int = 64,
    lora_alpha: float = 64,
    lora_dropout: float = 0.05
):
    """
    按路径替换模块为 PiSSA 层（防止重复替换）
    """
    modified_modules = []

    for full_name in target_modules:
        parts = full_name.split('.')
        parent = pissa_model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]
        target_layer = getattr(parent, child_name)

        if isinstance(target_layer, PiSSALayer):
            continue  # 已是 PiSSA 层，跳过

        if isinstance(target_layer, nn.Linear):
            new_layer = PiSSALayer(
                original_layer=target_layer,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            setattr(parent, child_name, new_layer)
            modified_modules.append(full_name)
            print(f"✅ Applied PiSSA to {full_name}")

    return pissa_model, modified_modules


def create_pissa_model(
    pissa_model: nn.Module,
    target_modules: List[str],
    r: int = 64,
    lora_alpha: float = 64,
    lora_dropout: float = 0.05
):
    """
    应用 PiSSA 到指定模块路径列表
    """
    if not target_modules:
        raise ValueError("You must provide `target_modules` with explicit full paths.")

    return apply_pissa_to_model(
        pissa_model=pissa_model,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )



def find_target_modules(model: nn.Module, module_types: List[type] = [nn.Linear]) -> List[str]:
    """
    自动查找模型中可以应用PiSSA的目标模块
    """
    target_modules = []

    for name, module in model.named_modules():
        if any(isinstance(module, module_type) for module_type in module_types):
            target_modules.append(name)

    return target_modules


def load_custom_model(model_path: str, model_class=None, **kwargs):
    """
    加载自定义模型的通用函数
    """
    if model_path.endswith('.pth') or model_path.endswith('.pt'):
        # 加载PyTorch模型文件
        if model_class is None:
            raise ValueError("model_class must be provided when loading from .pth/.pt files")

        model = model_class(**kwargs)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model
    else:
        # 尝试作为模型目录加载
        try:
            # 如果是transformers模型
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_path, local_files_only=True)
            return model
        except:
            raise ValueError(f"Cannot load model from {model_path}. Please provide model_class for custom models.")


def save_pissa_model(model: nn.Module, save_path: str):
    """
    保存PiSSA模型，只保存可训练的LoRA参数
    """
    pissa_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, PiSSALayer):
            pissa_state_dict[f"{name}.lora_A"] = module.lora_A
            pissa_state_dict[f"{name}.lora_B"] = module.lora_B

    torch.save(pissa_state_dict, save_path)
    print(f"PiSSA parameters saved to {save_path}")


def load_pissa_weights(model: nn.Module, pissa_weights_path: str):
    """
    加载PiSSA权重到模型中
    """
    pissa_state_dict = torch.load(pissa_weights_path, map_location='cpu')

    for name, module in model.named_modules():
        if isinstance(module, PiSSALayer):
            if f"{name}.lora_A" in pissa_state_dict:
                module.lora_A.data = pissa_state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in pissa_state_dict:
                module.lora_B.data = pissa_state_dict[f"{name}.lora_B"]

    print(f"PiSSA weights loaded from {pissa_weights_path}")


def freeze_non_pissa_parameters(model: nn.Module):
    """
    冻结模型中除了PiSSA参数之外的所有参数
    """
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            # PiSSA的可训练参数
            param.requires_grad = True
            trainable_count += 1
            print(f"Trainable: {name}")
        else:
            # 冻结其他所有参数
            param.requires_grad = False
            frozen_count += 1

    print(f"\nParameter freezing summary:")
    # print(f"  Frozen parameters: {frozen_count}")
    # print(f"  Trainable parameters: {trainable_count}")

    return model


def verify_parameter_status(model: nn.Module):
    """
    验证参数的可训练状态
    """
    print("\nParameter Status Verification:")
    print("-" * 50)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✓ TRAINABLE: {name} - Shape: {param.shape}")
        else:
            print(f"✗ FROZEN: {name} - Shape: {param.shape}")

    # 验证缓冲区（残差矩阵）
    print("\nBuffer Status (Residual Matrices):")
    print("-" * 50)
    for name, buffer in model.named_buffers():
        if 'residual_weight' in name:
            print(f"✓ BUFFER (Frozen): {name} - Shape: {buffer.shape}")


def count_trainable_parameters(model: nn.Module):
    """
    计算可训练参数数量，详细分析PiSSA参数
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    pissa_params = 0

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count

        if param.requires_grad:
            trainable_params += param_count
            if 'lora_A' in name or 'lora_B' in name:
                pissa_params += param_count
        else:
            frozen_params += param_count

    # 计算缓冲区参数（残差矩阵）
    buffer_params = 0
    for name, buffer in model.named_buffers():
        if 'residual_weight' in name:
            buffer_params += buffer.numel()

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': frozen_params,
        'pissa_parameters': pissa_params,
        'residual_parameters': buffer_params,
        'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0,
        'pissa_percentage': 100 * pissa_params / total_params if total_params > 0 else 0
    }