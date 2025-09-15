import torch
import torch.nn.functional as F
import operator

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """
    cache: dict[class_idx] = list of entries
      - 正缓存 entry 格式: [feat, loss]
      - 负缓存 entry 格式: [feat, avg_prob, prob_map]
    features_loss:
      - 正缓存时: [feat, loss]
      - 负缓存时: [feat, None, prob_map]
    """
    with torch.no_grad():
        if include_prob_map:
            feat, _, prob_map = features_loss
            # 用平均概率作为排序依据
            score = float(prob_map.mean().item())
            item = [feat, score, prob_map]
        else:
            feat, loss = features_loss
            score = float(loss)
            item = [feat, score]

        if pred in cache:
            cache[pred].append(item)
            # 正缓存按 loss 升序（loss 小的更可靠）
            # 负缓存按 avg_prob 降序（更“不确定”的负样本排前面）
            if include_prob_map:
                cache[pred] = sorted(cache[pred], key=lambda x: x[1], reverse=True)
            else:
                cache[pred] = sorted(cache[pred], key=lambda x: x[1])
            # 保留容量最大的那几条
            cache[pred] = cache[pred][:shot_capacity]
        else:
            cache[pred] = [item]

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        # 保证 cache_values 和 cache_keys 同 dtype/device
        device = cache_keys.device
        dtype = cache_keys.dtype

        if neg_mask_thresholds:
            mask_vals = torch.cat(cache_values, dim=0)
            mask = (mask_vals > neg_mask_thresholds[0]) & (mask_vals < neg_mask_thresholds[1])
            cache_values = mask.to(dtype=dtype, device=device)
        else:
            idxs = torch.tensor(cache_values, dtype=torch.int64, device=device)
            cache_values = F.one_hot(idxs, num_classes=clip_weights.size(0))
            cache_values = cache_values.to(dtype=dtype, device=device)

        affinity = image_features @ cache_keys
        alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
        beta_t = torch.tensor(beta, dtype=dtype, device=device)
        cache_logits = torch.exp(- (beta_t - beta_t * affinity)) @ cache_values
        return alpha_t * cache_logits