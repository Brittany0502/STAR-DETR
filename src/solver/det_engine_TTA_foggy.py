"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..self_training.self_training_utils import (get_pseudo_label_via_threshold, deal_pesudo_label,
                                                 rescale_pseudo_targets, convert_to_list_format)
from ..SVD_Lora.init_pissa import *

def train_one_epoch_eval(model: torch.nn.Module, criterion: torch.nn.Module,
                         data_loader: Iterable, data_loader_val: Iterable, optimizer: torch.optim.Optimizer,
                         device: torch.device, epoch: int, max_norm: float = 0,
                         postprocessor=None, coco_evaluator=None,clip_prompt_learner = None,clip_model =None,clip_transform = None,
                         clip_text = None,optimizer_prompt_learner = None,logit_scale = None,
                         background_text_features = None, classnames = None, reward_model = None,
                         **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 1)
    writer: SummaryWriter = kwargs.get('writer', None)

    ema: ModelEMA = kwargs.get('ema', None)
    scaler: GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)

    teacher_model = ema.module
    teacher_model.eval()
    alpha_ema = 0.999

    # ----warm up ----------
    use_pissa = epoch >= 0

    # ----SVD_Lora optimizer--------
    pissa_optimizer = None
    pissa_params = []

    if use_pissa:
        print(f"Epoch {epoch}: Enabling PISSA for decoder layers")

        # -------teacher SVD_Lora -----------
        teacher_model, _ = create_pissa_model(
            pissa_model=teacher_model,
            target_modules=[
                "decoder.decoder.layers.4.self_attn.out_proj",
                "decoder.decoder.layers.5.self_attn.out_proj",
            ],
            r=8,
            lora_alpha=8,
            lora_dropout=0.05
        )
        teacher_model = freeze_non_pissa_parameters(teacher_model)

        # -------student SVD_Lora---------
        model, modified_modules = create_pissa_model(
            pissa_model=model,
            target_modules=[
                "decoder.decoder.layers.4.self_attn.out_proj",
                "decoder.decoder.layers.5.self_attn.out_proj",
            ],
            r=8,
            lora_alpha=8,
            lora_dropout=0.05
        )
        model = freeze_non_pissa_parameters(model)

        # ---------SVD_Lora optimizer--------
        for name, param in model.named_parameters():
            if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
                pissa_params.append(param)

        if pissa_params:
            pissa_optimizer = torch.optim.AdamW(pissa_params, lr=5e-5, weight_decay=1e-4)

        param_stats = count_trainable_parameters(model)
        print(f"\nParameter Statistics (with PISSA):")
        print(f"Total parameters: {param_stats['total_parameters']:,}")
        print(f"Trainable parameters: {param_stats['trainable_parameters']:,}")
        print(f"Trainable percentage: {param_stats['trainable_percentage']:.2f}%")

    else:
        print(f"Epoch {epoch}: Using standard training without PISSA")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100

        print(f"\nParameter Statistics (without PISSA):")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {trainable_percentage:.2f}%")


    #-------clip------------
    image_encoder = clip_model.visual

    num_classes_prompt = len(classnames) if classnames else 9  # prompt number
    num_classes_dataset = len(classnames) - 1 if classnames else 8
    background_class_idx = num_classes_prompt - 1

    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    criterion.eval()


    for i, ((samples, targets, samples_val, targets_val, img_original), (_, _, _, _, _)) in enumerate(
            zip(metric_logger.log_every(data_loader, print_freq, header), data_loader_val)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # ----test samples
        samples_val = samples_val.to(device)
        targets_val = [{k: v.to(device) for k, v in t.items()} for t in targets_val]

        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        with torch.no_grad():
            teacher_predict_results = teacher_model(samples_val)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets_val], dim=0)
        teacher_predict_results = postprocessor(teacher_predict_results, orig_target_sizes)

        threshold = np.asarray([0.3] * num_classes_dataset)
        idx_list, labels_dict, boxes_dict, scores_dict = get_pseudo_label_via_threshold(teacher_predict_results,
                                                                                        threshold=threshold)
        target_pseudo_labels = deal_pesudo_label(targets_val, idx_list, labels_dict, boxes_dict, scores_dict)
        target_pseudo_labels = rescale_pseudo_targets(samples_val, target_pseudo_labels)
        target_pseudo_labels_list = convert_to_list_format(target_pseudo_labels)

        filter_pseudo_list = []
        clip_total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        # =========================================================
        for idx, pseudo_label in enumerate(target_pseudo_labels_list):

            orig_size = pseudo_label['orig_size']
            image_id = pseudo_label['image_id'].item()
            img = img_original[0]

            scores = pseudo_label['scores']  # Tensor [N]
            boxes = pseudo_label['boxes']  # Tensor [N,4]
            labels = pseudo_label['labels']  # Tensor [N]

            preprocessed = []
            valid_indices = []
            margin = 0.2
            for box_id, box in enumerate(boxes):
                assert (box >= 0).all() and (box[2:] <= orig_size).all(), "Box coordinates are out of bounds"
                # print(f"Box coordinates: {box}, Image size: {orig_size}")
                x, y, w, h = (int(box[0] * orig_size[1]),
                              int(box[1] * orig_size[0]),
                              int(box[2] * orig_size[1]),
                              int(box[3] * orig_size[0]))
                # margin
                x_min = max(0, x - w // 2 - int(w * margin))
                y_min = max(0, y - h // 2 - int(h * margin))
                x_max = min(orig_size[1], x + w // 2 + int(w * margin))
                y_max = min(orig_size[0], y + h // 2 + int(h * margin))
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                if x_max <= x_min or y_max <= y_min:
                    print(f"Invalid crop region for box {box_id}, skipping...")
                    continue

                patch = img.crop((x_min, y_min, x_max, y_max))
                preprocessed.append(clip_transform(patch))
                valid_indices.append(box_id)

            if not preprocessed:
                continue

            valid_scores = scores[valid_indices]
            valid_boxes = boxes[valid_indices]
            valid_labels = labels[valid_indices]

            preprocessed = torch.stack(preprocessed).to(device)
            with torch.no_grad():
                img_feats = image_encoder(preprocessed)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            prompts = clip_prompt_learner()
            tokenized = clip_prompt_learner.tokenized_prompts
            t_feats = clip_text(prompts, tokenized)
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)

            # background feature
            if background_text_features is not None:
                background_feat_normalized = background_text_features / background_text_features.norm(dim=-1,
                                                                                                      keepdim=True)
                t_feats[background_class_idx:background_class_idx + 1] = (t_feats[
                                                                          background_class_idx:background_class_idx + 1] + background_feat_normalized) / 2.0  # 平均融合

            logits = img_feats @ t_feats.t()  # [N, num_classes]
            pred_clip = logits.argmax(dim=1)  # [N]

            # -----------dual threshold------------
            high_mask = valid_scores.to(device) > 0.5
            mid_mask = (valid_scores.to(device) >= 0.3) & (valid_scores.to(device) <= 0.5)  # 0.3-0.5 Consistency regularization

            label_0_mask = valid_labels == 0


            background_pred_mask = pred_clip == background_class_idx
            background_filter_threshold = 0.3
            high_conf_background_mask = background_pred_mask & (valid_scores.to(device) > background_filter_threshold)

            high_conf_background_mask = high_conf_background_mask & (~label_0_mask)
            high_mask = high_mask & ((~high_conf_background_mask) | label_0_mask)
            mid_mask = mid_mask & ((~high_conf_background_mask) | label_0_mask)

            if mid_mask.any():
                mid_keep = torch.zeros_like(mid_mask, dtype=torch.bool)

                for i in torch.where(mid_mask)[0]:
                    dataset_label = valid_labels[i].item()
                    clip_pred = pred_clip[i].item()

                    if dataset_label == 0:
                        mid_keep[i] = True

                    elif clip_pred < num_classes_dataset and clip_pred == dataset_label:
                        mid_keep[i] = True

                combined_mask = high_mask | mid_keep
            else:
                combined_mask = high_mask

            if combined_mask.any():
                final_boxes = valid_boxes[combined_mask]
                final_labels = valid_labels[combined_mask]
                final_scores = valid_scores[combined_mask]

                filter_pseudo_list.append({
                    'boxes': final_boxes,
                    'labels': final_labels,
                    'scores': final_scores,
                    'image_id': pseudo_label['image_id'],
                    'area': pseudo_label['area'],
                    'iscrowd': pseudo_label['iscrowd'],
                    'orig_size': pseudo_label['orig_size'],
                    'idx': pseudo_label['idx'],
                })

                # ------------reward model--------------

                scale = logit_scale.exp()
                feats_all = img_feats[combined_mask]
                labels_all = valid_labels.to(device)[combined_mask]
                t_feats_dataset = t_feats[:num_classes_dataset]

                if logit_scale is not None:
                    scale = logit_scale.exp()
                    logits_all = scale * (feats_all @ t_feats_dataset.t())  # [N_all, 8]
                else:
                    logits_all = feats_all @ t_feats_dataset.t()  # [N_all, 8]

                if reward_model is not None:
                    with torch.no_grad():
                        # 1. reward model gets img features
                        all_patches = preprocessed[combined_mask]
                        clip_image_features = reward_model.extract_image_features(all_patches)

                        # 2. text feauture
                        all_text = [classnames[label.item()] for label in labels_all]
                        clip_text_features = reward_model.extract_text_features(captions=all_text)

                        # 3. compute CLIPScore
                        reward_scale = reward_model.clip_model.logit_scale.exp()
                        sim = torch.sum(clip_image_features * clip_text_features, dim=-1)
                        clip_rewards = torch.clamp(reward_scale * sim, min=0.0)

                        # 4. postprocess
                        rewards = reward_model.rewards_post_process(clip_rewards)
                        rewards = torch.clamp(rewards, min=0.5, max=1.0)

                    per_loss = F.cross_entropy(logits_all, labels_all, reduction='none')
                    weighted_loss = torch.mean(rewards * per_loss)
                    clip_total_loss = clip_total_loss + weighted_loss
                else:
                    clip_loss = F.cross_entropy(logits_all, labels_all)
                    clip_total_loss = clip_total_loss + clip_loss


        if len(idx_list) >= 1:
            #  #==============================TTA==============================================================
            optimizer.zero_grad()
            optimizer_prompt_learner.zero_grad()
            outputs = model(samples)
            loss_dict = criterion(outputs, filter_pseudo_list, **metas)
            loss = sum(loss_dict.values())
            total_loss = loss + clip_total_loss * 0.2
            total_loss.backward()
            # print(clip_prompt_learner.ctx)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(clip_prompt_learner.parameters(), max_norm)

            optimizer.step()
            optimizer_prompt_learner.step()

        if use_pissa and pissa_optimizer is not None:
            pissa_optimizer.step()


        # ----------------inference---------------
        teacher_model.eval()
        criterion.eval()
        outputs = teacher_model(samples_val)

        if len(idx_list) >= 1:
            loss_dict = criterion(outputs, target_pseudo_labels_list, **metas)
        else:
            loss_dict = criterion(outputs, targets_val, **metas)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets_val], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets_val, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        model.train()
        # ==============each iteration，update teacher model==========================
        with torch.no_grad():
            student_model_state_dict = model.state_dict()
            teacher_model_state_dict = teacher_model.state_dict()
            for entry in teacher_model_state_dict.keys():
                teacher_param = teacher_model_state_dict[entry].clone().detach()
                student_param = student_model_state_dict[entry].clone().detach()
                new_param = (teacher_param * alpha_ema) + (student_param * (1. - alpha_ema))
                teacher_model_state_dict[entry] = new_param
            teacher_model.load_state_dict(teacher_model_state_dict)

        # ==========================================================

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if use_pissa and pissa_optimizer is not None:
            metric_logger.add_meter('pissa_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.update(pissa_lr=pissa_optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # -------------eval------------------
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    if dist_utils.is_main_process():
        with open("./output/cityscape_cityscapefoggy_TTA.txt", "a") as f:
            f.write(f"Epoch: {epoch},")

            if coco_evaluator is not None:
                for iou_type in iou_types:
                    if iou_type == 'bbox':
                        coco_eval = coco_evaluator.coco_eval[iou_type]
                        f.write(
                            f"{coco_eval.stats[1]:.3f}\n")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader,
             coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    for samples, targets, samples1, targets1, _ in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator

