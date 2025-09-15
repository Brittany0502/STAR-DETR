"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import json
import datetime

import torch

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine_TTA_foggy import train_one_epoch_eval, evaluate
from src.clip.custom_clip import PromptLearner, TextEncoder

from copy import deepcopy
from src.clip import load, tokenize

# -------reward----------
from ..clip_reward import CLIPRewards


class DetSolver(BaseSolver):

    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        arch = "ViT-L/14"
        DOWNLOAD_ROOT_CLIP = '/data/caixinyi/TTA/创新点/code/RLCF/pretrained/clip'
        clip, embed_dim, _transform = load(arch, device='cuda', download_root=DOWNLOAD_ROOT_CLIP)

        classnames = ["person", "car", "train", "rider", "truck", "motorcycle", "bicycle", "bus", "background"]

        # 为background类别定义prompt
        background_prompts = [
            "a window",
            "a building window",
            "a glass window",
            "architectural elements",
            "building facade",
            "wall structure",
            "urban background",
            "static objects",
            "non-vehicle non-person objects"
        ]

        batch_size = None
        n_ctx = 16
        ctx_init = "a foggy photo of a"
        ctx_position = 'end'
        learned_cls = False
        prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)

        textencoder_clip = TextEncoder(clip)
        logit_scale = clip.logit_scale.data

        # define optimizer
        trainable_param_prompt_learner = prompt_learner.parameters()
        optimizer_prompt_learner = torch.optim.AdamW(trainable_param_prompt_learner, 7e-3, weight_decay=5e-4)
        # optim_state_prompt_learner = deepcopy(optimizer_prompt_learner.state_dict())

        # reward model
        reward_model = CLIPRewards(device='cuda', arch="ViT-L/14", classification=True,
                                   amplify_rewards=False, sample_k=5, reward_process=True, process_batch=False,
                                   default_resolutions=224)

        # background average feature
        with torch.no_grad():
            background_text_features_list = []
            for prompt in background_prompts:
                text_tokens = tokenize([prompt]).cuda()
                text_features = clip.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                background_text_features_list.append(text_features.squeeze(0))
            background_text_features = torch.stack(background_text_features_list).mean(dim=0)

        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch_eval(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.val_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                postprocessor=self.postprocessor,
                coco_evaluator=self.evaluator,
                clip_prompt_learner=prompt_learner,
                clip_model=clip,
                clip_transform=_transform,
                clip_text=textencoder_clip,
                logit_scale=logit_scale,
                optimizer_prompt_learner=optimizer_prompt_learner,
                classnames = classnames,
                background_text_features=background_text_features,
                reward_model = reward_model
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # module = self.ema.module if self.ema else self.model
            module = self.model

            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            module1 = self.ema.module if self.ema else self.model

            test_stats, coco_evaluator = evaluate(
                module1,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                                              self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
