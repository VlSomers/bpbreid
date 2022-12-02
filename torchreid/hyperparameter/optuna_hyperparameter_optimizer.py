# import os
# import sys
# import time
# import os.path as osp
# import argparse
#
# import cv2
# import optuna
# import torch
# import torch.nn as nn
# from optuna import Trial
# from optuna.samplers import GridSampler
#
# import torchreid
# from torchreid.utils import (
#     Logger, check_isfile, set_random_seed, collect_env_info,
#     resume_from_checkpoint, load_pretrained_weights, compute_model_complexity, Writer
# )
#
# from scripts.default_config import (
#     imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
#     get_default_config, lr_scheduler_kwargs
# )
#
#
# def build_datamanager(cfg):
#     if cfg.data.type == 'image':
#         return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
#     else:
#         return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))
#
#
# def build_engine(cfg, datamanager, model, optimizer, scheduler, writer):
#     if cfg.data.type == 'image':
#         if cfg.loss.name == 'softmax':
#             engine = torchreid.engine.ImageSoftmaxEngine(
#                 datamanager,
#                 model,
#                 optimizer=optimizer,
#                 scheduler=scheduler,
#                 use_gpu=cfg.use_gpu,
#                 label_smooth=cfg.loss.softmax.label_smooth,
#                 save_model_flag=cfg.model.save_model_flag,
#                 writer=writer
#             )
#
#         elif cfg.loss.name == 'triplet':
#             engine = torchreid.engine.ImageTripletEngine(
#                 datamanager,
#                 model,
#                 optimizer=optimizer,
#                 margin=cfg.loss.triplet.margin,
#                 weight_t=cfg.loss.triplet.weight_t,
#                 weight_x=cfg.loss.triplet.weight_x,
#                 scheduler=scheduler,
#                 use_gpu=cfg.use_gpu,
#                 label_smooth=cfg.loss.softmax.label_smooth,
#                 save_model_flag=cfg.model.save_model_flag,
#                 writer=writer
#             )
#
#         elif cfg.loss.name == 'part_based':
#             engine = torchreid.engine.ImagePartBasedEngine(
#                 datamanager,
#                 model,
#                 optimizer=optimizer,
#                 loss_name=cfg.loss.part_based.name,
#                 config=cfg,
#                 margin=cfg.loss.triplet.margin,
#                 scheduler=scheduler,
#                 use_gpu=cfg.use_gpu,
#                 save_model_flag=cfg.model.save_model_flag,
#                 writer=writer,
#                 mask_filtering_training=cfg.model.bpbreid.mask_filtering_training,
#                 mask_filtering_testing=cfg.model.bpbreid.mask_filtering_testing,
#                 mask_filtering_threshold=cfg.model.bpbreid.mask_filtering_threshold,
#                 batch_debug_freq=cfg.train.batch_debug_freq,
#                 batch_size_pairwise_dist_matrix=cfg.test.batch_size_pairwise_dist_matrix
#             )
#
#     else:
#         if cfg.loss.name == 'softmax':
#             engine = torchreid.engine.VideoSoftmaxEngine(
#                 datamanager,
#                 model,
#                 optimizer=optimizer,
#                 scheduler=scheduler,
#                 use_gpu=cfg.use_gpu,
#                 label_smooth=cfg.loss.softmax.label_smooth,
#                 pooling_method=cfg.video.pooling_method,
#                 save_model_flag=cfg.model.save_model_flag,
#                 writer=writer
#             )
#
#         else:
#             engine = torchreid.engine.VideoTripletEngine(
#                 datamanager,
#                 model,
#                 optimizer=optimizer,
#                 margin=cfg.loss.triplet.margin,
#                 weight_t=cfg.loss.triplet.weight_t,
#                 weight_x=cfg.loss.triplet.weight_x,
#                 scheduler=scheduler,
#                 use_gpu=cfg.use_gpu,
#                 label_smooth=cfg.loss.softmax.label_smooth,
#                 save_model_flag=cfg.model.save_model_flag,
#                 writer=writer
#             )
#
#     return engine
#
#
# def reset_config(cfg, args):
#     if args.root:
#         cfg.data.root = args.root
#     if args.sources:
#         cfg.data.sources = args.sources
#     if args.targets:
#         cfg.data.targets = args.targets
#     if args.transforms:
#         cfg.data.transforms = args.transforms
#
#
# def merge_optuna_hyperparams(cfg, trial):
#     # cfg.data.sources
#     # cfg.data.targets
#     # cfg.sampler.num_instances
#     # cfg.train.optim
#     # cfg.train.lr
#     # cfg.train.weight_decay
#     # cfg.train.max_epoch
#     # cfg.train.start_epoch
#     # cfg.train.batch_size
#     # cfg.train.fixbase_epoch
#     # cfg.train.open_layers
#     # cfg.train.staged_lr
#     # cfg.train.new_layers
#     # cfg.train.base_lr_mult
#     # cfg.train.lr_scheduler
#     # cfg.train.stepsize
#     # cfg.train.gamma
#     # cfg.train.seed
#     # cfg.train.eval_freq
#     # cfg.train.batch_debug_freq
#     # cfg.sgd.momentum
#     # cfg.loss.triplet.margin
#     # cfg.sgd.dampening
#     # cfg.sgd.nesterov
#     # cfg.rmsprop.alpha
#     # cfg.adam.beta1
#     # cfg.adam.beta2
#
#     # Categorical parameter
#     # optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])
#
#     # Int parameter
#     # num_layers = trial.suggest_int('num_layers', 1, 3)
#
#     # Uniform parameter
#     # dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)
#
#     # Loguniform parameter
#     # learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
#
#     # Discrete-uniform parameter
#     # drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)
#
#     if len(cfg.tuning.model.bpbreid.pooling) != 0:
#         cfg.model.bpbreid.pooling = trial.suggest_categorical("model.bpbreid.pooling", cfg.tuning.model.bpbreid.pooling)
#     if len(cfg.tuning.model.bpbreid.normalization) != 0:
#         cfg.model.bpbreid.normalization = trial.suggest_categorical("model.bpbreid.normalization", cfg.tuning.model.bpbreid.normalization)
#     if len(cfg.tuning.model.bpbreid.mask_filtering_training) != 0:
#         cfg.model.bpbreid.mask_filtering_training = trial.suggest_categorical("model.bpbreid.mask_filtering_training", cfg.tuning.model.bpbreid.mask_filtering_training)
#     if len(cfg.tuning.model.bpbreid.mask_filtering_testing) != 0:
#         cfg.model.bpbreid.mask_filtering_testing = trial.suggest_categorical("model.bpbreid.mask_filtering_testing", cfg.tuning.model.bpbreid.mask_filtering_testing)
#     if len(cfg.tuning.model.bpbreid.mask_filtering_threshold) != 0:
#         cfg.model.bpbreid.mask_filtering_threshold = trial.suggest_categorical("model.bpbreid.mask_filtering_threshold", cfg.tuning.model.bpbreid.mask_filtering_threshold)
#     if len(cfg.tuning.loss.part_based.name) != 0:
#         cfg.loss.part_based.name = trial.suggest_categorical("loss.part_based.name", cfg.tuning.loss.part_based.name)
#
#
# def main():
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument(
#         '--config-file', type=str, default='', help='path to config file'
#     )
#     parser.add_argument(
#         '-s',
#         '--sources',
#         type=str,
#         nargs='+',
#         help='source datasets (delimited by space)'
#     )
#     parser.add_argument(
#         '-t',
#         '--targets',
#         type=str,
#         nargs='+',
#         help='target datasets (delimited by space)'
#     )
#     parser.add_argument(
#         '--transforms', type=str, nargs='+', help='data augmentation'
#     )
#     parser.add_argument(
#         '--root', type=str, default='', help='path to data root'
#     )
#     parser.add_argument(
#         'opts',
#         default=None,
#         nargs=argparse.REMAINDER,
#         help='Modify config options using the command-line'
#     )
#     args = parser.parse_args()
#
#     cfg = get_default_config()
#     cfg.use_gpu = torch.cuda.is_available()
#     if args.config_file:
#         cfg.merge_from_file(args.config_file)
#         cfg.project.config_file = os.path.basename(args.config_file)
#     reset_config(cfg, args)
#     cfg.merge_from_list(args.opts)
#
#     # Create objective function with access to config from main context
#     def objective(trial: Trial):
#         # import trial hyper parameters into corresponding config field
#         merge_optuna_hyperparams(cfg, trial)
#
#         if cfg.project.debug_mode:
#             torch.autograd.set_detect_anomaly(True)
#         writer = Writer(cfg)
#         set_random_seed(cfg.train.seed)
#         log_name = 'test_log' if cfg.test.evaluate else 'train_log'
#         log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
#         log_name += '.txt'
#         sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))
#         print('Show configuration\n{}\n'.format(cfg))
#         print('Collecting env info ...')
#         print('** System info **\n{}\n'.format(collect_env_info()))
#         if cfg.use_gpu:
#             torch.backends.cudnn.benchmark = True
#         datamanager = build_datamanager(cfg)
#         print('Building model: {}'.format(cfg.model.name))
#         model = torchreid.models.build_model(
#             name=cfg.model.name,
#             num_classes=datamanager.num_train_pids,
#             loss=cfg.loss.name,
#             pretrained=cfg.model.pretrained,
#             use_gpu=cfg.use_gpu,
#             pooling=cfg.model.bpbreid.pooling,
#             normalization=cfg.model.bpbreid.normalization
#         )
#         num_params, flops = compute_model_complexity(
#             model, (1, 3, cfg.data.height, cfg.data.width)
#         )
#         print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))
#         if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
#             load_pretrained_weights(model, cfg.model.load_weights)
#         if cfg.use_gpu:
#             model = nn.DataParallel(model).cuda()
#         optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
#         scheduler = torchreid.optim.build_lr_scheduler(
#             optimizer, **lr_scheduler_kwargs(cfg)
#         )
#         if cfg.model.resume and check_isfile(cfg.model.resume):
#             cfg.train.start_epoch = resume_from_checkpoint(
#                 cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
#             )
#         print(
#             'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
#         )
#         engine = build_engine(cfg, datamanager, model, optimizer, scheduler, writer)
#         mAP = engine.run(**engine_run_kwargs(cfg))
#         return mAP
#
#     # gridSampler = GridSampler()
#     # study = optuna.create_study(direction="maximize", sampler=gridSampler)
#     # TODO ETA
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=4, timeout=600)
#
#     pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
#     complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
#
#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))
#     print("  Number of pruned trials: ", len(pruned_trials))
#     print("  Number of complete trials: ", len(complete_trials))
#
#     print("Best trial:")
#     trial = study.best_trial
#
#     print("  Value: ", trial.value)
#
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))
#
#
# if __name__ == '__main__':
#     main()
