import random
import uuid
from datetime import datetime
from yacs.config import CfgNode as CN
from torchreid.utils.constants import *
from deepdiff import DeepDiff
import re
import pprint


def get_default_config():
    cfg = CN()

    # project
    cfg.project = CN()
    cfg.project.name = "BPBreID"  # will be used as WanDB project name
    cfg.project.experiment_name = ""
    cfg.project.diff_config = ""
    cfg.project.notes = ""
    cfg.project.tags = []
    cfg.project.config_file = ""
    cfg.project.debug_mode = False
    cfg.project.logger = CN()  # Choose experiment manager client to use or simply use disk dump / matplotlib
    cfg.project.logger.use_clearml = False
    cfg.project.logger.use_neptune = False
    cfg.project.logger.use_tensorboard = False
    cfg.project.logger.use_wandb = False
    cfg.project.logger.matplotlib_show = False
    cfg.project.logger.save_disk = True  # save images to disk
    cfg.project.job_id = random.randint(0, 1_000_000_000)
    cfg.project.experiment_id = str(uuid.uuid4())
    cfg.project.start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%MS")

    # model
    cfg.model = CN()
    cfg.model.name = 'bpbreid'
    cfg.model.pretrained = True  # automatically load pretrained model weights if available (For example HRNet
    # pretrained weights on ImageNet)
    cfg.model.load_weights = ''  # path to model weights, for doing inference with a model that was saved on disk with 'save_model_flag'
    cfg.model.load_config = False  # load config saved with model weights and overwrite current config
    cfg.model.resume = ''  # path to checkpoint for resume training
    cfg.model.save_model_flag = False  # path to checkpoint for resume training
    # configs for our part-based model BPBreID
    cfg.model.bpbreid = CN()
    cfg.model.bpbreid.pooling = 'gwap'  # ['gap', 'gmp', 'gwap', 'gwap2']
    cfg.model.bpbreid.normalization = 'identity'  # ['identity', 'batch_norm_2d'] - obsolete, always use identity
    cfg.model.bpbreid.mask_filtering_training = False  # use visibility scores at training - do not have an influence on testing performance yet, to be improved
    cfg.model.bpbreid.mask_filtering_testing = True  # use visibility scores at testing - do have a big influence on testing performance when activated
    cfg.model.bpbreid.last_stride = 1  # last stride of the resnet backbone - 1 for better performance
    cfg.model.bpbreid.dim_reduce = 'after_pooling'  #  where to apply feature dimensionality reduction (before or after global pooling) ['none', 'before_pooling', 'after_pooling', 'before_and_after_pooling', 'after_pooling_with_dropout']
    cfg.model.bpbreid.dim_reduce_output = 512  # reduce feature dimension to this value when above config is not 'none'
    cfg.model.bpbreid.backbone = 'resnet50'  # ['resnet50', 'hrnet32', 'fastreid_resnet_ibn_nl']
    cfg.model.bpbreid.learnable_attention_enabled = True  # use learnable attention mechanism to pool part features, otherwise, use fixed attention weights from external (pifpaf) heatmaps/masks
    cfg.model.bpbreid.test_embeddings = ['bn_foreg', 'parts']  # embeddings to use at inference among ['globl', 'foreg', 'backg', 'conct', 'parts']: append 'bn_' suffix to use batch normed embeddings
    cfg.model.bpbreid.test_use_target_segmentation = 'none'  # ['soft', 'hard', 'none'] - use external part mask to further refine the attention weights at inference
    cfg.model.bpbreid.training_binary_visibility_score = True  # use binary visibility score (0 or 1) instead of continuous visibility score (0 to 1) at training
    cfg.model.bpbreid.testing_binary_visibility_score = True  # use binary visibility score (0 or 1) instead of continuous visibility score (0 to 1) at testing
    cfg.model.bpbreid.shared_parts_id_classifier = False  # if each part branch uses share weights for the identity classifier. Used only when the identity loss is used on part-based embeddings.
    cfg.model.bpbreid.hrnet_pretrained_path = "pretrained_models/" # path to pretrained weights for HRNet backbone, download on our Google Drive or on https://github.com/HRNet/HRNet-Image-Classification
    # number of horizontal stripes desired. When BPBreID is used, this variable will be automatically filled depending
    # on "data.masks.preprocess"
    cfg.model.bpbreid.masks = CN()
    cfg.model.bpbreid.masks.type = 'disk'  # when 'disk' is used, load part masks from storage in 'cfg.model.bpbreid.masks.dir' folder
    # when 'stripes' is used, divide the image in 'cfg.model.bpbreid.masks.parts_num' horizontal stripes in a PCB style.
    # 'stripes' with parts_num=1 can be used to emulate the global method Bag of Tricks (BoT)
    cfg.model.bpbreid.masks.parts_num = 1  # number of part-based embedding to extract. When PCB is used, change this parameter to the number of stripes required
    cfg.model.bpbreid.masks.dir = 'pifpaf_maskrcnn_filtering'  # masks will be loaded from 'dataset_path/masks/<cfg.model.bpbreid.masks.dir>' directory
    cfg.model.bpbreid.masks.preprocess = 'eight'  # how to group the 36 pifpaf parts into smaller human semantic groups ['eight', 'five', 'four', 'two', ...], more combination available inside 'torchreid/data/masks_transforms/__init__.masks_preprocess_pifpaf'
    cfg.model.bpbreid.masks.softmax_weight = 15
    cfg.model.bpbreid.masks.background_computation_strategy = 'threshold'  # threshold, diff_from_max
    cfg.model.bpbreid.masks.mask_filtering_threshold = 0.5

    # data
    cfg.data = CN()
    cfg.data.type = 'image'
    cfg.data.root = 'reid-data'
    cfg.data.sources = ['market1501']
    cfg.data.targets = ['market1501']
    cfg.data.workers = 4 # number of data loading workers, set to 0 to enable breakpoint debugging in dataloader code
    cfg.data.split_id = 0 # split index
    cfg.data.height = 256 # image height
    cfg.data.width = 128 # image width
    cfg.data.combineall = False # combine train, query and gallery for training
    cfg.data.transforms = ['rc', 're']  # data augmentation from ['rf', 'rc', 're', 'cj'] = ['random flip', 'random crop', 'random erasing', 'color jitter']
    cfg.data.ro = CN()  # parameters for random occlusion data augmentation with Pascal VOC, to be improved, not maintained
    cfg.data.ro.path = ""
    cfg.data.ro.p = 0.5
    cfg.data.ro.n = 1
    cfg.data.ro.min_overlap = 0.5
    cfg.data.ro.max_overlap = 0.8
    cfg.data.cj = CN()  # parameters for color jitter data augmentation
    cfg.data.cj.brightness = 0.2
    cfg.data.cj.contrast = 0.15
    cfg.data.cj.saturation = 0.
    cfg.data.cj.hue = 0.
    cfg.data.cj.always_apply = False
    cfg.data.cj.p = 0.5
    cfg.data.norm_mean = [0.485, 0.456, 0.406] # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225] # default is imagenet std
    cfg.data.save_dir = 'logs'  # save figures, images, logs, etc. in this folder
    cfg.data.load_train_targets = False

    # specific datasets
    cfg.market1501 = CN()
    cfg.market1501.use_500k_distractors = False # add 500k distractors to the gallery set for market1501
    cfg.cuhk03 = CN()
    cfg.cuhk03.labeled_images = False # use labeled images, if False, use detected images
    cfg.cuhk03.classic_split = False # use classic split by Li et al. CVPR14
    cfg.cuhk03.use_metric_cuhk03 = False # use cuhk03's metric for evaluation

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomIdentitySampler' # sampler for source train loader
    cfg.sampler.train_sampler_t = 'RandomIdentitySampler' # sampler for target train loader
    cfg.sampler.num_instances = 4 # number of instances per identity for RandomIdentitySampler

    # video reid setting
    cfg.video = CN()
    cfg.video.seq_len = 15 # number of images to sample in a tracklet
    cfg.video.sample_method = 'evenly' # how to sample images from a tracklet 'random'/'evenly'/'all'
    cfg.video.pooling_method = 'avg' # how to pool features over a tracklet

    # train
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.lr = 0.00035
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 120
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 64
    cfg.train.fixbase_epoch = 0 # number of epochs to fix base layers
    cfg.train.open_layers = [
        'classifier'
    ] # layers for training while keeping others frozen
    cfg.train.staged_lr = False # set different lr to different layers
    cfg.train.new_layers = ['classifier'] # newly added layers with default lr
    cfg.train.base_lr_mult = 0.1 # learning rate multiplier for base layers
    cfg.train.lr_scheduler = 'warmup_multi_step'
    cfg.train.stepsize = [40, 70] # stepsize to decay learning rate
    cfg.train.gamma = 0.1 # learning rate decay multiplier
    cfg.train.seed = 1 # random seed
    cfg.train.eval_freq = -1 # evaluation frequency (-1 means to only test after training)
    cfg.train.batch_debug_freq = 0
    cfg.train.batch_log_freq = 0

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9 # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0. # dampening for momentum
    cfg.sgd.nesterov = False # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99 # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9 # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999 # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = 'part_based'  # use part based engine to train bpbreid with GiLt loss
    cfg.loss.part_based = CN()
    cfg.loss.part_based.name = 'part_averaged_triplet_loss' # ['inter_parts_triplet_loss', 'intra_parts_triplet_loss', 'part_max_triplet_loss', 'part_averaged_triplet_loss', 'part_min_triplet_loss', 'part_max_min_triplet_loss', 'part_random_max_min_triplet_loss']
    cfg.loss.part_based.ppl = "cl" # body part prediction loss: ['cl', 'fl', 'dl'] = [cross entropy loss with label smoothing, focal loss, dice loss]
    cfg.loss.part_based.weights = CN()  # weights to apply for the different losses and different types of embeddings, for more details, have a look at 'torchreid/losses/GiLt_loss.py'
    cfg.loss.part_based.weights[GLOBAL] = CN()
    cfg.loss.part_based.weights[GLOBAL].id = 1.
    cfg.loss.part_based.weights[GLOBAL].tr = 0.
    cfg.loss.part_based.weights[FOREGROUND] = CN()
    cfg.loss.part_based.weights[FOREGROUND].id = 1.
    cfg.loss.part_based.weights[FOREGROUND].tr = 0.
    cfg.loss.part_based.weights[CONCAT_PARTS] = CN()
    cfg.loss.part_based.weights[CONCAT_PARTS].id = 1.
    cfg.loss.part_based.weights[CONCAT_PARTS].tr = 0.
    cfg.loss.part_based.weights[PARTS] = CN()
    cfg.loss.part_based.weights[PARTS].id = 0.
    cfg.loss.part_based.weights[PARTS].tr = 1.
    cfg.loss.part_based.weights[PIXELS] = CN()
    cfg.loss.part_based.weights[PIXELS].ce = 0.35
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True # use label smoothing regularizer
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3 # distance margin
    cfg.loss.triplet.weight_t = 1. # weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 0. # weight to balance cross entropy loss

    # test
    cfg.test = CN()
    cfg.test.batch_size = 128
    cfg.test.batch_size_pairwise_dist_matrix = 500  # query to gallery distance matrix is computed on the GPU by batch of gallery samples with this size.
    # To avoid out of memory issue, we don't compute it for all gallery samples at the same time, but we compute it
    # in batches of 'batch_size_pairwise_dist_matrix' gallery samples.
    cfg.test.dist_metric = 'euclidean' # distance metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = True # normalize feature vectors before computing distance
    cfg.test.ranks = [1, 5, 10, 20] # cmc ranks
    cfg.test.evaluate = False # test only
    cfg.test.start_eval = 0 # start to evaluate after a specific epoch
    cfg.test.rerank = False # use person re-ranking
    cfg.test.visrank = False # visualize ranked results (only available when cfg.test.evaluate=True)
    cfg.test.visrank_topk = 10 # top-k ranks to visualize
    cfg.test.visrank_count = 10 # number of top-k ranks to plot
    cfg.test.visrank_q_idx_list = [0, 1, 2, 3, 4, 5]  # list of ids of queries for which we want to plot topk rank. If len(visrank_q_idx_list) < visrank_count, remaining ids will be random
    cfg.test.vis_feature_maps = False
    cfg.test.visrank_per_body_part = False
    cfg.test.vis_embedding_projection = False
    cfg.test.save_features = False # save test set extracted features to disk
    cfg.test.detailed_ranking = True  # display ranking performance for each part individually
    cfg.test.part_based = CN()
    cfg.test.part_based.dist_combine_strat = "mean"  # ['mean', 'max'] local part based distances are combined into a global distance using this strategy

    # inference
    cfg.inference = CN()
    cfg.inference.enabled = False
    cfg.inference.input_folder = ""

    return cfg

keys_to_ignore_in_diff = {
    "cfg.project",
    "cfg.model.save_model_flag",
    "cfg.model.bpbreid.backbone",
    "cfg.model.bpbreid.learnable_attention_enabled",
    "cfg.model.bpbreid.masks.parts_num",
    "cfg.model.bpbreid.masks.dir",
    "cfg.data.type",
    "cfg.data.root",
    "cfg.data.sources",
    "cfg.data.targets",
    "cfg.data.workers",
    "cfg.data.split_id",
    "cfg.data.combineall",
    "cfg.data.save_dir",
    "cfg.train.eval_freq",
    "cfg.train.batch_debug_freq",
    "cfg.train.batch_log_freq",
    "cfg.test.batch_size",
    "cfg.test.batch_size_pairwise_dist_matrix",
    "cfg.test.dist_metric",
    "cfg.test.ranks",
    "cfg.test.evaluate",
    "cfg.test.start_eval",
    "cfg.test.rerank",
    "cfg.test.visrank",
    "cfg.test.visrank_topk",
    "cfg.test.visrank_count",
    "cfg.test.visrank_q_idx_list",
    "cfg.test.vis_feature_maps",
    "cfg.test.visrank_per_body_part",
    "cfg.test.vis_embedding_projection",
    "cfg.test.save_features",
    "cfg.test.detailed_ranking",
    "cfg.train.open_layers",
    "cfg.model.load_weights",
}

def imagedata_kwargs(cfg):
    return {
        'config': cfg,
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'load_train_targets': cfg.data.load_train_targets,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        'train_sampler_t': cfg.sampler.train_sampler_t,
        # image
        'cuhk03_labeled': cfg.cuhk03.labeled_images,
        'cuhk03_classic_split': cfg.cuhk03.classic_split,
        'market1501_500k': cfg.market1501.use_500k_distractors,
        'use_masks': cfg.loss.name == 'part_based',
        'masks_dir': cfg.model.bpbreid.masks.dir,
    }


def videodata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        # video
        'seq_len': cfg.video.seq_len,
        'sample_method': cfg.video.sample_method
    }


def optimizer_kwargs(cfg):
    return {
        'optim': cfg.train.optim,
        'lr': cfg.train.lr,
        'weight_decay': cfg.train.weight_decay,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
        'staged_lr': cfg.train.staged_lr,
        'new_layers': cfg.train.new_layers,
        'base_lr_mult': cfg.train.base_lr_mult
    }


def lr_scheduler_kwargs(cfg):
    return {
        'lr_scheduler': cfg.train.lr_scheduler,
        'stepsize': cfg.train.stepsize,
        'gamma': cfg.train.gamma,
        'max_epoch': cfg.train.max_epoch
    }


def engine_run_kwargs(cfg):
    return {
        'save_dir': cfg.data.save_dir,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'open_layers': cfg.train.open_layers,
        'test_only': cfg.test.evaluate,
        'dist_metric': cfg.test.dist_metric,
        'normalize_feature': cfg.test.normalize_feature,
        'visrank': cfg.test.visrank,
        'visrank_topk': cfg.test.visrank_topk,
        'visrank_q_idx_list': cfg.test.visrank_q_idx_list,
        'visrank_count': cfg.test.visrank_count,
        'use_metric_cuhk03': cfg.cuhk03.use_metric_cuhk03,
        'ranks': cfg.test.ranks,
        'rerank': cfg.test.rerank,
        'save_features': cfg.test.save_features
    }


def display_config_diff(cfg, default_cfg_copy):
    def iterdict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                iterdict(v)
            else:
                if type(v) == list:
                    v = str(v)
                d.update({k: v})
        return d

    ddiff = DeepDiff(iterdict(default_cfg_copy), iterdict(cfg.clone()), ignore_order=True)
    cfg_diff = {}
    if 'values_changed' in ddiff:
        for k, v in ddiff['values_changed'].items():
            reformatted_key = "cfg." + k.replace("root['", "").replace("']['", ".").replace("']", "")
            if "[" in reformatted_key:
                reformatted_key = reformatted_key.split("[")[0]
            reformatted_key_split = reformatted_key.split(".")
            ignore_key = False
            for i in range(2, len(reformatted_key_split) + 1):
                prefix = ".".join(reformatted_key_split[0:i])
                if prefix in keys_to_ignore_in_diff:
                    ignore_key = True
                    break
            if not ignore_key:
                key = re.findall(r"\['([A-Za-z0-9_]+)'\]", k)[-1]
                cfg_diff[key] = v['new_value']
    print("Diff from default config :")
    pprint.pprint(cfg_diff)
    if len(str(cfg_diff)) < 128:
        cfg.project.diff_config = str(cfg_diff)
    else:
        cfg.project.diff_config = str(cfg_diff)[0:124] + "..."