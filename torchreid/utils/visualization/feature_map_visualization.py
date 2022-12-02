import math
import os.path as osp
import json

import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
import torch


# from reid import datasets, models
# from reid.utils.data import transforms as T
# from reid.utils.data.preprocessor import Preprocessor
# from reid.utils.osutils import set_paths
# from reid.utils.serialization import load_checkpoint
# from reid.models.CompactBilinearPooling_dsybaik import CompactBilinearPooling

# Source: https://github.com/yuminsuh/part_bilinear_reid/blob/master/vis_featmap.ipynb
from torchreid.utils import Writer, Logger
from torchreid.utils.constants import PARTS


def flatten(maps):
    flattened = np.transpose(maps, (1, 0, 2, 3)).reshape(maps.shape[1], int(maps.size / maps.shape[1]))
    return flattened


def organize(flattened, num_map, feat_dim, h, w):
    maps = flattened.reshape(feat_dim, num_map, h, w)
    maps = np.transpose(maps, (1, 0, 2, 3))
    return maps


def feat_to_color(maps):
    num_img, dim_feat, h, w = maps.shape
    maps_flatten = flatten(maps)
    maps_flatten_reduced = PCA(n_components=3).fit_transform(maps_flatten.transpose())
    maps_reduced = organize(maps_flatten_reduced.transpose(), num_img, 3, h, w)
    return maps_reduced


def normalize_01(m):
    max_value = m.max()
    min_value = m.min()
    r = max_value - min_value
    n = (m - min_value) / r
    return n


def mapwise_normalize(map1, feats):
    num_sample, num_channel, h, w = map1.shape # FIXME TOO MANY VALUES TO UNPACK WITH INDEITITy MASK
    # mode 1
    # norms = np.linalg.norm(np.sqrt(np.sum(map1**2, axis=(2,3))), axis=1)
    # normalized = map1
    # mode 2
    norms = np.sqrt(np.linalg.norm(feats, axis=1))
    normalized = map1 / np.reshape(norms, (num_sample, 1, 1, 1))
    return normalized


# TODO output big grid of n identities horizontally and m samples per identity vertically
# TODO test without norm stuff
# TODO refactor into beautiful code
# TODO cannot apply PCA batch by batch, should be global
def visualize_pca_multi(maps_all, feats, pids, tag):
    maps_all_reduced = []
    for maps in maps_all[1:]:
        if len(maps.shape) == 4:
            maps_all_reduced.append(feat_to_color(mapwise_normalize(maps, feats)))
    num_person = maps_all_reduced[0].shape[0]

    num_samples = len(pids)
    num_samples_per_id = 0
    for i, pid in enumerate(pids):
        if pid != pids[0]:
            num_samples_per_id = i
            break



    n_rows = num_samples_per_id
    n_cols = math.ceil(num_samples / num_samples_per_id)
    fig = plt.figure(figsize=(n_cols*4, n_rows*2), constrained_layout=False)
    outer_grid = fig.add_gridspec(n_rows, n_cols)

    # count = 0
    # for row in range(4):
    #     for col in range(5):
    #         print("grid {}-{}".format(row, col))
    #         # gridspec inside gridspec
    #         inner_grid = outer_grid[row, col].subgridspec(1, 3)
    #         axs = inner_grid.subplots()  # Create all subplots for the inner grid.
    #         triplet = triplets[count]
    #         pos, anc, neg, pos_dist, neg_dist = triplet
    #         ax1, ax2, ax3 = axs[0], axs[1], axs[2]
    #         show_instance(ax1, pos, pos_dist, green)
    #         show_instance(ax2, anc, 0, black)
    #         show_instance(ax3, neg, neg_dist, red)
    #         count += 1

    # for person_idx in range(min(num_person, num_vis)):
    person_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if person_idx < num_person:
                # print(person_idx)
                # fig, ax = plt.subplots(1, len(maps_all))
                #         fig.set_size_inches((10,10))
                inner_grid = outer_grid[row, col].subgridspec(1, 3)
                axs = inner_grid.subplots()  # Create all subplots for the inner grid.
                for idx, maps_reduced in enumerate([maps_all[0]] + maps_all_reduced):
                    axs[idx].imshow(normalize_01(np.transpose(maps_reduced[person_idx, ::-1, :, :], (1, 2, 0))))
                    axs[idx].set_axis_off()
                person_idx += 1
    # plt.show()

    # plt.tight_layout()
    plt.tight_layout(pad=1.30, h_pad=1.6, w_pad=1.6)
    Logger.current_logger().add_figure("features_part_maps_{}".format(tag), fig, False)
    plt.close(fig)

#         fig.savefig(savepath.format(save_id, person_idx)) 


def display_feature_maps(embeddings_dict, spatial_features, body_part_masks, imgs_path, pids):
    # TODO call at test time: display top 10 ranking for 5 queries? Take 10 random pids with 1 samples in query and
    #  find 9 corresponding samples with same pid in gallery

    # TODO put config
    # TODO fix with new model output

    writer = Writer.current_writer()
    def extract_images(imgs_path):
        imgs = []
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 256)) # TODO size = config
            img = img / 255
            imgs.append(img)

        imgs = np.asarray(imgs)
        imgs = np.transpose(imgs, axes=[0, 3, 1, 2])
        return imgs

    if writer.engine_state.epoch == writer.engine_state.max_epoch-1 and writer.engine_state.batch < 10 and writer.cfg.test.vis_feature_maps: # TODO move away
        # TODO
        body_parts_features = embeddings_dict[PARTS]
        body_part_masks = torch.unsqueeze(body_part_masks, 2)  # [N, M, 1, Hf, Wf]
        spatial_features = torch.unsqueeze(spatial_features, 1)  # [N, 1, D, Hf, Wf]
        images_np = extract_images(imgs_path)
        if len(body_parts_features.shape) == 3:
            body_parts_features = body_parts_features.flatten(1, 2)
        body_parts_features_np = body_parts_features.squeeze().detach().cpu().numpy().copy()
        body_part_masks_np = body_part_masks.squeeze().detach().cpu().numpy().copy()
        spatial_features_np = spatial_features.squeeze().detach().cpu().numpy().copy()
        visualize_pca_multi([images_np, spatial_features_np, body_part_masks_np], body_parts_features_np, pids, "e_{}_b_{}".format(writer.engine_state.epoch, writer.engine_state.batch))

# def sel_random_target(target):
#     labels = np.unique([t[1] for t in target])
#     labels = np.delete(labels, labels == 0)
#     sel_labels = np.random.choice(labels, 200, replace=False)
#     sel_target = [t for t in target if t[1] in sel_labels]
#     _, sel_target = map(list, zip(*sorted(zip([t[1] for t in sel_target], sel_target))))
#     return sel_target
#
# """ Settings """
# exp_dir = "logs/market1501/d2_b250"
# batch_size = 60
# args = json.load(open(osp.join(exp_dir, "args.json"), "r"))
# set_paths('paths')
# np.random.seed(0)
# # np.random.seed(int(time.time()))
# # savepath = 'vis_examples/{}_{}.png'
#
# """ Load dataset """
# dataset = datasets.create(args['dataset'], "data/{}".format(args['dataset']))
# test_transformer = T.Compose([
#     T.RectScale(args['height'], args['width']),
#     T.CenterCrop((args['crop_height'], args['crop_width'])),
#     T.ToTensor(),
#     T.RGB_to_BGR(),
#     T.NormalizeBy(255),
# ])
# target = sel_random_target(list(set(dataset.query) | set(dataset.gallery)))
# test_loader = DataLoader(
#     Preprocessor(target,
#                  root=dataset.images_dir, transform=test_transformer),
#     batch_size=batch_size, num_workers=args['workers'],
#     shuffle=False, pin_memory=True)
#
# """ Load model """
# model = models.create(args['arch'], features=args['features'],
#                       dilation=args['dilation'], initialize=False).cuda()
# model_weight = osp.join(exp_dir, 'epoch_750.pth.tar')
# checkpoint = load_checkpoint(model_weight)
# model.app_feat_extractor.load_state_dict(checkpoint['app_state_dict'])
# model.part_feat_extractor.load_state_dict(checkpoint['part_state_dict'])
# model.eval()
#
# """ Extract feature maps """
# num_test = len(target)
# feat1, feat2, feat_out, h, w = 512, 128, 512, 20, 10
# pool = CompactBilinearPooling(feat1, feat2, feat_out, sum_pool=True)
#
# app_feats = np.zeros((num_test, feat1, h, w))
# part_feats = np.zeros((num_test, feat2, h, w))
# bilinear_feats = np.zeros((num_test, feat_out))
# target_imgs = np.zeros((num_test, 3, args['crop_height'], args['crop_width']))
# for i, (imgs, fnames, pids, _) in enumerate(test_loader):
#     app_feat = model.app_feat_extractor(imgs.cuda())
#     part_feat = model.part_feat_extractor(imgs.cuda())
#     bilinear_feat = pool(app_feat, part_feat)
#
#     i_start = i * batch_size
#     i_end = min((i + 1) * batch_size, num_test)
#     app_feats[i_start:i_end, :, :, :] = app_feat.detach().cpu().numpy().copy()
#     part_feats[i_start:i_end, :, :, :] = part_feat.detach().cpu().numpy().copy()
#     bilinear_feats[i_start:i_end, :] = bilinear_feat.detach().cpu().numpy().copy()
#     target_imgs[i_start:i_end, :, :, :] = imgs.detach().cpu().numpy().copy()
#
# """ Visualize maps """
# num_vis = 200
# visualize_pca_multi([target_imgs[:num_vis], app_feats[:num_vis], part_feats[:num_vis]], bilinear_feats[:num_vis],
#                     num_vis=num_vis, save_id="")