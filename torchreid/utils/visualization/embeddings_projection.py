import cv2
import torch
import numpy as np

from torchreid.utils import Logger
from torchreid.utils.engine_state import EngineState


def visualize_embeddings(qf, gf, q_pids, g_pids, test_loader, dataset_name, qf_parts_visibility, gf_parts_visibility, mAP, rank1):
    query_dataset = test_loader['query'].dataset
    gallery_dataset = test_loader['gallery'].dataset
    # TODO 1000 identities and 5 samples per identity
    sample_size = 1000
    q_embeddings, q_imgs, q_meta, q_idx_list = extract_samples(qf, query_dataset, sample_size)
    g_embeddings, g_imgs, g_meta, g_idx_list = extract_samples(gf, gallery_dataset, sample_size)

    embeddings = torch.cat([q_embeddings, g_embeddings], 0)
    imgs = torch.cat([q_imgs, g_imgs], 0)
    meta = q_meta + g_meta

    logger = Logger.current_logger()
    for body_part_idx in range(0, embeddings.shape[1]):
        logger.add_embeddings("{} query-gallery embeddings projection for {} with mAP {} and rank-1 {}".format(dataset_name, body_part_idx, mAP, rank1), embeddings[:, body_part_idx], meta, imgs, EngineState.current_engine_state().epoch)


def extract_samples(features, dataset, sample_size):
    sample_size = min(sample_size, len(dataset))
    remaining_idx = np.arange(0, len(dataset))

    idx_list = np.random.choice(remaining_idx, replace=False, size=sample_size)

    embeddings = []
    meta = []
    imgs = []
    for idx in idx_list:
        _, pid, camid, img_path, masks = dataset[idx]
        embeddings.append(features[idx, :, :])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = torch.from_numpy(img)

        imgs.append(img)
        meta.append(str(pid))

    embeddings = torch.stack(embeddings)
    imgs = torch.stack(imgs)
    imgs = imgs.permute(0, 3, 1, 2)

    return embeddings, imgs, meta, idx_list
