import torch
import tqdm
import glob
import os
import numpy as np
from scripts.default_config import get_default_config, display_config_diff
from tools.feature_extractor import FeatureExtractor


def extract_part_based_features(extractor, image_list, batch_size=400):

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    all_embeddings = []
    all_visibility_scores = []
    all_masks = []

    images_chunks = chunks(image_list, batch_size)
    for chunk in tqdm.tqdm(images_chunks):
        embeddings, visibility_scores, masks = extractor(chunk)

        embeddings = embeddings.cpu().detach()
        visibility_scores = visibility_scores.cpu().detach()
        masks = masks.cpu().detach()

        all_embeddings.append(embeddings)
        all_visibility_scores.append(visibility_scores)
        all_masks.append(masks)

    all_embeddings = torch.cat(all_embeddings, 0).numpy()
    all_visibility_scores = torch.cat(all_visibility_scores, 0).numpy()
    all_masks = torch.cat(all_masks, 0).numpy()

    return {
        "parts_embeddings": all_embeddings,
        "parts_visibility_scores": all_visibility_scores,
        "parts_masks": all_masks,
    }


def extract_det_idx(img_path):
    return int(os.path.basename(img_path).split("_")[0])


def extract_reid_features(cfg, base_folder, out_path, model=None, model_path=None, num_classes=None):
    extractor = FeatureExtractor(
        cfg,
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_classes=num_classes,
        model=model
    )

    print("Looking for video folders with images crops in {}".format(base_folder))
    folder_list = glob.glob(base_folder + '/*')
    for folder in folder_list:
        image_list = glob.glob(os.path.join(folder, "*.png"))
        image_list.sort(key=extract_det_idx)
        print("{} images to process for folder {}".format(len(image_list), folder))
        results = extract_part_based_features(extractor, image_list, batch_size=50)

        # dump to disk
        video_name = os.path.splitext(os.path.basename(folder))[0]
        parts_embeddings_filename = os.path.join(out_path, "embeddings_" + video_name + ".npy")
        parts_visibility_scores_filanme = os.path.join(out_path, "visibility_scores_" + video_name + ".npy")
        parts_masks_filename = os.path.join(out_path, "masks_" + video_name + ".npy")

        os.makedirs(os.path.dirname(parts_embeddings_filename), exist_ok=True)
        os.makedirs(os.path.dirname(parts_visibility_scores_filanme), exist_ok=True)
        os.makedirs(os.path.dirname(parts_masks_filename), exist_ok=True)

        np.save(parts_embeddings_filename, results['parts_embeddings'])
        np.save(parts_visibility_scores_filanme, results['parts_visibility_scores'])
        np.save(parts_masks_filename, results['parts_masks'])

        print("features saved to {}".format(out_path))
