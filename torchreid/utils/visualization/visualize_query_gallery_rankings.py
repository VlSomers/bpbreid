import ntpath
import random

import cv2
import matplotlib
import numpy as np

from torchreid.utils import Logger, perc
from torchreid.utils.engine_state import EngineState

GRID_SPACING_V = 100
GRID_SPACING_H = 100
QUERY_EXTRA_SPACING = 30
TOP_MARGIN = 350
LEFT_MARGIN = 150
RIGHT_MARGIN = 500
BOTTOM_MARGIN = 300
ROW_BACKGROUND_LEFT_MARGIN = 75
ROW_BACKGROUND_RIGHT_MARGIN = 75
LEFT_TEXT_OFFSET = 10
BW = 12  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (255, 255, 0)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 0, 0)
TEXT_LINE_TYPE = cv2.LINE_AA
WIDTH = 128
HEIGHT = 256
cmap = matplotlib.cm.get_cmap('hsv')

# TODO document and make code easier to read and adapt, i.e. less intricate
def visualize_ranking_grid(distmat, body_parts_distmat, test_loader, dataset_name, qf_parts_visibility, gf_parts_visibility, q_parts_masks, g_parts_masks, mAP, rank1, save_dir, topk, visrank_q_idx_list, visrank_count, config=None, bp_idx=None):
    num_q, num_g = distmat.shape
    query_dataset = test_loader['query'].dataset
    gallery_dataset = test_loader['gallery'].dataset
    assert num_q == len(query_dataset)
    assert num_g == len(gallery_dataset)
    indices = np.argsort(distmat, axis=1)

    mask_filtering_flag = qf_parts_visibility is not None or gf_parts_visibility is not None
    if qf_parts_visibility is None:
        qf_parts_visibility = np.ones((num_q, body_parts_distmat.shape[0]), dtype=bool)

    if gf_parts_visibility is None:
        gf_parts_visibility = np.ones((num_g, body_parts_distmat.shape[0]), dtype=bool)

    n_missing = visrank_count - len(visrank_q_idx_list)
    if n_missing > 0:
        q_idx_list = visrank_q_idx_list
        remaining_idx = np.arange(0, num_q)
        q_idx_list = np.append(q_idx_list, np.random.choice(remaining_idx, replace=False, size=n_missing))
    elif n_missing < 0:
        q_idx_list = np.array(visrank_q_idx_list[:visrank_count])
    else:
        q_idx_list = np.array(visrank_q_idx_list)

    q_idx_list = q_idx_list.astype(int)
    print("visualize_ranking_grid for dataset {}, bp {} and ids {}".format(dataset_name, bp_idx, q_idx_list))
    for q_idx in q_idx_list:
        if q_idx >= len(query_dataset):
            # FIXME this happen when using multiple target dataset with 'visrank_q_idx_list' provided for another dataset
            new_q_idx = random.randint(0, len(query_dataset)-1)
            print("Invalid query index {}, using random index {} instead".format(q_idx, new_q_idx))
            q_idx = new_q_idx
        query = query_dataset[q_idx]
        qpid, qcamid, qimg_path = query['pid'], query['camid'], query['img_path']
        qmasks = q_parts_masks[q_idx]
        if bp_idx is not None:
            qmasks = qmasks[bp_idx:bp_idx+1]
        query_sample = (q_idx, qpid, qcamid, qimg_path, qmasks, qf_parts_visibility[q_idx, :])
        gallery_topk_samples = []
        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gallery = gallery_dataset[g_idx]
            gpid, gcamid, gimg_path = gallery['pid'], gallery['camid'], gallery['img_path']
            invalid = test_loader['query'].dataset.gallery_filter(np.array(qpid),
                                                                  np.array(qcamid),
                                                                  None,
                                                                  np.array(gpid),
                                                                  np.array(gcamid),
                                                                  None).item()
            invalid = invalid or distmat[q_idx, g_idx] < 0

            if not invalid:
                # matched = gpid == qpid
                gmasks = g_parts_masks[g_idx]
                if bp_idx is not None:
                    gmasks = gmasks[bp_idx:bp_idx+1]
                gallery_sample = (g_idx, gpid, gcamid, gimg_path, gmasks, gf_parts_visibility[g_idx, :], qpid == gpid,
                                  distmat[q_idx, g_idx],
                                  body_parts_distmat[:, q_idx, g_idx])
                gallery_topk_samples.append(gallery_sample)
                rank_idx += 1
                if rank_idx > topk:
                    break
        if len(gallery_topk_samples) > 0:
            show_ranking_grid(query_sample, gallery_topk_samples, mAP, rank1, dataset_name, config, mask_filtering_flag, bp_idx)
        else:
            print("Skip ranking plot of query id {} ({}), no valid gallery available".format(q_idx, qimg_path))


def show_ranking_grid(query_sample, gallery_topk_samples, mAP, rank1, dataset_name, config, mask_filtering_flag, bp_idx=None):
    qidx, qpid, qcamid, qimg_path, qmasks, qf_parts_visibility = query_sample

    topk = len(gallery_topk_samples)
    bp_num = len(qf_parts_visibility)

    num_cols = bp_num + 1
    num_rows = topk + 1
    grid_img = 255 * np.ones(
        (
            num_rows * HEIGHT + (num_rows + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN + BOTTOM_MARGIN,
            num_cols * WIDTH + (num_cols + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN + RIGHT_MARGIN,
            3
        ),
        dtype=np.uint8
    )

    samples = [query_sample] + gallery_topk_samples

    insert_background_line(grid_img, BLUE, 0, HEIGHT, 120, 0)
    insert_background_line(grid_img, BLUE, len(samples), HEIGHT, 0, -75)


    pos = (int(grid_img.shape[1]/2), 0)
    filtering_str = "body part filtering with threshold {}".format(config.model.bpbreid.masks.mask_filtering_threshold) if config.model.bpbreid.mask_filtering_testing else "no body part filtering"
    align_top_text(grid_img, "Ranking for dataset {}, {}, pid {}, mAP {:.2f}%, rank1 {:.2f}%, loss {}, {}".format(dataset_name, config.project.job_id, qpid, mAP * 100, rank1 * 100, config.loss.part_based.name, filtering_str), pos, 3.5, 7, 120)

    for row, sample in enumerate(samples):
        display_sample_on_row(grid_img, sample, row, (WIDTH, HEIGHT), mask_filtering_flag, qf_parts_visibility)

    for col in range(1, num_cols):
        parts_visibility_count = 0
        row = topk+1
        bp_idx = col - 1
        distances = []
        for i, sample in enumerate(samples):
            if i == 0:
                idx, pid, camid, img_path, masks, parts_visibility = sample
            else:
                idx, pid, camid, img_path, masks, parts_visibility, matched, dist_to_query, body_parts_dist_to_query = sample
                distances.append(body_parts_dist_to_query[bp_idx])
            parts_visibility_count += parts_visibility[bp_idx]
        distances = np.asarray(distances)
        min = distances.min()
        max = distances.max()
        mean = distances.mean()
        pos = (col * WIDTH + int(WIDTH / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
               (row) * HEIGHT + int(HEIGHT / 2) + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)

        align_top_multi_text(grid_img, "Bp={}/{}\nMin={:.1f}\nMean={:.1f}\nMax={:.1f}".format(
                parts_visibility_count, topk + 1, min, mean, max), pos, 1, 2, 60)

    if bp_idx is not None:
        filename = "_{}_{}_qidx_{}_qpid_{}_{}_part_{}.jpg".format(config.project.job_id, dataset_name, qidx, qpid, ntpath.basename(qimg_path), bp_idx)
    else:
        filename = "_{}_{}_qidx_{}_qpid_{}_{}.jpg".format(config.project.job_id, dataset_name, qidx, qpid, ntpath.basename(qimg_path))
    # path = os.path.join(save_dir, filename)
    # Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    # cv2.imwrite(path, grid_img)
    Logger.current_logger().add_image("Ranking grid", filename, cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB), EngineState.current_engine_state().epoch)


def insert_background_line(grid_img, match_color, row, height, padding_top=0, padding_bottom=0):
    alpha = 0.1
    color = (255 * (1-alpha) + match_color[0] * alpha,
             255 * (1-alpha) + match_color[1] * alpha,
             255 * (1-alpha) + match_color[2] * alpha)
    hs = row * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN - int(GRID_SPACING_V/2) + 15 - padding_top
    he = (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN + int(GRID_SPACING_V/2) + 15 + padding_bottom
    ws = ROW_BACKGROUND_LEFT_MARGIN
    we = grid_img.shape[1] - ROW_BACKGROUND_RIGHT_MARGIN
    grid_img[hs:he, ws:we, :] = color


def display_sample_on_row(grid_img, sample, row, img_shape, mask_filtering_flag, q_parts_visibility):
    if row == 0:
        idx, pid, camid, img_path, masks, parts_visibility = sample
        matched, dist_to_query, body_parts_dist_to_query = None, None, None
    else:
        idx, pid, camid, img_path, masks, parts_visibility, matched, dist_to_query, body_parts_dist_to_query = sample

    masks = masks.numpy()
    width, height = img_shape
    bp_num = masks.shape[0]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height))

    for col in range(0, bp_num + 1):
        bp_idx = col - 1
        if row == 0 and col == 0:
            img_to_insert = img
            img_to_insert = make_border(img_to_insert, BLUE, BW)
            pos = ((bp_num + 1) * width + (bp_num + 2) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                   row * height + int(height / 2) + (row + 1) * GRID_SPACING_V + TOP_MARGIN)
            align_left_multitext(grid_img, "*Id = {}*\n"
                                      "Visible = {}/{}".format(
                pid, parts_visibility.sum(), bp_num), pos, 1.1, 2, 15)
        elif col == 0:
            match_color = GREEN if matched else RED
            insert_background_line(grid_img, match_color, row, height)
            img_to_insert = make_border(img, match_color, BW)
            pos = (LEFT_MARGIN + GRID_SPACING_H,
                   row * height + int(height / 2) + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
            align_right_text(grid_img, str(row), pos, 3, 6, 30)
            pos = (LEFT_MARGIN + GRID_SPACING_H + int(width / 2),
                   (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
            g_to_q_vis_score = np.sqrt(q_parts_visibility * parts_visibility).sum() / bp_num
            align_top_text(grid_img, "{}% | {:.2f}".format(int(perc(g_to_q_vis_score, 0)), dist_to_query), pos, 1.2, 2, 10)

            pos = ((bp_num + 1) * width + (bp_num + 2) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                   row * height + int(height / 2) + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
            if len(parts_visibility) == 1 or parts_visibility.sum() == 0:
                valid_body_parts_dist = body_parts_dist_to_query
            else:
                valid_body_parts_dist = body_parts_dist_to_query[parts_visibility > 0]

            align_left_multitext(grid_img, "*Id = {}*\n"
                                      "Idx = {}\n"
                                      "Cam id = {}\n"
                                      "Name = {}\n"
                                      "Bp Visibles = {}/{}\n"
                                      "[{:.2f}; {:.2f}; {:.2f}]\n"
                                      "[{:.2f}; {:.2f}; {:.2f}]".format(
                pid, idx, camid, ntpath.basename(img_path), (parts_visibility > 0).sum(), bp_num,
                body_parts_dist_to_query.min(), body_parts_dist_to_query.mean(), body_parts_dist_to_query.max(),
                valid_body_parts_dist.min(), valid_body_parts_dist.mean(), valid_body_parts_dist.max()), pos, 1, 2, 15, match_color)
        else:
            if row == 0:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       TOP_MARGIN + GRID_SPACING_V)
                align_bottom_text(grid_img, str(bp_idx), pos, 2, 5, 35)
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + TOP_MARGIN)
                align_top_text(grid_img, "{}%".format(int(perc(parts_visibility[bp_idx], 0))), pos, 0.9, 2, 10)
            if row != 0:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
                thickness = 3 if body_parts_dist_to_query.argmax() == bp_idx or body_parts_dist_to_query.argmin() == bp_idx else 2
                align_top_text(grid_img, "{}% | {:.2f}".format(int(perc(parts_visibility[bp_idx], 0)), body_parts_dist_to_query[bp_idx]), pos, 0.9, thickness, 10)
            mask = masks[bp_idx, :, :]
            img_with_mask_overlay = mask_overlay(img, mask, interpolation=cv2.INTER_CUBIC)
            if mask_filtering_flag:
                # match_color = GREEN if parts_visibility[bp_idx] else RED
                match_color = cmap(parts_visibility[bp_idx].item()/3, bytes=True)[0:-1]  # divided by three because hsv colormap goes from red to green inside [0, 0.333]
                img_to_insert = make_border(img_with_mask_overlay, (int(match_color[2]), int(match_color[1]), int(match_color[0])), BW)
            else:
                img_to_insert = img_with_mask_overlay

        insert_img_into_grid(grid_img, img_to_insert, row, col)


def mask_overlay(img, mask, clip=True, interpolation=cv2.INTER_NEAREST):
    width, height = img.shape[1], img.shape[0]
    mask = cv2.resize(mask, dsize=(width, height), interpolation=interpolation)
    if clip:
        mask = np.clip(mask, 0, 1)
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = np.interp(mask, (mask.min(), mask.max()), (0, 255)).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    masked_img = cv2.addWeighted(img, 0.5, mask_color.astype(img.dtype), 0.5, 0)
    return masked_img


def align_top_text(img, text, pos, fontScale=1.0, thickness=1, padding=4):
    textsize = cv2.getTextSize(text, TEXT_FONT, fontScale, thickness)[0]
    textX = int(pos[0] - (textsize[0] / 2))
    textY = pos[1] + textsize[1] + padding
    cv2.putText(img, text, (textX, textY), TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=thickness,
                lineType=TEXT_LINE_TYPE)


def align_top_multi_text(img, text, pos, fontScale=1.0, thickness=1, padding=4, text_color=(0, 0, 0)):
    v_padding = 20
    text_lines = text.split('\n')
    text_line_height = cv2.getTextSize(text_lines[0], TEXT_FONT, fontScale, thickness)[0][1]
    text_height = len(text_lines) * text_line_height + (len(text_lines)-1) * v_padding
    textY = int(pos[1] - text_height + text_line_height) + padding

    for i, text_line in enumerate(text_lines):
        bold_marker = "*"
        bold = text_line.startswith(bold_marker) and text_line.endswith(bold_marker)
        line_thickness = thickness+1 if bold else thickness
        if bold:
            text_line = text_line[len(bold_marker):len(text_line)-len(bold_marker)]
        textsize = cv2.getTextSize(text_line, TEXT_FONT, fontScale, thickness)[0]
        text_line_pos = (int(pos[0] - (textsize[0] / 2)), textY + (text_line_height + v_padding) * i)
        text_color = text_color if i == 0 else TEXT_COLOR
        cv2.putText(img, text_line, text_line_pos, TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=line_thickness,
                    lineType=TEXT_LINE_TYPE)


def align_bottom_text(img, text, pos, fontScale=1.0, thickness=1, padding=4):
    textsize = cv2.getTextSize(text, TEXT_FONT, fontScale, thickness)[0]
    textX = int(pos[0] - (textsize[0] / 2))
    textY = pos[1] - padding
    cv2.putText(img, text, (textX, textY), TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=thickness,
                lineType=TEXT_LINE_TYPE)


def align_right_text(img, text, pos, fontScale=1.0, thickness=1, padding=4):
    textsize = cv2.getTextSize(text, TEXT_FONT, fontScale, thickness)[0]
    textX = pos[0] - textsize[0] - padding
    textY = int(pos[1] + (textsize[1] / 2))
    cv2.putText(img, text, (textX, textY), TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=thickness,
                lineType=TEXT_LINE_TYPE)


def align_left_multitext(img, text, pos, fontScale=1.0, thickness=1, padding=4, text_color=(0, 0, 0)):
    v_padding = 20
    text_lines = text.split('\n')
    text_line_height = cv2.getTextSize(text_lines[0], TEXT_FONT, fontScale, thickness)[0][1]
    text_height = len(text_lines) * text_line_height + (len(text_lines)-1) * v_padding
    textX = pos[0] + padding
    textY = int(pos[1] - (text_height / 2) + text_line_height)

    for i, text_line in enumerate(text_lines):
        bold_marker = "*"
        bold = text_line.startswith(bold_marker) and text_line.endswith(bold_marker)
        line_thickness = thickness+1 if bold else thickness
        if bold:
            text_line = text_line[len(bold_marker):len(text_line)-len(bold_marker)]
        pos = (textX, textY + (text_line_height + v_padding) * i)
        text_color = text_color if i == 0 else TEXT_COLOR
        cv2.putText(img, text_line, pos, TEXT_FONT, fontScale=fontScale, color=text_color, thickness=line_thickness,
                    lineType=TEXT_LINE_TYPE)


def centered_text(img, text, pos, fontScale=1, thickness=1):
    textsize = cv2.getTextSize(text, TEXT_FONT, fontScale, thickness)[0]
    textX = int(pos[0] - (textsize[0] / 2))
    textY = int(pos[1] + (textsize[1] / 2))
    cv2.putText(img, text, (textX, textY), TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=thickness,
                lineType=TEXT_LINE_TYPE)


def insert_img_into_grid(grid_img, img, row, col):
    extra_spacing_h = QUERY_EXTRA_SPACING if row > 0 else 0
    extra_spacing_w = QUERY_EXTRA_SPACING if col > 0 else 0
    width, height = img.shape[1], img.shape[0]
    hs = row * height + (row + 1) * GRID_SPACING_V + extra_spacing_h + TOP_MARGIN
    he = (row + 1) * height + (row + 1) * GRID_SPACING_V + extra_spacing_h + TOP_MARGIN
    ws = col * width + (col + 1) * GRID_SPACING_H + extra_spacing_w + LEFT_MARGIN
    we = (col + 1) * width + (col + 1) * GRID_SPACING_H + extra_spacing_w + LEFT_MARGIN
    grid_img[hs:he, ws:we, :] = img


def make_border(img, border_color, bw):
    img_b = cv2.copyMakeBorder(
        img,
        bw, bw, bw, bw,
        cv2.BORDER_CONSTANT,
        value=border_color
    )
    img_b = cv2.resize(img_b, (img.shape[1], img.shape[0]))
    return img_b

#####################################
#   Matplotlib version - too slow   #
#####################################

# GRID_SPACING = 20
# QUERY_EXTRA_SPACING = 60
# BW = 12  # border width
# GREEN = (0, 255, 0)
# RED = (0, 0, 255)
# BLUE = (255, 0, 0)
# YELLOW = (255,255,0)
# FONT = cv2.FONT_HERSHEY_SIMPLEX
# TEXT_COLOR = (0, 0, 0)
# # width = 128
# # height = 256
#
#
# def mask_overlay(img, mask, clip=True):
#     width, height = img.shape[1], img.shape[0]
#     mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
#     if clip:
#         mask = np.clip(mask, 0, 1)
#         mask = (mask*255).astype(np.uint8)
#     else:
#         mask = np.interp(mask, (mask.min(), mask.max()), (0, 255)).astype(np.uint8)
#     mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
#     mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
#     masked_img = cv2.addWeighted(img, 0.5, mask_color, 0.5, 0)
#     return masked_img
#
#
# def show_ranking_grid(query_sample, gallery_topk_samples, config, osp=None):
#     width = 128
#     height = 256
#     samples = [query_sample] + gallery_topk_samples
#
#     print('start {}'.format(time.time()))
#
#     plt.close('all')
#     fig = plt.figure(figsize=(100, 66), constrained_layout=True)
#     outer_grid = fig.add_gridspec(len(samples), 1)
#     # outer_grid = plt.GridSpec(len(samples), 1, wspace=1, hspace=1)
#
#     for row, sample in enumerate(samples):
#         print('row {} {}'.format(row, time.time()))
#         display_sample_on_row(outer_grid[row, 0], sample, row, (width, height))
#
#     # plt.savefig('/Users/vladimirsomers/Downloads/test_ranking_viz_matplotlib/test_grid_viz_plt_{}.pdf'.format(int(time.time())), format='pdf')
#     print('savefig {}'.format(time.time()))
#     plt.savefig('/Users/vladimirsomers/Downloads/test_ranking_viz_matplotlib/test_grid_viz_plt_{}.jpg'.format(int(time.time())), format='jpg')
#     print('end {}'.format(time.time()))
#     plt.close('all')
#     # plt.show()
#     # plt.waitforbuttonpress()
#
#
# def display_sample_on_row(subplot, sample, row, img_shape):
#     if row == 0:
#         pid, camid, img_path, masks_path, parts_visibility = sample
#         matched, dist_to_query, body_parts_dist_to_query = None, None, None
#     else:
#         pid, camid, img_path, masks_path, parts_visibility, matched, dist_to_query, body_parts_dist_to_query = sample
#
#     width, height = img_shape
#     masks = read_masks(masks_path)
#     bp_num = masks.shape[0]
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (width, height))
#     cols = bp_num+1
#
#     inner_grid = subplot.subgridspec(1, cols)
#     axs = inner_grid.subplots()
#     # plt.subplots_adjust(right=0.8)
#
#     for col in range(0, cols):
#         if row == 0 and col == 0:
#             img_to_insert = img
#         elif col == 0:
#             border_color = GREEN if matched else RED
#             img_to_insert = make_border(img, border_color, BW)
#         else:
#             bp_idx = col - 1
#             border_color = GREEN if parts_visibility[bp_idx] else RED
#             mask = masks[bp_idx, :, :]
#             img_with_mask_overlay = mask_overlay(img, mask)
#             img_to_insert = make_border(img_with_mask_overlay, border_color, BW)
#
#         ax = axs[col]
#         ax.imshow(img_to_insert)
#         ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False,
#                        labelleft=False)
#         # ax.set_title("Id = {}\n{}".format(mask_idx, body_part))
#         # ax.set_xlabel('Dist = {}'.format(dist))
#         # for axis in ['top', 'bottom', 'left', 'right']:
#         #     ax.spines[axis].set_color(color)
#         #     ax.spines[axis].set_linewidth(4)
#
#
# def insert_img_into_grid(grid_img, img, row, col):
#     extra_spacing_h = QUERY_EXTRA_SPACING if row > 0 else GRID_SPACING
#     extra_spacing_w = QUERY_EXTRA_SPACING if col > 0 else GRID_SPACING
#     width, height = img.shape[1], img.shape[0]
#     hs = (row) * height + row * GRID_SPACING + extra_spacing_h
#     he = (row + 1) * height + row * GRID_SPACING + extra_spacing_h
#     ws = (col) * width + col * GRID_SPACING + extra_spacing_w
#     we = (col + 1) * width + col * GRID_SPACING + extra_spacing_w
#     grid_img[hs:he, ws:we, :] = img
#
#
# def make_border(img, border_color, bw):
#     img_b = cv2.copyMakeBorder(
#         img,
#         bw, bw, bw, bw,
#         cv2.BORDER_CONSTANT,
#         value=border_color
#     )
#     img_b = cv2.resize(img_b, (img.shape[1], img.shape[0]))
#     return img_b
