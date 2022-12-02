import cv2
import matplotlib.pyplot as plt

__all__ = ['show_triplet']

# try:
#     import matplotlib.cm
#     CMAP_JET = copy.copy(matplotlib.cm.get_cmap('jet'))
#     CMAP_JET.set_bad('white', alpha=0.5)
# except ImportError:
#     CMAP_JET = None
from torchreid.utils import Logger
from torchreid.utils.engine_state import EngineState

red = [1, 0, 0]
green = [0, 1, 0]
black = [0, 0, 0]
img_size = (128, 256)

def show_triplet_grid(triplets):

    fig11 = plt.figure(figsize=(40, 50), constrained_layout=False)
    outer_grid = fig11.add_gridspec(4, 5)

    count = 0
    for a in range(4):
        for b in range(5):
            print("grid {}-{}".format(a, b))
            # gridspec inside gridspec
            inner_grid = outer_grid[a, b].subgridspec(1, 3)
            axs = inner_grid.subplots()  # Create all subplots for the inner grid.
            triplet = triplets[count]
            pos, anc, neg, pos_dist, neg_dist = triplet
            ax1, ax2, ax3 = axs[0], axs[1], axs[2]
            show_instance(ax1, pos, pos_dist, green)
            show_instance(ax2, anc, 0, black)
            show_instance(ax3, neg, neg_dist, red)
            count += 1

    # show only the outside spines
    # for ax in fig11.get_axes():
    #     ax.spines['top'].set_visible(ax.is_first_row())
    #     ax.spines['bottom'].set_visible(ax.is_last_row())
    #     ax.spines['left'].set_visible(ax.is_first_col())
    #     ax.spines['right'].set_visible(ax.is_last_col())
    Logger.current_logger().add_figure("Batch triplets", fig11, EngineState.current_engine_state().epoch)
    # plt.show()
    # plt.waitforbuttonpress()

def show_triplet(anc, pos, neg, pos_dist, neg_dist):
    # instance = (image, masks, id, body_part_id, body_part_name)
    f, axarr = plt.subplots(1, 3)
    ax1, ax2, ax3 = axarr[0], axarr[1], axarr[2]
    show_instance(ax1, pos, pos_dist, green)
    show_instance(ax2, anc, 0, black)
    show_instance(ax3, neg, neg_dist, red)
    f.matplotlib_show()
    plt.waitforbuttonpress()


def add_border(img, color):
    # border widths
    top, bottom, left, right = [5] * 4
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def show_instance(ax, instance, dist, color):
    mask_idx = instance[2]
    body_part = instance[3]
    img = instance[0]
    mask = instance[1]

    # img = overlay_mask_1(img, mask)
    img = cv2.resize(img, img_size)
    # img = add_border(img, color)

    ax.imshow(img)
    mask = cv2.resize(mask, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # mask = add_border(mask, color)
    ax.imshow(mask, cmap='jet', vmin=0, vmax=1, alpha=0.5)

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False,
                   labelleft=False)
    ax.set_title("Id = {}\n{}".format(mask_idx, body_part))
    ax.set_xlabel('Dist = {}'.format(dist))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color(color)
        ax.spines[axis].set_linewidth(4)