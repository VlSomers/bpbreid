# Source: https://github.com/isarandi/synthetic-occlusion/blob/master/augmentation.py
import math
import os.path
import random
import sys
import xml.etree.ElementTree
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import cv2
import PIL.Image
from albumentations import (
    DualTransform, functional
)


def main():
    """Demo of how to use the code"""

    # path = 'something/something/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    path = sys.argv[1]

    print('Loading occluders from Pascal VOC dataset...')
    occluders = load_occluders(pascal_voc_root_path=path)
    print('Found {} suitable objects'.format(len(occluders)))

    original_im = cv2.resize(skimage.data.astronaut(), (256, 256))
    fig, axarr = plt.subplots(3, 3, figsize=(7, 7))
    for ax in axarr.ravel():
        occluded_im = occlude_with_objects(original_im, occluders)
        ax.imshow(occluded_im, interpolation="none")
        ax.axis('off')

    fig.tight_layout(h_pad=0)
    # plt.savefig('examples.jpg', dpi=150, bbox_inches='tight')
    plt.show()


def load_occluders(
        pascal_voc_root_path,
        classes_filter=None,
):
    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    if classes_filter is None:
        # classes_filter = ["person", "bicycle", "boat", "bus", "car", "motorbike", "train", "chair", "dining", "table", "plant", "sofa"]
        classes_filter = ["person", "bicycle", "boat", "bus", "car", "motorbike", "train"]
        # classes_filter = ["person"]
    annotation_paths = list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
    for annotation_path in annotation_paths:
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = (xml_root.find('segmented').text != '0')

        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall('object')):
            is_authorized_class = (obj.find('name').text in classes_filter)
            is_difficult = (obj.find('difficult').text != '0')
            is_truncated = (obj.find('truncated').text != '0')
            if is_authorized_class and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        im_filename = xml_root.find('filename').text
        seg_filename = im_filename.replace('jpg', 'png')

        im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
        seg_path = os.path.join(pascal_voc_root_path, 'SegmentationObject', seg_filename)

        im = np.asarray(PIL.Image.open(im_path))
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8) * 255
            object_image = im[ymin:ymax, xmin:xmax]
            if cv2.countNonZero(object_mask) < 500:
                # Ignore small objects
                continue

            # Reduce the opacity of the mask along the border for smoother blending
            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)

            # Downscale for efficiency
            object_with_mask = resize_by_factor(object_with_mask, 0.5)
            occluders.append(object_with_mask)

    return occluders


def occlude_with_objects(im, occluders, n=1, min_overlap=0.1, max_overlap=0.6):
    """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""

    result = im.copy()
    width_height = np.asarray([im.shape[1], im.shape[0]])
    im_area = im.shape[1] * im.shape[0]
    count = np.random.randint(1, n+1)

    for _ in range(count):
        occluder = random.choice(occluders)
        occluder_area = occluder.shape[1] * occluder.shape[0]
        overlap = random.uniform(min_overlap, max_overlap)
        scale_factor = math.sqrt(overlap * im_area / occluder_area)
        occluder = resize_by_factor(occluder, scale_factor)
        assert (occluder.shape[1] * occluder.shape[0]) / im_area == overlap
        center = np.random.uniform([0, 0], width_height)
        paste_over(im_src=occluder, im_dst=result, center=center)
    return result


def paste_over(im_src, im_dst, center, is_mask=False):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.

    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).

    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """

    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32) / 255

    if is_mask:  # if this is a segmentation mask, just apply alpha erasing
        im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (1 - alpha) * region_dst
    else:
        im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)
        im_area = im_src.shape[1] * im_src.shape[0]
        bbox_overlap = (color_src.shape[0] * color_src.shape[1]) / im_area
        pxls_overlap = np.count_nonzero(alpha) / im_area
        return bbox_overlap, pxls_overlap


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))


class RandomOcclusion(DualTransform):
    def __init__(self,
                 path,
                 im_shape,
                 always_apply=False,
                 p=.5,
                 n=1,
                 min_overlap=0.5,
                 max_overlap=0.8,
                 ):
        super(RandomOcclusion, self).__init__(always_apply, p)
        print('Loading occluders from Pascal VOC dataset...')
        self.all_occluders = load_occluders(pascal_voc_root_path=path)
        self.bbox_overlaps = []
        self.pxls_overlaps = []
        self.count = 0
        self.n = n
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.im_shape = im_shape

    def check_range(self, dimension):
        if isinstance(dimension, float) and not 0 <= dimension < 1.0:
            raise ValueError(
                "Invalid value {}. If using floats, the value should be in the range [0.0, 1.0)".format(dimension)
            )

    def apply(self, image, occluders=(), centers=(), **params):
        for occluder, center in zip(occluders, centers):
            bbox_overlap, pxls_overlap = paste_over(im_src=occluder, im_dst=image, center=center)
            self.bbox_overlaps.append(bbox_overlap)
            self.pxls_overlaps.append(pxls_overlap)
        self.count += 1
        if self.count % 10000 == 0:
            bbox_overlaps = np.array(self.bbox_overlaps)
            pxls_overlaps = np.array(self.pxls_overlaps)
            print("RandomOcclusion #{}: bbox_overlap=[{:.2f},{:.2f},{:.2f}], pxls_overlap=[{:.2f},{:.2f},{:.2f}]"
                  .format(self.count,
                          bbox_overlaps.min(),
                          bbox_overlaps.max(),
                          bbox_overlaps.mean(),
                          pxls_overlaps.min(),
                          pxls_overlaps.max(),
                          pxls_overlaps.mean()
                          )
                  )
        return image

    def apply_to_mask(self, image, occluders=(), centers=(), **params):
        for occluder, center in zip(occluders, centers):
            paste_over(im_src=occluder, im_dst=image, center=center, is_mask=True)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        count = np.random.randint(1, self.n + 1)
        width_height = np.asarray([img.shape[1], img.shape[0]])
        im_area = self.im_shape[1] * self.im_shape[0]
        occluders = []
        centers = []
        for _ in range(count):
            occluder = random.choice(self.all_occluders)
            occluder_area = occluder.shape[1] * occluder.shape[0]
            overlap = random.uniform(self.min_overlap, self.max_overlap)
            scale_factor = math.sqrt(overlap * im_area / occluder_area)
            occluder = resize_by_factor(occluder, scale_factor)
            # assert abs((occluder.shape[1] * occluder.shape[0]) / im_area - overlap) < 0.005
            center = np.random.uniform([0, 0], width_height)
            occluders.append(occluder)
            centers.append(center)

        return {"occluders": occluders,
                "centers": centers}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "max_holes",
            "max_height",
            "max_width",
            "min_holes",
            "min_height",
            "min_width",
            "fill_value",
            "mask_fill_value",
        )


if __name__ == '__main__':
    main()
