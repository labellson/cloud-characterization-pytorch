import sys
import os
import numpy as np


def get_images_names_from(directory, name_only=False, key=None):
    """
    Return a list of image paths
    :param path: folder with images
    :return: list
    """
    file_list = list()
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)) and not f == sys.argv[0]:
            if name_only:
                file_list.append(f)
            else:
                file_list.append(os.path.join(directory, f))

    if key is not None:
        return sorted(file_list, key=key)
    else:
        return file_list


def sliding_window(image, windowSize, stepSize=4):
    """
    Slice a window over an image with the requested params
    :param image: source image
    :param windowSize: tuple with size (x, y) of window
    :param stepSize: step in pixels between windows
    :return: a tuple with the top left corner of the window,
             and the window -> (x, y, roi)
    """
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def non_max_suppression(boxes, overlap_thres=0.3):
    """
    Apply Non Maxima Suppression on a set of overlaping image windows
    :param boxes: image windows [(x0, y0, x1, y1), ...]
    :param overlap_thres: overlapping area threshold between boxes
    :return: an array with non overlapping boxes [(x0, y0, x1, y1), ...]
    """
    if len(boxes) == 0:
        return list()

    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    # List of picked indexes
    pick = list()

    # Grab coords of top left and bottom right
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]

    area = (x1 - x0 + 1) * (y1 - y0 + 1)
    idx = np.argsort(y1)

    # Loop until no indexes remain on index list
    while len(idx) > 0:
        # Grab last idx and add its value to the picked list
        last = len(idx) - 1
        i = idx[last]
        pick.append(i)

        # Find the largest top left (x,y) coordinate and the smallest bottom
        # right (x,y) coordinate of the bounding boxes with the current one
        xx0 = np.maximum(x0[i], x0[idx[:last]])
        yy0 = np.maximum(y0[i], y0[idx[:last]])
        xx1 = np.minimum(x1[i], x1[idx[:last]])
        yy1 = np.minimum(y1[i], y1[idx[:last]])

        w = np.maximum(0, xx1 - xx0 + 1)
        h = np.maximum(0, yy1 - yy0 + 1)

        overlap = (w * h) / area[idx[:last]]

        # Delete all de indexes correpondant to overlaping boxes
        idx = np.delete(idx, np.concatenate(([last],
                                             np.where(overlap >
                                                      overlap_thres)[0])))

    return boxes[pick].astype('int')
