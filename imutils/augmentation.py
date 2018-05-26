import numpy as np

def flip_img(image, axis=0):
    """
    Flip vertically or horizontally a matrix.
    :param image: source image
    :param axis: axis which will be flipped
    :return: the flipped matrix
    """
    return np.flip(image, axis)

def rot90_img(image, clockwise=True):
    """
    Rotate by 90 degrees an image
    :param image: source image
    :para clockwise: True for clockwise rotation
    :return: The rotated image
    """
    axes = (0, 1) if clockwise else (1, 0)
    return np.rot90(image, axes=axes)
