import numpy as np 
import cv2 
import torch 
import time
import random

def apply_lut(image, table):
    """Map an input image to a new one using a lookup table.
    Added in 0.4.0.
    **Supported dtypes**:
        See :func:`~imgaug.imgaug.apply_lut_`.
    Parameters
    ----------
    image : ndarray
        See :func:`~imgaug.imgaug.apply_lut_`.
    table : ndarray or list of ndarray
        See :func:`~imgaug.imgaug.apply_lut_`.
    Returns
    -------
    ndarray
        Image after mapping via lookup table.
    """
    return apply_lut_(np.copy(image), table)


def _normalize_cv2_input_arr_(arr):
    flags = arr.flags
    if not flags["OWNDATA"]:
        arr = np.copy(arr)
        flags = arr.flags
    if not flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def apply_lut_(image, table):
    """Map an input image in-place to a new one using a lookup table.
    Added in 0.4.0.
    **Supported dtypes**:
        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no
    Parameters
    ----------
    image : ndarray
        Image of dtype ``uint8`` and shape ``(H,W)`` or ``(H,W,C)``.
    table : ndarray or list of ndarray
        Table of dtype ``uint8`` containing the mapping from old to new
        values. Either a ``list`` of ``C`` ``(256,)`` arrays or a single
        array of shape ``(256,)`` or ``(256, C)`` or ``(1, 256, C)``.
        In case of ``(256,)`` the same table is used for all channels,
        otherwise a channelwise table is used and ``C`` is expected to match
        the number of channels.
    Returns
    -------
    ndarray
        Image after mapping via lookup table.
        This *might* be the same array instance as provided via `image`.
    """

    image_shape_orig = image.shape
    nb_channels = 1 if len(image_shape_orig) == 2 else image_shape_orig[-1]

    if 0 in image_shape_orig:
        return image

    image = _normalize_cv2_input_arr_(image)

    # [(256,), (256,), ...] => (256, C)
    if isinstance(table, list):
        assert len(table) == nb_channels, (
            "Expected to get %d tables (one per channel), got %d instead." % (
                nb_channels, len(table)))
        table = np.stack(table, axis=-1)

    # (256, C) => (1, 256, C)
    if table.shape == (256, nb_channels):
        table = table[np.newaxis, :, :]

    assert table.shape == (256,) or table.shape == (1, 256, nb_channels), (
        "Expected 'table' to be any of the following: "
        "A list of C (256,) arrays, an array of shape (256,), an array of "
        "shape (256, C), an array of shape (1, 256, C). Transformed 'table' "
        "up to shape %s for image with shape %s (C=%d)." % (
            table.shape, image_shape_orig, nb_channels))

    if nb_channels > 512:
        if table.shape == (256,):
            table = np.tile(table[np.newaxis, :, np.newaxis],
                            (1, 1, nb_channels))

        subluts = []
        for group_idx in np.arange(int(np.ceil(nb_channels / 512))):
            c_start = group_idx * 512
            c_end = c_start + 512
            subluts.append(apply_lut_(image[:, :, c_start:c_end],
                                      table[:, :, c_start:c_end]))

        return np.concatenate(subluts, axis=2)

    assert image.dtype == np.dtype("uint8"), (
        "Expected uint8 image, got dtype %s." % (image.dtype.name,))

    image = cv2.LUT(image, table, dst=image)
    return image



def adjust_contrast_linear(arr, alpha):
    """Adjust contrast by scaling each pixel to ``127 + alpha*(v-127)``.
    **Supported dtypes**:
        * ``uint8``: yes; fully tested (1) (2)
        * ``uint16``: yes; tested (2)
        * ``uint32``: yes; tested (2)
        * ``uint64``: no (3)
        * ``int8``: yes; tested (2)
        * ``int16``: yes; tested (2)
        * ``int32``: yes; tested (2)
        * ``int64``: no (2)
        * ``float16``: yes; tested (2)
        * ``float32``: yes; tested (2)
        * ``float64``: yes; tested (2)
        * ``float128``: no (2)
        * ``bool``: no (4)
        - (1) Handled by ``cv2``. Other dtypes are handled by raw ``numpy``.
        - (2) Only tested for reasonable alphas with up to a value of
              around ``100``.
        - (3) Conversion to ``float64`` is done during augmentation, hence
              ``uint64``, ``int64``, and ``float128`` support cannot be
              guaranteed.
        - (4) Does not make sense for contrast adjustments.
    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.
    alpha : number
        Multiplier to linearly pronounce (``>1.0``), dampen (``0.0`` to
        ``1.0``) or invert (``<0.0``) the difference between each pixel value
        and the dtype's center value, e.g. ``127`` for ``uint8``.
    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.
    """

    # int8 is also possible according to docs
    # but here it seemed like `d` was 0 for CV_8S, causing that to fail
    iinfo = np.iinfo(arr.dtype)

    min_value, center_value, max_value = iinfo.min, iinfo.min + 0.5 * iinfo.max, iinfo.max
    # TODO get rid of this int(...)
    center_value = int(center_value)

    value_range = np.arange(0, 256, dtype=np.float32)

    # 127 + alpha*(I_ij-127)
    # using np.float32(.) here still works when the input is a numpy array
    # of size 1
    alpha = np.float32(alpha)
    table = center_value + alpha * (value_range - center_value)
    table = np.clip(table, min_value, max_value).astype(arr.dtype)
    arr_aug = apply_lut(arr, table)
    return arr_aug

def multiply(image, multiplier):
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]
    multiplier = np.float32(multiplier)
    value_range = np.arange(0, 256, dtype=np.float32)

    value_range = value_range * multiplier
    value_range = np.clip(value_range, 0, 255).astype(image.dtype)
    return apply_lut_(image, value_range)