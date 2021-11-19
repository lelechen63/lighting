import os
import cv2
import numpy as np
import imageio

root = '/data/home/uss00022/lelechen/github/lighting/checkpoints/gmesh_test/web/images'
images = []
for i in os.listdir('/data/home/uss00022/lelechen/github/lighting/checkpoints/gmesh_test/tmp'):
    images.append(imageio.imread('/data/home/uss00022/lelechen/github/lighting/checkpoints/gmesh_test/tmp/' + i))
    imageio.mimsave('/data/home/uss00022/lelechen/github/lighting/checkpoints/gmesh_test/tmp/movie.gif', images)


# gt = []
# for i in os.listdir(root):
#     if 'gt' in i:
#         gt.append(i)
# gt.sort()
# j = 0
# for i in gt:
#     gtp = root + '/' + i 
#     gti = cv2.imread(gtp)
#     recp = root + '/' + i.replace('gt','rec')
#     reci = cv2.imread(recp)
#     new  = np.concatenate((reci, gti), axis=1)
#     cv2.imwrite('/data/home/uss00022/lelechen/github/lighting/checkpoints/gmesh_test/tmp/%03d.png'%j, new)
#     j +=1



# path = '/data/home/uss00022/lelechen/data/Facescape/textured_meshes'
# for id in os.listdir(path):
#     c = path +'/' +  id
#     command = 'rm -rf ' + c + '/dpmap'
#     os.system(command)
# import numpy as np 
# import cv2 
# import torch 
# import time
# from imgaug import augmenters as iaa
# import random

# def np_normalize(x, low= 0, high = 1):
#     # input x: size 256x256x3, output: 3x245x256
#     if x.shape[0] != 3:
#         x = x.transpose(2, 0, 1)
#     x_max = x.reshape(*x.shape[:-2], -1).max(axis=-1)[0][..., None, None]

#     x_min = x.reshape(*x.shape[:-2], -1).min(axis=-1)[0][..., None, None]
#     scale = (high - low) / (x_max - x_min)

#     x =  (x - x_min) * scale + low

#     if x.shape[2] == 3:
#         x = x.transpose(2,0,1)
#     return x


# def np_augument_tex_color( img, smoothness=100, directionality=0.9, noise_sigma=2.0, contrast=20.0 ):
#     # img_device = img.device
#     # img : 3,256,256 (0-255)
#     # intensity histogram augmentation
#     if smoothness > 0:
#         x = np.arange(256).astype(float) / 255 - 0.5
#         smoothness = int(smoothness)
#         u = np.random.rand(smoothness, 3)
#         a = u[:, 0:1] - 0.5
#         b = u[:, 1:2] * 99 + 1
#         c = u[:, 2:3] * 10
#         x_sum = (c / (1.0 + np.exp(-b * (x[None, :] - a)))).sum(0)
#         F = x_sum[img]
#     else:
#         F = img.astype(float)
    
#     # spatially varying intensity multiplier (second degree polynomial)
#     if directionality > 0:
#         h, w = F.shape[-2:]
#         iy, ix = np.meshgrid(
#             (np.arange(h).astype(float) / (h // 2)) - 1,
#             (np.arange(w).astype(float) / (w // 2)) - 1,
#         )
#         iy = iy.reshape(*([1] * len(F.shape[:-2])), *F.shape[-2:])
#         ix = ix.reshape(*([1] * len(F.shape[:-2])), *F.shape[-2:])
#         # a = torch.rand([6, *F.shape[:-2], 1, 1], device=img_device) * 2 - 1
#         a = np.random.randn(6, *F.shape[:-2], 1, 1)
#         # L = a[0] * ix + a[1] * iy + a[2] * ix * iy + a[3] * ix ** 2 + a[4] * iy ** 2 + a[5]
#         # L = (a[0] * ix + a[1] * iy + a[2]) * (a[3] * ix + a[4] * iy + a[5])
#         L = np.power(a[0] * ix + a[1] * iy + a[2], 2) * a[3] + np.power(
#             -a[1] * ix + a[0] * iy + a[4], 2) * a[5]
#         # L = a[3] * ix ** 2 + a[4] * iy ** 2 + a[5]
#         s_range = np.random.rand(2, *F.shape[:-2], 1, 1) * directionality

#         F *= np_normalize(L, low=1 - s_range[0], high=1 + s_range[1])
#     # contrast normalization
#     v_range = (
#         (np.random.rand(2, *F.shape[:-2], 1, 1) - 0.5) * 2 * contrast
#     )
#     F = np_normalize(F, 0 + v_range[0], 255 + v_range[1])
#     # gaussian noise
#     if noise_sigma > 0:
#         F += np.random.randn(*F.shape) * noise_sigma
#     # convert to unsigned char
#     return F.clip( 0, 255 )

# def gallery(array, ncols=3):
#     nrows = np.math.ceil(len(array)/float(ncols))
#     cell_w = array.shape[2]
#     cell_h = array.shape[1]
#     channels = array.shape[3]
#     result = np.zeros((cell_h*nrows, cell_w*ncols, channels), dtype=array.dtype)
#     for i in range(0, nrows):
#         for j in range(0, ncols):
#             result[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = array[i*ncols+j]
#     return result

# seq = iaa.Sequential([
#     iaa.LinearContrast((0.75, 1.5)),
#     iaa.Multiply((0.8, 1.2), per_channel=0.2),
#     iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add((0, 50))))
# ])


# def apply_lut(image, table):
#     """Map an input image to a new one using a lookup table.
#     Added in 0.4.0.
#     **Supported dtypes**:
#         See :func:`~imgaug.imgaug.apply_lut_`.
#     Parameters
#     ----------
#     image : ndarray
#         See :func:`~imgaug.imgaug.apply_lut_`.
#     table : ndarray or list of ndarray
#         See :func:`~imgaug.imgaug.apply_lut_`.
#     Returns
#     -------
#     ndarray
#         Image after mapping via lookup table.
#     """
#     return apply_lut_(np.copy(image), table)


# def _normalize_cv2_input_arr_(arr):
#     flags = arr.flags
#     if not flags["OWNDATA"]:
#         arr = np.copy(arr)
#         flags = arr.flags
#     if not flags["C_CONTIGUOUS"]:
#         arr = np.ascontiguousarray(arr)
#     return arr


# def apply_lut_(image, table):
#     """Map an input image in-place to a new one using a lookup table.
#     Added in 0.4.0.
#     **Supported dtypes**:
#         * ``uint8``: yes; fully tested
#         * ``uint16``: no
#         * ``uint32``: no
#         * ``uint64``: no
#         * ``int8``: no
#         * ``int16``: no
#         * ``int32``: no
#         * ``int64``: no
#         * ``float16``: no
#         * ``float32``: no
#         * ``float64``: no
#         * ``float128``: no
#         * ``bool``: no
#     Parameters
#     ----------
#     image : ndarray
#         Image of dtype ``uint8`` and shape ``(H,W)`` or ``(H,W,C)``.
#     table : ndarray or list of ndarray
#         Table of dtype ``uint8`` containing the mapping from old to new
#         values. Either a ``list`` of ``C`` ``(256,)`` arrays or a single
#         array of shape ``(256,)`` or ``(256, C)`` or ``(1, 256, C)``.
#         In case of ``(256,)`` the same table is used for all channels,
#         otherwise a channelwise table is used and ``C`` is expected to match
#         the number of channels.
#     Returns
#     -------
#     ndarray
#         Image after mapping via lookup table.
#         This *might* be the same array instance as provided via `image`.
#     """

#     image_shape_orig = image.shape
#     nb_channels = 1 if len(image_shape_orig) == 2 else image_shape_orig[-1]

#     if 0 in image_shape_orig:
#         return image

#     image = _normalize_cv2_input_arr_(image)

#     # [(256,), (256,), ...] => (256, C)
#     if isinstance(table, list):
#         assert len(table) == nb_channels, (
#             "Expected to get %d tables (one per channel), got %d instead." % (
#                 nb_channels, len(table)))
#         table = np.stack(table, axis=-1)

#     # (256, C) => (1, 256, C)
#     if table.shape == (256, nb_channels):
#         table = table[np.newaxis, :, :]

#     assert table.shape == (256,) or table.shape == (1, 256, nb_channels), (
#         "Expected 'table' to be any of the following: "
#         "A list of C (256,) arrays, an array of shape (256,), an array of "
#         "shape (256, C), an array of shape (1, 256, C). Transformed 'table' "
#         "up to shape %s for image with shape %s (C=%d)." % (
#             table.shape, image_shape_orig, nb_channels))

#     if nb_channels > 512:
#         if table.shape == (256,):
#             table = np.tile(table[np.newaxis, :, np.newaxis],
#                             (1, 1, nb_channels))

#         subluts = []
#         for group_idx in np.arange(int(np.ceil(nb_channels / 512))):
#             c_start = group_idx * 512
#             c_end = c_start + 512
#             subluts.append(apply_lut_(image[:, :, c_start:c_end],
#                                       table[:, :, c_start:c_end]))

#         return np.concatenate(subluts, axis=2)

#     assert image.dtype == np.dtype("uint8"), (
#         "Expected uint8 image, got dtype %s." % (image.dtype.name,))

#     image = cv2.LUT(image, table, dst=image)
#     return image



# def adjust_contrast_linear(arr, alpha):
#     """Adjust contrast by scaling each pixel to ``127 + alpha*(v-127)``.
#     **Supported dtypes**:
#         * ``uint8``: yes; fully tested (1) (2)
#         * ``uint16``: yes; tested (2)
#         * ``uint32``: yes; tested (2)
#         * ``uint64``: no (3)
#         * ``int8``: yes; tested (2)
#         * ``int16``: yes; tested (2)
#         * ``int32``: yes; tested (2)
#         * ``int64``: no (2)
#         * ``float16``: yes; tested (2)
#         * ``float32``: yes; tested (2)
#         * ``float64``: yes; tested (2)
#         * ``float128``: no (2)
#         * ``bool``: no (4)
#         - (1) Handled by ``cv2``. Other dtypes are handled by raw ``numpy``.
#         - (2) Only tested for reasonable alphas with up to a value of
#               around ``100``.
#         - (3) Conversion to ``float64`` is done during augmentation, hence
#               ``uint64``, ``int64``, and ``float128`` support cannot be
#               guaranteed.
#         - (4) Does not make sense for contrast adjustments.
#     Parameters
#     ----------
#     arr : numpy.ndarray
#         Array for which to adjust the contrast. Dtype ``uint8`` is fastest.
#     alpha : number
#         Multiplier to linearly pronounce (``>1.0``), dampen (``0.0`` to
#         ``1.0``) or invert (``<0.0``) the difference between each pixel value
#         and the dtype's center value, e.g. ``127`` for ``uint8``.
#     Returns
#     -------
#     numpy.ndarray
#         Array with adjusted contrast.
#     """

#     # int8 is also possible according to docs
#     # but here it seemed like `d` was 0 for CV_8S, causing that to fail
#     iinfo = np.iinfo(arr.dtype)

#     min_value, center_value, max_value = iinfo.min, iinfo.min + 0.5 * iinfo.max, iinfo.max
#     # TODO get rid of this int(...)
#     center_value = int(center_value)

#     value_range = np.arange(0, 256, dtype=np.float32)

#     # 127 + alpha*(I_ij-127)
#     # using np.float32(.) here still works when the input is a numpy array
#     # of size 1
#     alpha = np.float32(alpha)
#     table = center_value + alpha * (value_range - center_value)
#     table = np.clip(table, min_value, max_value).astype(arr.dtype)
#     arr_aug = apply_lut(arr, table)
#     return arr_aug

# def multiply(image, multiplier):
#     nb_channels = 1 if image.ndim == 2 else image.shape[-1]
#     multiplier = np.float32(multiplier)
#     value_range = np.arange(0, 256, dtype=np.float32)

#     value_range = value_range * multiplier
#     value_range = np.clip(value_range, 0, 255).astype(image.dtype)
#     return apply_lut_(image, value_range)
# print ('1111')
# img = cv2.imread('/data/home/uss00022/lelechen/data/Facescape/textured_meshes/1/models_reg/10_dimpler.jpg')
# print ('1111')
# img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
# print (img.dtype)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# imgs = np.zeros((1, img.shape[0], img.shape[1], 3), dtype =np.uint8)

# print ('1111')

# for i in range(1):
#     imgs[i] = img
# print ('1111')
# print (imgs.dtype)

# t = time.time()
# images_aug = []
# print ('1111')
# for i in range(64):
#     im = adjust_contrast_linear(imgs[0], random.uniform(0.75, 1.5))
#     im = multiply(im, random.uniform(0.8, 1.2) )
#     # 
#     im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#     images_aug.append( im)
#     # images_aug.append( cv2.cvtColor(seq(images=imgs)[0], cv2.COLOR_RGB2BGR) )
# images_aug = np.asarray(images_aug)
# print ('1111')

# print (time.time() - t)
# img_grid = gallery(images_aug, 8)
# print (time.time() - t)

# cv2.imwrite('./gg.png', img_grid )





    