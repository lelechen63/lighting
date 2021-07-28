import numpy as np 
import cv2 
import torch 


def th_normalize(x, low=0, high=1):
    x_max = x.view(*x.shape[:-2], -1).max(dim=-1)[0][..., None, None]
    x_min = x.view(*x.shape[:-2], -1).min(dim=-1)[0][..., None, None]
    scale = (high - low) / (x_max - x_min)

    print (x.shape, x_min.shape)
    return (x - x_min) * scale + low


def np_normalize(x, low= 0, high = 1):
    # input x: size 256x256x3, output: 245x256x3
    if x.shape[0] != 3:
        x = x.transpose(2, 0, 1)
    print (x.shape)
    x_max = x.reshape(*x.shape[:-2], -1)
    print (x_max.shape)
    x_max =x_max.max(axis=-1)[0]
    print (x_max.shape)

    print ('!!!!!!!!!!!!!!!!!!!')
    x_max = x_max[..., None, None]

    print (x_max)
    x_min = x.view(*x.shape[:-2], -1)
    print (x_min.shape, '++++++++')
    x_min = x_min.min(axis=-1)[0]
    print (x_min, '!!')
    x_min = x_min[..., None, None]
    print (x_min, '++=')
    scale = (high - low) / (x_max - x_min)


    x =  (x - x_min) * scale + low

    if x.shape[0] == 3:
        x = x.transpose(1,2,0)
    return x

def th_augument_tex_color( img, smoothness=100, directionality=0.9, noise_sigma=2.0, contrast=20.0 ):
    img_device = img.device
    # img : 3,256,256 (0-255)
    # intensity histogram augmentation
    if smoothness > 0:
        x = torch.arange(256, device=img_device).float() / 255 - 0.5
        smoothness = int(smoothness)
        u = torch.rand((smoothness, 3), device=img_device)
        a = u[:, 0:1] - 0.5
        b = u[:, 1:2] * 99 + 1
        c = u[:, 2:3] * 10
        x_sum = (c / (1.0 + torch.exp(-b * (x[None, :] - a)))).sum(0)
        F = x_sum[img.long()]
    else:
        F = img.float()
    
    # spatially varying intensity multiplier (second degree polynomial)
    if directionality > 0:
        h, w = F.shape[-2:]
        iy, ix = torch.meshgrid(
            (torch.arange(h, device=img_device).float() / (h // 2)) - 1,
            (torch.arange(w, device=img_device).float() / (w // 2)) - 1,
        )
        iy = iy.view(*([1] * len(F.shape[:-2])), *F.shape[-2:])
        ix = ix.view(*([1] * len(F.shape[:-2])), *F.shape[-2:])
        # a = torch.rand([6, *F.shape[:-2], 1, 1], device=img_device) * 2 - 1
        a = torch.randn([6, *F.shape[:-2], 1, 1], device=img_device)
        # L = a[0] * ix + a[1] * iy + a[2] * ix * iy + a[3] * ix ** 2 + a[4] * iy ** 2 + a[5]
        # L = (a[0] * ix + a[1] * iy + a[2]) * (a[3] * ix + a[4] * iy + a[5])
        L = (a[0] * ix + a[1] * iy + a[2]).pow(2) * a[3] + (
            -a[1] * ix + a[0] * iy + a[4]
        ).pow(2) * a[5]
        # L = a[3] * ix ** 2 + a[4] * iy ** 2 + a[5]
        s_range = torch.rand([2, *F.shape[:-2], 1, 1], device=img_device) * directionality
        F *= th_normalize(L, low=1 - s_range[0], high=1 + s_range[1])
    # contrast normalization
    v_range = (
        (torch.rand([2, *F.shape[:-2], 1, 1], device=img_device) - 0.5) * 2 * contrast
    )
    F = th_normalize(F, 0 + v_range[0], 255 + v_range[1])
    # gaussian noise
    if noise_sigma > 0:
        F += torch.randn(F.shape, device=img_device) * noise_sigma
    # convert to unsigned char
    return F.clamp( min=0.0, max=255.0 )

def np_augument_tex_color( img, smoothness=100, directionality=0.9, noise_sigma=2.0, contrast=20.0 ):
    # img_device = img.device
    # img : 3,256,256 (0-255)
    # intensity histogram augmentation
    if smoothness > 0:
        x = np.arange(256).astype(float) / 255 - 0.5
        smoothness = int(smoothness)
        u = np.random.rand(smoothness, 3)
        a = u[:, 0:1] - 0.5
        b = u[:, 1:2] * 99 + 1
        c = u[:, 2:3] * 10
        x_sum = (c / (1.0 + np.exp(-b * (x[None, :] - a)))).sum(0)
        F = x_sum[img]
    else:
        F = img.astype(float)
    
    # spatially varying intensity multiplier (second degree polynomial)
    if directionality > 0:
        h, w = F.shape[-2:]
        iy, ix = np.meshgrid(
            (np.arange(h).astype(float) / (h // 2)) - 1,
            (np.arange(w).astype(float) / (w // 2)) - 1,
        )
        iy = iy.reshape(*([1] * len(F.shape[:-2])), *F.shape[-2:])
        ix = ix.reshape(*([1] * len(F.shape[:-2])), *F.shape[-2:])
        # a = torch.rand([6, *F.shape[:-2], 1, 1], device=img_device) * 2 - 1
        a = np.random.randn(6, *F.shape[:-2], 1, 1)
        # L = a[0] * ix + a[1] * iy + a[2] * ix * iy + a[3] * ix ** 2 + a[4] * iy ** 2 + a[5]
        # L = (a[0] * ix + a[1] * iy + a[2]) * (a[3] * ix + a[4] * iy + a[5])
        L = np.power(a[0] * ix + a[1] * iy + a[2], 2) * a[3] + np.power(
            -a[1] * ix + a[0] * iy + a[4], 2) * a[5]
        # L = a[3] * ix ** 2 + a[4] * iy ** 2 + a[5]
        s_range = np.random.rand(2, *F.shape[:-2], 1, 1) * directionality

        print (L.shape)
        F *= np_normalize(L, low=1 - s_range[0], high=1 + s_range[1])
    # contrast normalization
    v_range = (
        (np.random.rand(2, *F.shape[:-2], 1, 1) - 0.5) * 2 * contrast
    )
    F = np_normalize(F, 0 + v_range[0], 255 + v_range[1])
    # gaussian noise
    if noise_sigma > 0:
        F += np.random.randn(F.shape) * noise_sigma
    # convert to unsigned char
    return F.clip( 0, 255 )


img = cv2.imread('/data/home/us000042/lelechen/data/Facescape/textured_meshes/1/models_reg/10_dimpler.jpg')
# cv2.imwrite('./gg.png', img)
img = img.transpose(2,0,1)
newimg = np_augument_tex_color(img).transpose(1,2,0)
out = np.concatenate(img, newimg,axis =0 )
cv2.imwrite('./gg.png', out )
