import numpy as np
from skimage.metrics import structural_similarity as cal_ssim


def mae(pred, true):
    return np.mean(np.abs(pred - true), axis=(0, 1)).sum()


def mse(pred, true):
    return np.mean((pred - true) ** 2, axis=(0, 1)).sum()


def psnr(pred, true):
    mse_val = np.mean((np.uint8(pred * 255) - np.uint8(true * 255)) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse_val)


def metric(pred, true, mean, std, return_ssim_psnr=False, clip_range=(0, 1)):
    pred = pred * std + mean
    true = true * std + mean
    mae_value = mae(pred, true)
    mse_value = mse(pred, true)

    if not return_ssim_psnr:
        return mse_value, mae_value

    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    ssim_value = 0.0
    psnr_value = 0.0
    for b in range(pred.shape[0]):
        for f in range(pred.shape[1]):
            ssim_value += cal_ssim(
                pred[b, f].swapaxes(0, 2),
                true[b, f].swapaxes(0, 2),
                channel_axis=-1,
                data_range=clip_range[1] - clip_range[0],
            )
            psnr_value += psnr(pred[b, f], true[b, f])

    factor = pred.shape[0] * pred.shape[1]
    ssim_value /= factor
    psnr_value /= factor
    return mse_value, mae_value, ssim_value, psnr_value
