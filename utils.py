
import os, cv2
import numpy as np

def calc_dark_channel(img, window_size, color="bgr"):
    r"""
    Info:
        Calculate dark channel of given image.
    Args:
        img (Ndarray): img should have shape format (3, H, W) for RGB image or (H, W) for gray image.
        window_size (tuple | list | int): size of window for calculating dark channel.
        color (string): ["bgr"(default), "rgb", "gray"]
    Returns:
        dark_channel (Ndarray)
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    color = color.lower()
    if color not in ["bgr", "rgb", "gray"]:
        raise ValueError("Expect color is one of ['bgr', 'rgb', 'gray'], but got {}".format(color))
    if color in ["bgr", "rgb"] and img.shape[-1] != 3:
        raise ValueError("RGB image should have format (H, W, 3), but got shape {}".format(img.shape))
    if not isinstance(img, np.ndarray):
        raise TypeError("Expect input image is an instance of Ndarray or Tensor, but got {}".format(type(img)))
    window_size = cv2.getStructuringElement(cv2.MORPH_RECT, window_size)
    if color in ["bgr", "rgb"]:
        dark_channel = np.min(img, axis=-1)
        dark_channel = cv2.erode(dark_channel, kernel=window_size)
    else:
        dark_channel = cv2.erode(img, kernel=window_size)
    assert dark_channel.shape == img.shape[: 2]
    
    # Maximum filter for preventing morphological filtering.
    dark_channel = cv2.dilate(dark_channel, kernel=window_size)
    assert dark_channel.shape == img.shape[: 2]
    return dark_channel

def calc_airlight(img, dark_channel, color="bgr"):
    r"""
    Info:
        Calculate airlight.
    Args:
        img (ndarray): [3, H, W] or [H, W]
        dark_channel (ndarray): same size with img.
        color (string): takes value from ["bgr"(default), "rgb", "gray"]
    Returns:
        airlight (ndarray): a vector (RGB) or scalar (GRAY)
    """
    color = color.lower()
    if color not in ["bgr", "rgb", "gray"]:
        raise ValueError("Expect color is one of ['rgb', 'gray'], but got {}".format(color))
    if color in ["bgr", "rgb"] and img.shape[-1] != 3:
        raise ValueError("RGB image should have format (3, H, W), but got shape {}".format(img.shape))
    if not isinstance(img, np.ndarray):
        raise TypeError("Expect input image is an instance of Ndarray or Tensor, but got {}".format(type(img)))
    h, w = dark_channel.shape
    flattened_dark_channel = dark_channel.reshape(-1)
    top_max_indices = np.argsort(flattened_dark_channel)[-max(dark_channel.size//1000, 1): ]
    if color in ["bgr", "rgb"]:
        airlight = np.zeros((1, 1, 3))
        for chann in range(3):
            top_max_intensity = np.take_along_axis(img[..., chann].reshape(-1), top_max_indices, axis=None)
            top_max_intensity_indices = np.argsort(top_max_intensity)[-max(top_max_intensity.size//100, 1): ]
            airlight[0, 0, chann] = np.mean(top_max_intensity[top_max_intensity_indices])
    elif color == "gray":
        top_max_intensity = np.take_along_axis(img.reshape(-1), top_max_indices, axis=None)
        top_max_intensity_indices = np.argsort(top_max_intensity)[-max(top_max_intensity.size//100, 1): ]
        airlight = np.mean(top_max_intensity[top_max_intensity_indices])
    return airlight

def calc_and_refine_transmission(img, gray_image, airlight, window_size=15, omega=0.95, radius=60, eps=1E-4):
    r"""
    Info:
        Calculate and refine transmission map.
    Args:
        img (ndarray): [H, W, 3] or [H, W]
        omega (float): preserve some haze by controlling transmission.
        radius (tuple | list | int): radius for guided filter.
        eps (float): for guided filter.
    """
    # FIXME Failed to use multi-channel guided filter, try gray image.
    if isinstance(radius, int):
        radius = (radius, radius)
    trans = 1 - omega * calc_dark_channel(img / airlight, window_size)

    # Refine transmission by using guided filter.
    mean_i = cv2.boxFilter(gray_image, cv2.CV_64F, ksize=radius)
    mean_d = cv2.boxFilter(trans, cv2.CV_64F, ksize=radius)
    cov_id = cv2.boxFilter(gray_image*trans, cv2.CV_64F, ksize=radius) - mean_i * mean_d
    var_i = cv2.boxFilter(gray_image*gray_image, cv2.CV_64F, ksize=radius) - mean_i ** 2
    a = cov_id / (var_i + eps)
    b = mean_d - a*mean_i
    a = cv2.boxFilter(a, cv2.CV_64F, ksize=radius)
    b = cv2.boxFilter(b, cv2.CV_64F, ksize=radius)

    trans = gray_image * a + b
    return trans

def dcp_dehazing(hazy_image, window_size=15, omega=0.95, thre=0.1, radius=60, eps=1E-4, color="bgr"):
    r"""
    Info:
        Dehazing using dark channel prior.
    Args:
        hazy_image (Tensor | Ndarray): shape requirement: [H, W, 3] for RGB image or [H, W] for gray image.
        window_size (tuple | list): windows size for calculating dark channel.
        omega (float): to preserve some haze, controlling transmission.
        thre (float): controlling airlight.
        radius (int): radius for guided filter.
        eps (float): for guided filter.
        color (string): ["bgr" (default), "rgb", "gray"]
    """
    # color = "rgb" if len(hazy_image.shape) == 3 else "gray"
    I = hazy_image / 255.0
    if color == "bgr":
        gray_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2GRAY)
    elif color == "rgb":
        gray_image = cv2.cvtColor(hazy_image, cv2.COLOR_RGB2GRAY)
    # normalized_hazy_dark_channel = calc_dark_channel(I/airlight, window_size)
    hazy_image = hazy_image.astype(np.float)
    dark_channel = calc_dark_channel(hazy_image/255, window_size)
    airlight = calc_airlight(hazy_image/255, dark_channel, color=color)
    transmission = calc_and_refine_transmission(hazy_image/255, gray_image/255, airlight, window_size, omega, radius, eps)
    haze_free_image = (hazy_image / 255 - airlight) / np.maximum(transmission[:, :, np.newaxis], thre) + airlight
    haze_free_image = (haze_free_image*255)
    return haze_free_image

def dcp_dehazing_from_dir(src_dir, trg_dir):
    fns = os.listdir(src_dir)
    for idx, fn in enumerate(fns):
        if fn.split(".")[-1] not in ["png", "jpg", "PNG", "JPG"]:
            continue
        src = cv2.imread(os.path.join(src_dir, fn), -1)
        trg = dcp_dehazing(src)
        cv2.imwrite(os.path.join(trg_dir, fn), trg)

