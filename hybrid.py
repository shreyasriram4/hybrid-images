import sys
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    img_dims = img.shape
    kernel_dims = kernel.shape
    pad_height = kernel_dims[0] // 2
    pad_width = kernel_dims[1] // 2
    
    # Check if the image is grayscale or RGB
    if len(img_dims) == 2:  # Grayscale image
        padded_img_dims = (img_dims[0] + 2*pad_height, img_dims[1] + 2*pad_width)
        padded_img = np.zeros(padded_img_dims, dtype=img.dtype)
        # Copy img into the center of padded_img
        padded_img[pad_height:pad_height+img_dims[0], pad_width:pad_width+img_dims[1]] = img
    else:  # RGB image
        padded_img_dims = (img_dims[0] + 2*pad_height, img_dims[1] + 2*pad_width, img_dims[2])
        padded_img = np.zeros(padded_img_dims, dtype=img.dtype)
        # Copy img into the center of padded_img
        padded_img[pad_height:pad_height+img_dims[0], pad_width:pad_width+img_dims[1], :] = img

    out = np.zeros_like(img)
    
    for out_row in range(img_dims[0]):
        for out_col in range(img_dims[1]):
            for channel in range(img_dims[2]) if len(img_dims) == 3 else [0]:
                curr_val = 0
                for kernel_row in range(kernel_dims[0]):
                    for kernel_col in range(kernel_dims[1]):
                        if len(img_dims) == 3:  # RGB image
                            curr_val += padded_img[out_row + kernel_row, out_col + kernel_col, channel] * kernel[kernel_row, kernel_col]
                        else:  # Grayscale image
                            curr_val += padded_img[out_row + kernel_row, out_col + kernel_col] * kernel[kernel_row, kernel_col]
                if len(img_dims) == 3:
                    out[out_row, out_col, channel] = curr_val
                else:
                    out[out_row, out_col] = curr_val
    return out
    
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel_dim = np.shape(kernel)
    new_kernel = np.zeros(kernel_dim)
    for row in range(kernel_dim[0]):
        for col in range(kernel_dim[1]):
            new_kernel[row, col] = kernel[kernel_dim[0]-1-row, kernel_dim[1]-1-col]
    final = cross_correlation_2d(img, new_kernel)
    return final

    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    kernel = np.zeros((height, width))
    center_row = height // 2
    center_col = width // 2
    
    for row in range(height):
        for col in range(width):
            dist = ((center_row - row) ** 2 + (center_col - col) ** 2)
            kernel[row, col] = np.exp(-dist / (2 * sigma ** 2))
    
    # Normalize the kernel to ensure the sum of all elements equals 1
    kernel /= (2 * np.pi * sigma ** 2)
    kernel /= kernel.sum()
    
    return kernel
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    out = convolve_2d(img, kernel)
    return out
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    low_pass_img = low_pass(img, sigma, size)
    return img - low_pass_img
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

