import cv2
import os
import copy
import math
import cv2
import numpy as np

def fourier_descriptors(contour, num_coefficients = None):
    # Convert the contour points to complex numbers
    contour_complex = np.empty(len(contour), dtype=complex)
    contour_complex.real = contour[:, 0]
    contour_complex.imag = contour[:, 1]

    # Apply Fourier Transform
    descriptors = np.fft.fft(contour_complex)

    if num_coefficients is not None:
        descriptors_truncated = np.empty(num_coefficients, dtype=complex)
        for m in range(0,num_coefficients):
            if (m <= np.floor(num_coefficients/2)):
                descriptors_truncated[m] = descriptors[m]
            else:
                descriptors_truncated[m] = descriptors[len(descriptors) - num_coefficients + m]
            
        return descriptors_truncated
    else:
        return descriptors

def reduce_fourier_length(descriptors, num_coefficients):
    descriptors_truncated = np.empty(num_coefficients, dtype=complex)
    
    for m in range(0,num_coefficients):
        if (m <= np.floor(num_coefficients/2)):
            descriptors_truncated[m] = descriptors[m]
        else:
            descriptors_truncated[m] = descriptors[len(descriptors) - num_coefficients + m]
        
    return descriptors_truncated

def reconstruct_image(fourier_coeffs, original_shape):
    reconstructed_complex = np.fft.ifft(fourier_coeffs)
    reconstructed_contour = np.column_stack((np.real(reconstructed_complex), np.imag(reconstructed_complex)))

    if len(reconstructed_contour) > 0:
        return cv2.drawContours(np.zeros(original_shape), [reconstructed_contour.astype(int)], 0, (255), thickness=1)
    else:
        return np.zeros(original_shape)
