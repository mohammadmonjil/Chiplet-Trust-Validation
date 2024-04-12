import cv2
import cv2
import numpy as np
from scipy.optimize import minimize_scalar

def fourier_descriptors(contour_complex, num_coefficients = None):
    # Convert the contour points to complex numbers
    #contour_complex = np.empty(len(contour), dtype=complex)
    #contour_complex.real = contour[:, 0]
    #contour_complex.imag = contour[:, 1]

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

def make_start_point_invariant(G):
    phi = get_start_point_phase(G)
    Ga = shift_start_point_phase(G, phi)
    Gb = shift_start_point_phase(G, phi + np.pi)
    return Ga, Gb
    
def get_start_point_phase(G):
    mp = (len(G) - 1) // 2
    def fp(phi):
        sum_val = 0
        for m in range(1, mp + 1):
            Gm = G[(-m) % len(G)] * np.exp(-1j * m * phi)
            Gp = G[(m) % len(G)] * np.exp(1j * m * phi)
            sum_val += Gp.real * Gm.imag - Gp.imag * Gm.real
        return sum_val

    res = minimize_scalar(lambda phi: -fp(phi), bounds=(0, np.pi), method='bounded')
    return res.x

def shift_start_point_phase(G, phi):
    Gnew = G.copy()
    mp = (len(G) - 1) // 2
    
    for m in range(-mp, mp + 1):
        if m != 0:
            k = (m) % len(G)
            Gnew[k] *= np.exp(1j * m * phi)
    return Gnew

def complex_l2(f1,f2):
    a = f1 -f2
    return np.sqrt(np.sum((a.real)**2 + (a.imag)**2))

def complex_l2_sqrd(f1,f2):
    a = f1 -f2
    return np.sum((a.real)**2 + (a.imag)**2)