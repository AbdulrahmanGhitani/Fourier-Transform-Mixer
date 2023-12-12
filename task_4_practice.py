# from matplotlib.image import imread
# import matplotlib.pyplot as plt
# import numpy as np
""""
input_image = imread("images.jpeg")
# (rgb) color encoding
#slicing -->  [row,col,index]
r , g , b = input_image[:,:,0] , input_image[:,:,1] , input_image[:,:,2]

gamma = 1.04

# The values of r_const, g_const, and b_const are
# The luminance coefficients represent
# the sensitivity of the human eye to different wavelengths of light

r_const , g_const , b_const = 0.2126 , 0.7152, 0.0722
grayscale_image = r_const * r ** gamma + g_const * r ** gamma + b_const * r ** gamma
fig = plt.figure()
img_1 , img_2 = fig.add_subplot(121) , fig.add_subplot(122)
img_1.imshow(input_image)
img_2.imshow(grayscale_image, cmap=plt.cm.get_cmap('gray'))

fig.show()
plt.show()

"""
"""
import cv2

image = cv2.imread('images.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_resized = cv2.resize(image,(150,150))
# define the alpha and beta
alpha = 1.5 # Contrast control
beta = 10 # Brightness control

# call convertScaleAbs function
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# display the output image
cv2.imshow('adjusted', adjusted)
cv2.imshow('Original image', image)
cv2.imshow('Gray image', gray)
cv2.imshow('Resized image', image_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()


///////////////////////////////////
import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('images.jpeg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img = cv2.imread('messi5.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.imshow('Magnitude Spectrum', magnitude_spectrum)

///////////////////////////////
////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('images.jpeg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img = cv2.imread('messi5.jpg',0)
f = np.fft.fft2(img)

fshift = np.fft.fftshift(f)



# Separate real and imaginary parts
f_real = np.real(fshift)
f_imag = np.imag(fshift)

# Compute magnitude and phase
f_mag = np.abs(fshift)
f_phase = np.angle(fshift)

magnitude_spectrum = 20*np.log(np.abs(fshift))
phase_spectrum = np.angle(fshift)
real_spectrum = np.real(fshift)
imaginary_spectrum = np.imag(fshift)





plt.subplot(421),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(422),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(423),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(424),plt.imshow(phase_spectrum, cmap = 'gray')
plt.title('phase_spectrum '), plt.xticks([]), plt.yticks([])

plt.subplot(425),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(426),plt.imshow(real_spectrum, cmap = 'gray')
plt.title('real_spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(427),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(428),plt.imshow(imaginary_spectrum, cmap = 'gray')
plt.title('imaginary_spectrum'), plt.xticks([]), plt.yticks([])

plt.show()


# ////////////////////////-------------------------------------------------------------




import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image in grayscale
image = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

# Perform 2D Fourier Transform
f = np.fft.fft2(image)

# Shift zero frequency components to the center
fshift = np.fft.fftshift(f)

# Calculate magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Calculate phase spectrum
phase_spectrum = np.angle(fshift)

# Calculate real part spectrum
real_spectrum = 20 * np.log(np.real(fshift))

# Calculate imaginary part spectrum
imag_spectrum = np.imag(fshift)

# Display the input image
plt.subplot(231), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

# Display the magnitude spectrum
plt.subplot(232), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

# Display the phase spectrum
plt.subplot(233), plt.imshow(phase_spectrum, cmap='gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])

# Display the real part spectrum
plt.subplot(234), plt.imshow(real_spectrum, cmap='gray')
plt.title('Real Part Spectrum'), plt.xticks([]), plt.yticks([])

# Display the imaginary part spectrum
plt.subplot(235), plt.imshow(imag_spectrum, cmap='gray')
plt.title('Imaginary Part Spectrum'), plt.xticks([]), plt.yticks([])

# Show all the plots
plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('images.jpeg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
f = np.fft.fft2(img)

fshift = np.fft.fftshift(f)



# Separate real and imaginary parts
f_real = np.real(fshift)
f_imag = np.imag(fshift)

# Compute magnitude and phase
f_mag = np.abs(fshift)
f_phase = np.angle(fshift)

magnitude_spectrum = 20*np.log(np.abs(fshift))
phase_spectrum = np.angle(fshift)
real_spectrum = 20*np.log(np.real(fshift))
imaginary_spectrum = np.imag(fshift)





plt.subplot(421),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(422),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(423),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(424),plt.imshow(phase_spectrum, cmap = 'gray')
plt.title('phase_spectrum '), plt.xticks([]), plt.yticks([])

plt.subplot(425),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(426),plt.imshow(real_spectrum, cmap = 'gray')
plt.title('real_spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(427),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(428),plt.imshow(imaginary_spectrum, cmap = 'gray')
plt.title('imaginary_spectrum'), plt.xticks([]), plt.yticks([])

plt.show()






//////////

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image in grayscale
image = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

# Perform 2D Fourier Transform
f = np.fft.fft2(image)

# Shift zero frequency components to the center
fshift = np.fft.fftshift(f)

# Calculate magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Calculate phase spectrum
phase_spectrum = np.angle(fshift)

# Calculate real part spectrum
real_spectrum = 20 * np.log(np.real(fshift))

# Calculate imaginary part spectrum
imag_spectrum = np.imag(fshift)

# Display the input image
plt.subplot(231), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

# Display the magnitude spectrum
plt.subplot(232), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

# Display the phase spectrum
plt.subplot(233), plt.imshow(phase_spectrum, cmap='gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])

# Display the real part spectrum
plt.subplot(234), plt.imshow(real_spectrum, cmap='gray')
plt.title('Real Part Spectrum'), plt.xticks([]), plt.yticks([])

# Display the imaginary part spectrum
plt.subplot(235), plt.imshow(imag_spectrum, cmap='gray')
plt.title('Imaginary Part Spectrum'), plt.xticks([]), plt.yticks([])

# Show all the plots
plt.show()
"""







#  final code of real + imaginary + phase + magnitude
# import cv2
# import numpy as np
#
# # Read the image in grayscale
# image = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)
#
# # Perform 2D Fourier Transform
# f = np.fft.fft2(image)
#
# # Shift zero frequency components to the center
# fshift = np.fft.fftshift(f)
#
# # Calculate magnitude spectrum
# magnitude_spectrum = 20 * np.log(np.abs(fshift))
#
# # Calculate phase spectrum
# phase_spectrum = np.angle(fshift)
#
# # Calculate real part spectrum
# real_spectrum = 20 * np.log(np.real(fshift))
#
# # Calculate imaginary part spectrum
# imag_spectrum = np.imag(fshift)
#
# # Display the input image
# cv2.imshow('Input Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Display the magnitude spectrum
# cv2.imshow('Magnitude Spectrum', magnitude_spectrum.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Display the phase spectrum
# cv2.imshow('Phase Spectrum', phase_spectrum.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Display the real part spectrum
# cv2.imshow('Real Part Spectrum', real_spectrum.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Display the imaginary part spectrum
# cv2.imshow('Imaginary Part Spectrum', imag_spectrum.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
import cv2
import numpy as np

# Read input images
image1b = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)
image2b = cv2.imread('download.jpeg', cv2.IMREAD_GRAYSCALE)
image3b = cv2.imread('download (1).jpeg', cv2.IMREAD_GRAYSCALE)
image4b = cv2.imread('images (1).jpeg', cv2.IMREAD_GRAYSCALE)

# making the images the same size
image1 = cv2.resize(image1b,(500,500))
image2 = cv2.resize(image2b,(500,500))
# image3 = cv2.resize(image3b,(500,500))
# image4 = cv2.resize(image4b,(500,500))

# Define custom weights for magnitude, phase, real, and imaginary components
#
# weights_magnitude = [1.0, 0.8, 0.5, 0.3]
# weights_phase = [0.0, 0.0, 0.0, 0.0]
# weights_real = [0.8, 0.5, 0.2, 1.0]
# weights_imaginary = [0.0, 0.0, 0.0, 0.0]
#
# # Create a list of input images
# input_images = [image1]
#
# # Initialize an empty complex array for the output Fourier transform
# output_fft = np.zeros_like(np.fft.fft2(input_images[0]), dtype=np.complex128)
#
# # Compute the weighted sum of Fourier transforms
# for i in range(len(input_images)):
#     # Compute Fourier transform of the current image
#     fft = np.fft.fft2(input_images[i])
#
#     # Apply custom weights to magnitude, phase, real, and imaginary components
#     weighted_fft = (
#         weights_magnitude[i] * np.abs(fft) * np.exp(1j * weights_phase[i]) +
#         weights_real[i] * np.real(fft) +
#         1j * weights_imaginary[i] * np.imag(fft)
#     )
#     # Apply custom weight only to the phase component
#
#     # here we modify the phase
#     # weighted_fft = (
#     #         np.abs(fft) * np.exp(1j * weights_phase[i]) +
#     #         1j * weights_imaginary[i] * np.imag(fft)
#     # )
#
#
#     # Apply custom weight only to the magnitude component
#     # weighted_fft = (
#     #         weights_magnitude[i] * np.abs(fft) +
#     #         weights_real[i] * np.real(fft) +
#     #         1j * weights_imaginary[i] * np.imag(fft)
#     # )
#
#     # Accumulate the weighted Fourier transform
#     output_fft += weighted_fft
#
# # Compute the inverse Fourier transform of the weighted sum
# output_image = np.abs(np.fft.ifft2(output_fft)).astype(np.uint8)
#
# # Display the input images
# for i in range(len(input_images)):
#     cv2.imshow(f'Input Image {i+1}', input_images[i])
#
# # Display the output image
# cv2.imshow('Output Image', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#




fft4 = np.fft.fft2(image2)
mag_of_messi = np.abs(fft4)

fft9 = np.fft.fft2(image1)
phase_parrot = np.angle(fft9)

weighted_fft = (
        mag_of_messi * np.exp(1j * phase_parrot) +

        1j * np.imag(fft9)
)

output_image = np.abs(np.fft.ifft2(weighted_fft)).astype(np.uint8)
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



"""



# /////////////////////////////////////////////////////////////////////////////////
import cv2
import numpy as np
    # # iterate on all photos to resize the wrt the smallest one
    #
    # # Assuming img_list is a list containing your image paths
    # img_paths = ['images.jpeg', 'download.jpeg', 'download (1).jpeg', 'images (1).jpeg']
    #
    # # Read images
    # img_list = [cv2.imread(img_path) for img_path in img_paths]
    #
    # # Get the size of the smallest image
    # smallest_size = min(img.shape[:2] for img in img_list)
    #
    # # Resize all images to the size of the smallest image
    # resized_images = [cv2.resize(img, (smallest_size[1], smallest_size[0])) for img in img_list]
    #
    # # Display or save the resized images as needed
    # for i, resized_img in enumerate(resized_images):
    #     cv2.imshow(f"Resized Image {i+1}", resized_img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # # # start here -resize -import - gray scale - brightness and contrast
    # import cv2
    # import numpy as np
    #
    # image = cv2.imread('images.jpeg')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # image_resized = cv2.resize(image,(150,150))
    # # define the alpha and beta
    #
    # alpha = 1.5 # Contrast control
    # beta = 1.2 # Brightness control
    #
    # # call convertScaleAbs function
    # adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    #
    #
    # # display the output image
    # cv2.imshow('adjusted', adjusted)
    # cv2.imshow('Original image', image)
    # cv2.imshow('Gray image', gray)
    # cv2.imshow('Resized image', image_resized)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
#  final code of real + imaginary + phase + magnitude


    # # Read the image in grayscale
    # image = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)
    #
    # # Perform 2D Fourier Transform
    # f = np.fft.fft2(image)
    #
    # # Shift zero frequency components to the center
    # fshift = np.fft.fftshift(f)
    #
    # # Calculate magnitude spectrum
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))
    #
    # # Calculate phase spectrum
    # phase_spectrum = np.angle(fshift)
    #
    # # Calculate real part spectrum
    # real_spectrum = 20 * np.log(np.real(fshift))
    #
    # # Calculate imaginary part spectrum
    # imag_spectrum = np.imag(fshift)
    #
    # # Display the input image
    # cv2.imshow('Input Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # Display the magnitude spectrum
    # cv2.imshow('Magnitude Spectrum', magnitude_spectrum.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # Display the phase spectrum
    # cv2.imshow('Phase Spectrum', phase_spectrum.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # Display the real part spectrum
    # cv2.imshow('Real Part Spectrum', real_spectrum.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # Display the imaginary part spectrum
    # cv2.imshow('Imaginary Part Spectrum', imag_spectrum.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# --------------------------------------------------------------------------------------
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# custamize the weights of mag and phase
import cv2
import numpy as np

# Read input images
image1b = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)
image2b = cv2.imread('download.jpeg', cv2.IMREAD_GRAYSCALE)
image3b = cv2.imread('download (1).jpeg', cv2.IMREAD_GRAYSCALE)
image4b = cv2.imread('images (1).jpeg', cv2.IMREAD_GRAYSCALE)

# making the images the same size
image1 = cv2.resize(image1b,(500,500))
image2 = cv2.resize(image2b,(500,500))
image3 = cv2.resize(image3b,(500,500))
image4 = cv2.resize(image4b,(500,500))

# Define custom weights for magnitude, phase, real, and imaginary components

    # weights_magnitude = [1.0, 0.0, 0.0, 0.0]
    # weights_phase = [1.0, 0.0, 0.0, 0.0]
    # # weights_real = [0.8, 0.5, 0.2, 1.0]
    # # weights_imaginary = [0.0, 0.0, 0.0, 0.0]
    #
    # # Create a list of input images
    # input_images = [image1,image2,image3,image4]
    #
    # # Initialize an empty complex array for the output Fourier transform
    # output_fft = np.zeros_like(np.fft.fft2(input_images[0]), dtype=np.complex128)
    #
    # # Compute the weighted sum of Fourier transforms
    # for i in range(len(input_images)):
    #     # Compute Fourier transform of the current image
    #     fft = np.fft.fft2(input_images[i])
    #
    #     # Apply custom weights to magnitude, phase, real, and imaginary components
    #     weighted_fft = (
    #         weights_magnitude[i] * np.abs(fft) * np.exp(1j * weights_phase[i])
    #     )
    #
    #     output_fft += weighted_fft
    #
    # # Compute the inverse Fourier transform of the weighted sum
    # output_image = np.abs(np.fft.ifft2(output_fft)).astype(np.uint8)
    #
    # # Display the input images
    # for i in range(len(input_images)):
    #     cv2.imshow(f'Input Image {i+1}', input_images[i])
    #
    # # Display the output image
    # cv2.imshow('Output Image', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()








# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# messi code for asg
fft_messi = np.fft.fft2(image2)
mag_of_messi = np.abs(fft_messi)
phase_messi = np.angle(fft_messi)


fft_parrot = np.fft.fft2(image1)
mag_parrot = np.abs(fft_parrot)
phase_parrot = np.angle(fft_parrot)

fft_natural = np.fft.fft2(image3)
mag_of_natural = np.abs(fft_natural)

weighted_fft = (np.exp(1j * phase_messi)* mag_parrot +
    1j * np.imag(fft_messi)
)

output_image = np.abs(np.fft.ifft2(weighted_fft)).astype(np.uint8)
cv2.imshow('Output Image kkkk', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
##############################################################################

# FFT and magnitude for each image
fft_messi = np.fft.fft2(image2)
mag_of_messi = np.abs(fft_messi)
phase_messi = np.angle(fft_messi)

fft_parrot = np.fft.fft2(image1)
mag_parrot = np.abs(fft_parrot)
phase_parrot = np.angle(fft_parrot)

fft_natural = np.fft.fft2(image3)
mag_of_natural = np.abs(fft_natural)
phase_natural = np.angle(fft_natural)

# Define weights for magnitude and phase of each image
weight_messi_mag = 1
weight_messi_phase = 1.0

weight_parrot_mag = 0.0
weight_parrot_phase = 1.0

weight_natural_mag = 0.0
weight_natural_phase = 0.0

# Weighted FFT summation
weighted_fft = (
    weight_messi_mag * np.exp(1j * weight_messi_phase * phase_messi) * mag_parrot +
    weight_parrot_mag * np.exp(1j * weight_parrot_phase * phase_parrot) * mag_of_messi +
    weight_natural_mag * np.exp(1j * weight_natural_phase * phase_natural) * mag_of_natural
)

# Inverse FFT to get the output image
output_image = np.abs(np.fft.ifft2(weighted_fft)).astype(np.uint8)

# Display the output image
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
