import cv2
import numpy as np
import matplotlib.pyplot as plt

TODO = None

def match_histogram(image, c_histogram_ref, n_ref):
    """Modify the histogram of an image to match a reference histogram

    Parameters
    ----------
    image: numpy array of size N x M x C
        An input image of size N x M with C color channels
    c_histogram_ref: numpy array of size 256
        Cumulative histogram of the reference image
    n_ref: integer
        number of pixels in the reference image
    
    Returns
    -------
        new_image: a version of `image` modified to match `c_histogram_ref`
    """

    # The number of pixels in the target image
    n = image.shape[0] * image.shape[1]

    # The ratio that lets us compare values from the two histograms
    ratio = n / n_ref

    # --------------------------------------------------------------------------
    # TODO: Compute the non-cumulative histogram of `image`. (1 mark)
    # Hint: This is already being done somewhere else in this code.
    histogram_target, _ = np.histogram(image, bins=np.arange(256 + 1))
   
    # plt.bar(np.arange(256), histogram_im)
    # plt.show()

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # TODO: Compute the mapping F which will transform the pixels of `image` to
    # match the histogram of the reference image. (4 marks)

    F = np.zeros([256], dtype=np.uint8)
    F = np.full([256], 255, dtype=np.uint8)
    i = 0
    j = 0
    c = 0
    while i < 256 and j < 256:
        if c < ratio*c_histogram_ref[j]:
            c += histogram_target[i]
            F[i] = j
            i += 1
        else:
            j += 1

    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # TODO: Apply the mapping `F` to the input image. (2 marks)
    # print("image shape", image[0][2])
    new_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i][j] = F[image[i][j]]

    # --------------------------------------------------------------------------

    return new_image


if __name__ == "__main__":
    input_image_ref = cv2.imread("earth.png")
    input_image_target = cv2.imread("rocket.png")
    out = cv2.imread("out.png")

    gray_im_ref = np.asarray(np.mean(input_image_ref, axis=-1), np.uint8)
    gray_im_target = np.asarray(np.mean(input_image_target, axis=-1), np.uint8)
    out = np.asarray(np.mean(out, axis=-1), np.uint8)

    histogram_ref, _ = np.histogram(gray_im_ref, bins=np.arange(256 + 1))
    c_histogram_ref = np.cumsum(histogram_ref) - histogram_ref[0]
    n_ref = gray_im_ref.shape[0] * gray_im_ref.shape[1]

    result_image = match_histogram(gray_im_target, c_histogram_ref, n_ref)
    for i in range(result_image.shape[0]):
        for j in range(result_image.shape[1]):
            if result_image[i][j] - out[i][j] != 0:
                print("wrong")
                break
    while True:
        cv2.imshow("Image (press q to quit)", result_image)
        if cv2.waitKey(10) == ord("q"):
            break
