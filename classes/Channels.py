import numpy as np
from cv2 import cv2


class Channels(object):

    def __init__(self):
        self._cell_size = 6

    def _calculate_luv(self, image):
        luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        luv = self._convert2cell(luv)
        return luv

    def _calculate_hog(self, image):
        winSize = (60, 120)
        blockSize = (6, 6)
        blockStride = (6, 6)
        cellSize = (6, 6)
        nbins = 6
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hogs = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        hog_feats = hogs.compute(image)
        return hog_feats.reshape(20, 10, nbins)

    def _convert2cell(self, vol, h_cells=20, w_cells=10):
        features = np.zeros((h_cells, w_cells, vol.shape[2]))
        for i in range(h_cells - 1):
            for j in range(w_cells - 1):
                w_offset = j * self._cell_size
                features[i, j, :] = np.sum(vol[
                         i * self._cell_size:i * self._cell_size + self._cell_size,
                         w_offset:w_offset + self._cell_size,
                         :])
        return features

    def _calculate_gradients(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        H_size, W_size = gray.shape

        # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
        sobelx64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        abs_sobel64f = np.absolute(sobelx64f)
        sobel_8u_x = np.uint8(abs_sobel64f).reshape(H_size, W_size, 1)

        # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
        sobely64f = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        abs_sobel64f = np.absolute(sobely64f)
        sobel_8u_y = np.uint8(abs_sobel64f).reshape(H_size, W_size, 1)

        sobel_8u_x = self._convert2cell(sobel_8u_x)
        sobel_8u_y = self._convert2cell(sobel_8u_y)
        return np.dstack((sobel_8u_x, sobel_8u_y))

    def calculate(self, image):
        luv = self._calculate_luv(image)
        gradients = self._calculate_gradients(image)
        hog = self._calculate_hog(image)
        return np.dstack((luv, gradients, hog))
