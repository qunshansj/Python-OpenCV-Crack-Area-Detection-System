python

class CrackDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.with_nmsup = True
        self.fudgefactor = 1.8
        self.sigma = 21
        self.kernel = 2 * math.ceil(2 * self.sigma) + 1

    def orientated_non_max_suppression(self, mag, ang):
        ang_quant = np.round(ang / (np.pi/4)) % 4
        winE = np.array([[0, 0, 0],[1, 1, 1], [0, 0, 0]])
        winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

        magE = self.non_max_suppression(mag, winE)
        magSE = self.non_max_suppression(mag, winSE)
        magS = self.non_max_suppression(mag, winS)
        magSW = self.non_max_suppression(mag, winSW)

        mag[ang_quant == 0] = magE[ang_quant == 0]
        mag[ang_quant == 1] = magSE[ang_quant == 1]
        mag[ang_quant == 2] = magS[ang_quant == 2]
        mag[ang_quant == 3] = magSW[ang_quant == 3]
        return mag

    def non_max_suppression(self, data, win):
        data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
        data_max[data != data_max] = 0
        return data_max

    def detect_cracks(self):
        gray_image = cv2.imread(self.image_path, 0)

        gray_image = gray_image / 255.0
        blur = cv2.GaussianBlur(gray_image, (self.kernel, self.kernel), self.sigma)
        gray_image = cv2.subtract(gray_image, blur)

        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        ang = np.arctan2(sobely, sobelx)

        threshold = 4 * self.fudgefactor * np.mean(mag)
        mag[mag < threshold] = 0
        ......

