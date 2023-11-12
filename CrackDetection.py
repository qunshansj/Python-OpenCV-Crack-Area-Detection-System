python

class CrackDetector:
    def __init__(self, input_image_path, output_image_path):
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path

    def detect_cracks(self):
        # Read input image
        img = cv2.imread(self.input_image_path)

        # Convert into gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Image processing (smoothing)
        blur = cv2.blur(gray, (3, 3))

        # Apply logarithmic transform
        img_log = (np.log(blur + 1) / (np.log(1 + np.max(blur)))) * 255
        img_log = np.array(img_log, dtype=np.uint8)

        # Image smoothing: bilateral filter
        bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

        # Canny Edge Detection
        edges = cv2.Canny(bilateral, 100, 200)

        # Morphological Closing Operator
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Create feature detecting method
        orb = cv2.ORB_create(nfeatures=1500)
        ......

