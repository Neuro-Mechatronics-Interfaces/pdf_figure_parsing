import cv2
import numpy as np
import pytesseract
import csv
import os

class HistogramParser:
    def __init__(self, image_path, tesseract_path: str=r"C:\\msys64\\mingw64\\bin\\tesseract.exe"):
        self.image_path = image_path
        self.image = None
        self.thresh = None
        self.vertical_scale_input = 1
        self.vertical_scale_output = 1
        self.bins = None
        self.bin_values = []
        self.bin_to_segment_map = {}
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    import cv2
import numpy as np
import pytesseract
import csv
import os

class HistogramParser:
    def __init__(self, image_path, tesseract_path: str=r"C:\\msys64\\mingw64\\bin\\tesseract.exe"):
        self.image_path = image_path
        self.image = None
        self.thresh = None
        self.vertical_scale_input = 1
        self.vertical_scale_output = 1
        self.bins = None
        self.bin_values = []
        self.bin_to_segment_map = {}
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def load_and_preprocess(self):
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image to be twice as wide and twice as tall
        self.image = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Perform OCR to detect text regions
        ocr_boxes = pytesseract.image_to_boxes(self.image)

        # Blank out detected text regions in the thresholded image
        self.thresh = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV)[1]
        h, w = self.image.shape
        for box in ocr_boxes.splitlines():
            b = box.split()
            if len(b) == 6:
                x_min = int(b[1])
                y_min = h - int(b[2])  # Flip y-coordinates to match OpenCV
                x_max = int(b[3])
                y_max = h - int(b[4])
                cv2.rectangle(self.thresh, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)

    def detect_bins(self, x, w):
        vertical_edges = cv2.Sobel(self.thresh, cv2.CV_64F, 1, 0, ksize=3)
        vertical_edges = np.uint8(np.abs(vertical_edges))
        vertical_edges = cv2.threshold(vertical_edges, 50, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.bins = len(contours)
        return w / self.bins  # Bin width

    def map_bins_to_level(self, bin_width, segment_labels, histogram_roi, w, h):
        # Estimate the total number of bins
        total_bins = int(w // bin_width)

        # Distribute bins approximately equally among cervical levels
        bins_per_level = total_bins // len(segment_labels)
        remaining_bins = total_bins % len(segment_labels)

        bin_index = 0
        for level_index, label in enumerate(segment_labels):
            bins_for_this_level = bins_per_level + (1 if level_index < remaining_bins else 0)

            for sub_index in range(bins_for_this_level):
                if bin_index >= total_bins:
                    break

                bin_x = int(bin_index * bin_width)
                bin_roi = histogram_roi[:, bin_x:bin_x + int(bin_width)]

                # Calculate the height of the histogram bar relative to the bottom of the ROI
                bar_heights = np.where(bin_roi > 0)[0]  # Get indices of non-zero pixels
                if len(bar_heights) > 0:
                    bar_height = h - bar_heights.min()
                else:
                    bar_height = 0

                # Scale the height to the user-provided vertical scale
                bin_value = round((bar_height / self.vertical_scale_input) * self.vertical_scale_output)

                self.bin_values.append(bin_value)
                self.bin_to_segment_map[bin_index] = (label, sub_index)
                bin_index += 1

    def parse(self):
        self.load_and_preprocess()

        contour_image = self.image.copy()
        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)  # Convert to color image

        print("Set histogram ROI")
        x, y, w, h = cv2.selectROI("Set Histogram ROI", contour_image, showCrosshair=True)
        bin_width = self.detect_bins(x, w)

        print("Set reference pixel height for Y-Axis")
        reference_roi = cv2.selectROI("Set Reference Pixel ROI", contour_image, showCrosshair=True)

        # Extract the height in pixels of the selected reference
        _, _, _, reference_height = reference_roi

        self.vertical_scale_input = reference_height  # This replaces the need to extract it from the histogram ROI
        self.vertical_scale_output = int(input("Enter Y-Scale:"))
        segment_labels = input("Enter segment levels (split by ','):").split(",")

        histogram_roi = self.thresh[y:y + h, x:x + w]
        self.map_bins_to_level(bin_width, segment_labels, histogram_roi, w, h)
        return {
            "bins": self.bins,
            "bin_values": self.bin_values,
            "bin_to_segment_map": self.bin_to_segment_map,
        }

    def parse_report(self, output_path):
        if not self.bin_to_segment_map or not self.bin_values:
            raise ValueError("Bins or segmental map is not populated. Ensure parse() is called first.")

        # Prepare data for CSV
        rows = []
        global_index = 0
        for bin_idx, bin_value in enumerate(self.bin_values):
            level, sub_index = self.bin_to_segment_map.get(bin_idx, ("Unknown", "Unknown"))
            rows.append([global_index, level, sub_index, bin_value])
            global_index+=1

        # Write to CSV
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Section", "Level", "Subsection", "Count"])
            writer.writerows(rows)
        print(f"CSV report saved to {output_path}")
