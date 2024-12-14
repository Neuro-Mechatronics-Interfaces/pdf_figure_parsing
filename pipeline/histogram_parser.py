import cv2
import numpy as np
import pytesseract
import csv, os, re, shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        self.color_LUT = {
            "C4": (36 / 255, 36 / 255, 245 / 255),
            "C5": (73 / 255, 73 / 255, 240 / 255),
            "C6": (109 / 255, 109 / 255, 235 / 255),
            "C7": (146 / 255, 146 / 255, 225 / 255),
            "C8": (182 / 255, 182 / 255, 210 / 255),
            "T1": (150 / 255, 150 / 255, 150 / 255),
            "T2": (100 / 255, 100 / 255, 100 / 255),
            "Unknown": (50 / 255, 50 / 255, 50 / 255), 
        }
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def load_and_preprocess(self):
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image to be twice as wide and twice as tall
        self.image = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Perform OCR to detect text regions
        ocr_boxes = pytesseract.image_to_boxes(self.image, config="--psm 6")

        # Blank out detected text regions in the thresholded image
        self.thresh = cv2.adaptiveThreshold(
            self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        cv2.imshow("Binarized Image", self.thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        h, w = self.image.shape
        # Display OCR boxes and allow deselection
        def on_mouse_click(event, x, y, flags, param):
            if event == 4:  # Left mouse button click
                for i, (box, rect) in enumerate(param['boxes']):
                    x_min, y_max, x_max, y_min = rect
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        if i in param['deselected']:
                            param['deselected'].remove(i)  # Remove from deselected (add back)
                            print(f"Re-added box {i}: {box}")
                        else:
                            param['deselected'].add(i)  # Add to deselected
                            print(f"Deselected box {i}: {box}")
                        break


        temp_image = self.image.copy()
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)
        ocr_box_data = []

        # Draw all OCR-detected boxes
        for box in ocr_boxes.splitlines():
            b = box.split()
            if len(b) == 6:
                x_min, y_min = int(b[1]), h - int(b[2])
                x_max, y_max = int(b[3]), h - int(b[4])
                ocr_box_data.append((b, (x_min, y_min, x_max, y_max)))
                cv2.rectangle(temp_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Interactive deselection
        deselected_boxes = set()
        cv2.imshow("Select boxes to deselect", temp_image)
        cv2.setMouseCallback("Select boxes to deselect", on_mouse_click, {'boxes': ocr_box_data, 'deselected': deselected_boxes})
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Apply text rejection, excluding deselected boxes
        for i, (box, rect) in enumerate(ocr_box_data):
            if i not in deselected_boxes:
                x_min, y_min, x_max, y_max = rect
                cv2.rectangle(self.thresh, (max(x_min-2,0), min(y_min+2,h)), (min(x_max+2,w), max(y_max-2,0)), (0, 0, 0), -1)
        cv2.imshow("Binarized Image: Updated", self.thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_bins(self, x, w):
        vertical_edges = cv2.Sobel(self.thresh, cv2.CV_64F, 1, 0, ksize=3)
        vertical_edges = np.uint8(np.abs(vertical_edges))
        vertical_edges = cv2.threshold(vertical_edges, 50, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.bins = len(contours)
        return w / self.bins  # Bin width

    def generate_plot(self, output_image_path):
        if not self.bin_to_segment_map or not self.bin_values:
            raise ValueError("Bins or segment map is not populated. Ensure parse() is called first.")

        # Extract the muscle and monkey names from the image path
        filename = os.path.basename(self.image_path)
        match = re.match(r"(M\d+-\d+)_(\w+)\.png", filename)
        monkey, muscle = match.groups() if match else ("Unknown Monkey", "Unknown Muscle")

        # Generate staircase plot
        x = np.arange(len(self.bin_values))
        y = self.bin_values

        plt.figure(figsize=(12, 8))
        fig, ax = plt.subplots(figsize=(12, 8))

        # Customize the y-axis (single scale bar)
        max_y = max(self.bin_values)+3
        ax.axhline(0, color="black", linewidth=1.5, zorder=-1)
        ax.axhline(self.vertical_scale_output, color="grey", linewidth=1.0, linestyle="--")
        ax.plot([0, 0], [0, max_y], color="black", linewidth=2.5)  # Dummy Y-axis vert line
        ax.text(-1, self.vertical_scale_output + 0.5, f"{self.vertical_scale_output}",
                fontsize=16, fontname="Tahoma", verticalalignment="bottom", horizontalalignment="right", color="grey")

        # Hide all axis lines, ticks, and tick labels
        ax.axis("off")

        # Title and Subtitle
        plt.title(f"{muscle} MN Counts", fontsize=20, fontname="Tahoma", weight="bold")
        plt.suptitle(f"{monkey}", fontsize=16, fontname="Tahoma", weight="bold")

        # Add labels for cervical segments and color rectangles
        segments = list(set([v[0] for v in self.bin_to_segment_map.values()]))
        segments.sort()  # Ensure order is correct

        segment_boundaries = []
        for segment in segments:
            indices = [i for i, v in self.bin_to_segment_map.items() if v[0] == segment]
            if indices:
                segment_boundaries.append((indices[0], indices[-1], segment))

        for start, end, label in segment_boundaries:
            color = self.color_LUT.get(label, (0.8, 0.8, 0.8))  # Default to light gray if label not in self.color_LUT
            rect = patches.Rectangle((start, -2.5), end - start, 2.0, color=color, alpha=0.5)
            ax.add_patch(rect)
            ax.text((start + end) / 2, -1.5, label, ha="center", va="center", fontsize=14,
                    fontname="Tahoma", weight="bold", color="white")
            ax.bar(x[start:end+1], y[start:end+1], linewidth=1, edgecolor="black", color=color, zorder=3)

        # Adjust y-axis range
        ax.set_ylim(-3, max_y)

        # Save the plot
        plt.savefig(output_image_path, bbox_inches="tight", dpi=300)
        plt.show()
        print(f"Bar plot saved to {output_image_path}")

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
        cv2.destroyAllWindows()

        self.map_bins_to_level(bin_width, segment_labels, histogram_roi, w, h)
        return {
            "bins": self.bins,
            "bin_values": self.bin_values,
            "bin_to_segment_map": self.bin_to_segment_map,
        }

    def parse_report(self):
        if not self.bin_to_segment_map or not self.bin_values:
            raise ValueError("Bins or segment map is not populated. Ensure parse() is called first.")

        # Extract the directory and filename
        input_dir = os.path.dirname(self.image_path)
        input_filename = os.path.basename(self.image_path)

        results_dir = os.path.join(input_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        output_filename = input_filename.replace(".png", ".csv")
        output_path = os.path.join(results_dir, output_filename)

        # Prepare data for CSV
        rows = []
        global_bin = 0
        for bin_idx, bin_value in enumerate(self.bin_values):
            level, sub_index = self.bin_to_segment_map.get(bin_idx, ("Unknown", "Unknown"))
            rows.append([global_bin, level, sub_index, bin_value])
            global_bin += 1

        # Write to CSV
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Section", "Level", "Subsection", "Count"])
            writer.writerows(rows)
        print(f"CSV report saved to {output_path}")                
        
        # Construct the output file path
        output_image_filename = input_filename.replace(".png", "-parsed.png")
        output_image_path = os.path.join(results_dir, output_image_filename)

        # Generate and save plot
        self.generate_plot(output_image_path)

        # Create the parsed sub-folder
        parsed_dir = os.path.join(input_dir, "parsed")
        os.makedirs(parsed_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Move the image file to new folder indicating it has been parsed
        transfer_path = os.path.join(parsed_dir, input_filename)
        shutil.move(self.image_path, transfer_path)
