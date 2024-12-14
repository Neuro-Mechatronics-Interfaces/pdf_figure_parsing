# PDF Figure Parsing #
Contains Python toolkit for parsing figures from old PDF figures from the 80's.  
For now there is just a tool for getting the y-value counts from histograms based on screenshotted images from the PDF.  

--- 

## HistogramParser Overview ##
The Histogram Parser is a Python-based tool designed to extract, process, and analyze data from histogram plots saved as images. It leverages OpenCV for image processing and Tesseract OCR to handle text detection, enabling users to generate CSV reports of histogram data while accurately mapping bins to labeled sections (e.g., cervical levels).

## HistogramParser Features
- Automatically preprocesses images by removing text regions detected using OCR.
- Allows manual selection of histogram and reference pixel ROIs.
- Scales histogram bin heights using user-defined vertical scales.
- Maps histogram bins to labeled segments (e.g., cervical levels).
- Generates detailed CSV reports with segment-level bin data.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR
- Required Python packages (listed in `requirements.txt`)

### Steps
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure Tesseract OCR is installed and added to your system's PATH:
    - On Windows:
      Download and install Tesseract OCR from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract).
    - On Linux:
      ```bash
      sudo apt-get install tesseract-ocr
      ```
    - On macOS:
      ```bash
      brew install tesseract
      ```

4. Verify installation by running:
    ```bash
    tesseract --version
    ```

## HistogramParser Usage

### Command-Line Interface
Use the provided `main.py` to parse single or batch image files.

#### Arguments:
- `--input_file`: Path to a single histogram image file.
- `--batch`: Path to a directory containing multiple histogram images.
- `--tesseract_exe`: Full path to the Tesseract executable (default: `C:\msys64\mingw64\bin\tesseract.exe`).

#### Example:
Process a single file:
```bash
python main.py --input_file data/example.png --tesseract_exe "C:\msys64\mingw64\bin\tesseract.exe"
```

Process a batch of files:
```bash
python main.py --batch data/ --tesseract_exe "C:\msys64\mingw64\bin\tesseract.exe"
```

### Interactive Steps
1. **Set Histogram ROI**:
   - Manually select the region containing the histogram.
2. **Set Reference Pixel ROI for Y-Axis**:
   - Manually select a reference region for determining vertical scaling.
3. **Input Parameters**:
   - Enter the vertical scale (counts) and labeled segments (e.g., `C7, C8, T1`).
4. **Generate Output**:
   - The tool processes the image and saves results in a CSV file (e.g., `example.csv`).

### CSV Output
The generated CSV contains three columns:
1. **Cervical Level**: The label of the segment (e.g., `C7`).
2. **Sub-index**: The index of the bin within the segment.
3. **Count**: The scaled value for the bin.

## File Structure
```
.
├── data/                      # Directory for input images
├── pipeline/                  # Core processing module
│   ├── histogram_parser.py    # Main HistogramParser class
│   ├── __init__.py            # Package initialization
├── main.py                    # CLI script for running the parser
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation (this file)
```

## Troubleshooting
- **Tesseract OCR not found**:
  Ensure Tesseract is installed and its path is correctly provided in the `--tesseract_exe` argument.
- **ROI selection issues**:
  Make sure to accurately select the histogram and reference ROIs during the interactive process.
- **Unexpected CSV results**:
  Verify that the input vertical scale matches the unit of the Y-axis in the image.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and open a pull request.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments
- OpenCV for image processing.
- Tesseract OCR for text detection and recognition.

