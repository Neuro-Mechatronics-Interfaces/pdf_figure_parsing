import argparse
import os
import re
from pipeline.histogram_parser import HistogramParser

def parse_single_file(input_file, tesseract_exe):
    parser = HistogramParser(input_file, tesseract_path=tesseract_exe)
    results = parser.parse()

    # Save results to CSV
    output_csv = input_file.replace('.png', '.csv')
    parser.parse_report(output_csv)

    return results

def batch_parse(directory, tesseract_exe):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            match = re.match(r"(M\d+)-\d+_(\w+)\.png", filename)
            if match:
                monkey, muscle = match.groups()
            else:
                monkey, muscle = "Unknown", "Unknown"

            parser = HistogramParser(file_path, tesseract_path=tesseract_exe)
            file_results = parser.parse()

            # Save results to CSV
            output_csv = file_path.replace('.png', '.csv')
            parser.parse_report(output_csv)

            file_results["monkey"] = monkey
            file_results["muscle"] = muscle
            results.append(file_results)

    return results

def main():
    parser = argparse.ArgumentParser(description="Process histogram plot images to extract data.")
    parser.add_argument("--input_file", type=str, help="Path to a single input image file containing the histogram plot.")
    parser.add_argument("--batch", type=str, help="Path to a directory containing multiple histogram plot images.")
    parser.add_argument("--tesseract_exe", default=r"C:\\msys64\\mingw64\\bin\\tesseract.exe", type=str, help="Full file path to tesseract executable binary file. (Default: \"C:\\msys64\\mingw64\\bin\\tesseract.exe\")")
    args = parser.parse_args()

    if args.input_file:
        results = parse_single_file(args.input_file, args.tesseract_exe)
        print("Results for single file:")
        print("Number of Bins:", results["bins"])
        print("Bin Values:", results["bin_values"])
        print("Bin to Cervical Section Mapping:", results["bin_to_segment_map"])
    elif args.batch:
        results = batch_parse(args.batch, args.tesseract_exe)
        print("Results for batch processing:")
        for file_result in results:
            print("---")
            print(f"Monkey: {file_result['monkey']}, Muscle: {file_result['muscle']}")
            print("Number of Bins:", file_result["bins"])
            print("Bin Values:", file_result["bin_values"])
            print("Bin to Cervical Section Mapping:", file_result["bin_to_segment_map"])
    else:
        print("Please specify either --input_file or --batch.")

if __name__ == "__main__":
    main()
