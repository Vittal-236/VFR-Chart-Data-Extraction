import cv2
import numpy as np
import os
from skimage.feature import match_template

def run_phase_4_debug(chart_path, template_folder):
    # Normalize paths for Windows
    chart_path = os.path.normpath(chart_path)
    template_folder = os.path.normpath(template_folder)

    # Debug: Print the path being checked
    print(f"Checking for chart at: {chart_path}")
    
    if not os.path.exists(chart_path):
        print(f"Error: The file '{chart_path}' does not exist. Check the name!")
        # List files in the directory to help you find the right name
        parent_dir = os.path.dirname(chart_path)
        print(f"Files available in that folder: {os.listdir(parent_dir)}")
        return

    chart = cv2.imread(chart_path, 0)
    if chart is None:
        print("Error: OpenCV could not decode the image. Is it corrupted?")
        return
    
    chart_norm = chart.astype(np.float32) / 255.0
    template_files = [f for f in os.listdir(template_folder) if f.lower().endswith(('.png', '.jpg'))]

    print(f"\n{'Template Name':<20} | {'Max Score Found':<15}")
    print("-" * 40)

    for filename in template_files:
        t_path = os.path.join(template_folder, filename)
        template = cv2.imread(t_path, 0)
        if template is None: continue
        
        best_score = 0
        # Multi-scale check (70% to 130% size)
        for scale in np.linspace(0.7, 1.3, 7): 
            nw = int(template.shape[1] * scale)
            nh = int(template.shape[0] * scale)
            if nw < 10 or nh < 10: continue
            
            t_resized = cv2.resize(template, (nw, nh)).astype(np.float32) / 255.0
            res = match_template(chart_norm, t_resized)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
        
        print(f"{filename:<20} | {best_score:.4f}")

# --- UPDATE THESE TO YOUR ACTUAL FILENAMES ---
chart_file = r"D:\Cognida Internship\VFR extraction\test.png" 
template_dir = r"D:\Cognida Internship\VFR extraction\Template"
run_phase_4_debug(chart_file, template_dir)
