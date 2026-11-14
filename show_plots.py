"""
Interactive Visualization Viewer
Displays generated model plots directly on screen in graphical windows.
"""

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

viz_dir = os.path.join(os.path.dirname(__file__), "visualizations")
images = glob.glob(os.path.join(viz_dir, "*.png"))

if not images:
    print("No images found. Running main.py first...")
    import subprocess
    subprocess.run(["python", "main.py"])
    images = glob.glob(os.path.join(viz_dir, "*.png"))

print(f"Displaying {len(images)} evaluation charts. Close each window to view the next...")

for img_path in sorted(images):
    filename = os.path.basename(img_path)
    img = mpimg.imread(img_path)
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Model Evaluation: {filename}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

print("All charts displayed successfully!")
