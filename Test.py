import glob 
from PIL import Image
from ade_config import *
import numpy as np
import matplotlib.pyplot as plt

paths = glob.glob(data_path+images_folder+"/*"+images_format)

resolutions = []
widths = []
heights = []
for path in paths: 
    img = np.array(Image.open(path))
    w = img.shape[0]
    h = img.shape[1]
    resolutions.append(w*h)
    widths.append(w)
    heights.append(h)

resolutions = np.array(resolutions)

print(f"Max height = {np.max(heights)}")
print(f"Min height = {np.min(heights)}\n")
print(f"Max width = {np.max(widths)}")
print(f"Min width = {np.min(widths)}\n")
print(f"Max res = {np.max(resolutions)}")
print(f"Min res = {np.min(resolutions)}\n")
print(f"Number res < 1000000 = {np.sum(resolutions<1000000)}\n")
print(f"Max res < 1000000 = {np.max(resolutions[resolutions<1000000])}\n")

import pdb 
pdb.set_trace()



plt.figure()
plt.hist(resolutions, density=False, bins=30)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Resolution')
plt.savefig("./resolution.png")
plt.close()

plt.figure()
plt.scatter(resolutions, np.ones_like(resolutions))  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Resolution')
plt.savefig("./resolution2.png")
plt.close()