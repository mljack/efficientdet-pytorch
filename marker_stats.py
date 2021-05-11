import os
import sys
import json
import collections
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from scipy.stats import kde

def listdir(path):
    result = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isfile(full_path):
            result.append(full_path)
            continue
        result += listdir(full_path)

    return result

def hist(d, title, x_range=None, y_range=None, nbins=None):
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=d, bins='auto' if nbins is None else nbins, color='#0504aa', alpha=0.7, rwidth=10)
    plt.grid(axis='x', alpha=0.75)
    plt.grid(axis='y', alpha=0.75)
    #plt.xlabel('Value')
    #plt.ylabel('Frequency')
    plt.title(title)
    #plt.text(23, 45, r'$\mu=15, b=3$')
    #max_x = 8
    max_y = n.max()
    # Set a clean upper y-axis limit.
    #plt.xlim(xmax=8)
    if x_range is not None:
        plt.xlim(xmin=x_range[0], xmax=x_range[1])
    if y_range is not None:
        plt.ylim(ymin=y_range[0], ymax=y_range[1])
    else:
        plt.ylim(ymax=np.ceil(max_y / 10) * 10 if max_y % 10 else max_y + 10)
    plt.show()


def hist_2d(pts):
   
    # Create data: 200 points
    data = np.array(pts)
    x, y = data.T
    nbins = 20

    # 2D Histogram
    plt.hist2d(x, y, bins=nbins)
    plt.show()

    # # Create a figure with 6 plot areas
    # fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(21, 5))
    
    # # Everything starts with a Scatterplot
    # axes[0].set_title('Scatterplot')
    # axes[0].plot(x, y, 'ko')
    # # As you can see there is a lot of overlapping here!
    
    # # Thus we can cut the plotting window in several hexbins
    # nbins = 20
    # axes[1].set_title('Hexbin')
    # axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)
    
    # # 2D Histogram
    # axes[2].set_title('2D Histogram')
    # axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)
    
    # # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    # k = kde.gaussian_kde(data.T)
    # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    # # plot a density
    # axes[3].set_title('Calculate Gaussian KDE')
    # axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.BuGn_r)
    
    # # add shading
    # axes[4].set_title('2D Density with shading')
    # axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    
    # # contour
    # axes[5].set_title('Contour')
    # axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    # axes[5].contour(xi, yi, zi.reshape(xi.shape) )

def run(folder):
    ratios = []
    areas = []
    angles = []
    lengths = []
    widths = []
    pts = []
    scores = []
    certainties = []
    certainties2 = []
    file_list = listdir(folder)
    #random.shuffle(file_list)
    for path in file_list:
        if path.find(".vehicle_markers.json") != -1:
            with open(path) as json_file:
                markers = json.load(json_file)
            for mm in markers:
                m = mm[0]
                #print(m["length"], m["width"], m["heading_angle"])
                ratios.append(m["length"] / m["width"])
                areas.append(m["length"] * m["width"])
                angles.append(m["heading_angle"] % 360.0)
                lengths.append(m["length"])
                widths.append(m["width"])
                score = m["score"]
                certainty = m["certainty"]
                certainty2 = m["certainty2"]
                if 1:
                #if certainty2 < 0.2:
                    scores.append(score)
                if 1:
                #if score < 0.9:
                #if score > 0.7 and score < 0.9:
                #if certainty < 0.8:
                    certainties.append(certainty)
                    certainties2.append(certainty2)
        if path.find(".txt") != -1:
            scale = 768
            with open(path) as text_file:
                for line in text_file.readlines():
                    values = [float(v) for v in line.replace("\n", "").split(" ")]
                    #print(values)
                    t, x, y, length, width, heading_angle = values
                    x *= scale
                    y *= scale
                    length *= scale
                    width *= scale
                    ratios.append(length / width)
                    areas.append(length * width)
                    angles.append(heading_angle % 360.0)
                    lengths.append(length)
                    widths.append(width)
                    pts.append((x,y))
                    if width > 50.0:
                        print(path)
                        print(t, x, y, length, width, heading_angle)
                        img = cv2.imread(path.replace(".txt", ".jpg"))
                        cv2.imshow("test" ,img)
                        k = cv2.waitKey(0)
                        if k == 27:
                            exit(0)


    #print(ratios)
    # hist_2d(pts)
    # hist(ratios, "Aspect Ratios")
    # hist(areas, "Areas")
    # hist(angles, "Heading Angles")
    # hist(lengths, "Length")
    # hist(widths, "Width")
    y_max = 500

    hist(scores, "Score", x_range=[0.0, 1.0], y_range=[0.0, float(y_max)], nbins=100)
    hist(certainties, "Certainty - iou", x_range=[0.0, 1.0], y_range=[0.0, float(y_max)], nbins=100)
    hist(certainties2, "Certainty - iou * confidence", x_range=[0.0, 1.0], y_range=[0.0, float(y_max)], nbins=100)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python markers_stats.py folder")
    else:
        run(sys.argv[1])