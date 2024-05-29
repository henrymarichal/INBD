import numpy as np
import argparse
import cv2
from shapely.geometry import Polygon
from pathlib import Path
def mannually(filename, img_path, output_img):
    import os
    #open txt with pandas
    f = open(filename, "r")
    poly = []
    for line in f:
        print(line)
        y, x = line.replace("[", "").replace("]", "").split(",")
        poly.append([float(y), float(x)])

    f.close()

    #print(df)
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, np.array([poly], dtype=np.int32), 255)
    #save mask
    cv2.imwrite(output_img, mask)
    print(output_img)

    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_poly", type=str, required=True)
    parser.add_argument("--input_img", type=str, required=True)
    parser.add_argument("--output_img", type=str, required=True)

    args = parser.parse_args()

    mannually(args.input_poly, args.input_img, args.output_img)


