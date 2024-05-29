import numpy as np
import argparse
import cv2

def mannually(filename):
    import os
    os.system("ls -lah .")
    raise
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--type", type=int, required=True, default=0)
    args = parser.parse_args()

    mannually(args.input)


