import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

from src import (util)

def from_numpy_to_pil(image: np.ndarray) -> Image:
    return Image.fromarray(image)

def main(model_path="~/Documents/repo/fing/INBD/models/segmentation/inbd_4/model.pt.zip",
         dataset_dir="/data/maestria/datasets/Pinus_Taeda/PinusTaedaV3/images/segmented/"):

    image_path = "/data/maestria/resultados/inbd_4/InputImages/L03e.png"
    image_path = "/data/maestria/datasets/Pinus_Taeda/PinusTaedaV3/images/segmented/C17-2.jpg"
    image_path = "/data/maestria/datasets/gleditsia_triacanthos/images/segmented/20230504_194426_zoom_in.jpg"
    image_path = "/data/maestria/datasets/TreeTrace_Douglas_format/discs_zoom_in/images/segmented/A07d.jpeg"
    image_path = "/data/maestria/datasets/INBD/EH/inputimages/EH_0055.jpg"
    segmentationmodel = util.load_segmentationmodel(model_path)

    background_list, center_list, boundary_list, image_list = [], [], [], []
    for image_path in Path(dataset_dir).rglob("*.jpg"):

        output = segmentationmodel.process_image(str(image_path), upscale_result=False)
        background = output.background
        background -= background.min()
        center = output.center
        center -= center.min()
        boundary = output.boundary
        boundary -= boundary.min()

        #

        background_list.append(from_numpy_to_pil(background).convert("L"))
        center_list.append(from_numpy_to_pil(center).convert("L"))
        boundary_list.append(from_numpy_to_pil(boundary).convert("L"))

        img = Image.open(str(image_path))
        #resize image to background shape
        img = img.resize(background.shape[::-1])

        image_list.append(img)

    images = []
    #generate pdf
    for idx in range(len(background_list)):
        images.append(image_list[idx])
        images.append(boundary_list[idx])
        images.append(background_list[idx])
        images.append(center_list[idx])


    pdf_path = "./output/segmentation/pinus_taeda_v3_segmented.pdf"
    images[0].save(pdf_path, save_all=True, append_images=images[1:], quality=100)

    return


if __name__ == "__main__":
    main()