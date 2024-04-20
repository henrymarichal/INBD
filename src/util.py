import time, os, glob, shutil, sys
import typing as tp
import numpy as np
import scipy
import torch

def select_largest_connected_component(x:np.ndarray) -> np.ndarray:
    '''Remove all connected components from binary array mask except the largest'''
    x_labeled      = scipy.ndimage.label(x)[0]
    labels,counts  = np.unique(x_labeled[x_labeled!=0], return_counts=True)
    if len(labels) == 0:
        return x
    maxlabel       = labels[np.argmax(counts)]
    return scipy.ndimage.binary_fill_holes( x_labeled == maxlabel )

def filter_labelmap(labelmap:np.ndarray, threshold=0.001) -> np.ndarray:
    N              = np.prod( labelmap.shape )
    labels, counts = np.unique( labelmap, return_counts=True )
    result         = labelmap.copy()
    for l,c in zip(labels, counts):
        if c/N < threshold:
            result[labelmap==l] = 0
    return result


def backup_code(destination:str) -> str:
    destination = time.strftime(destination)
    cwd      = os.path.realpath(os.getcwd())+'/'
    srcfiles = glob.glob('src/**/*.py', recursive=True) + ['main.py']
    for src_f in srcfiles:
        src_f = os.path.realpath(src_f)
        dst_f = os.path.join(destination, src_f.replace(cwd, ''))
        os.makedirs(os.path.dirname(dst_f), exist_ok=True)
        shutil.copy(src_f, dst_f)
    open(os.path.join(destination, 'args.txt'), 'w').write(' '.join(sys.argv))
    return destination

def output_name(args):
    reso = f'a{args.angular_density:.1f}' if args.modeltype=='INBD' else f'x{args.downsample}'
    name = f'%Y-%m-%d_%Hh%Mm%Ss_{args.modeltype}_{args.epochs}e_{reso}_{args.suffix}'
    name = os.path.join(args.output, name)
    name = time.strftime(name)
    return name

def load_model(path:str):
    importer = torch.package.PackageImporter(path)
    model    = importer.load_pickle('model', 'model.pkl')
    return model


def _infer_segmentationmodel_backbone_name(model):
    backbone = getattr(model, 'backbone_name', None)
    if backbone is None:
        if 'Hardswish' in str(model):
            backbone = 'mobilenet3l'
        else:
            backbone = 'resnet18'
    return backbone

def load_segmentationmodel(path):
    from . import segmentation
    importer = torch.package.PackageImporter(path)
    model    = importer.load_pickle('model', 'model.pkl')
    backbone = _infer_segmentationmodel_backbone_name(model)
    state    = model.state_dict()
    model    = segmentation.SegmentationModel(downsample_factor=model.scale, backbone=backbone)
    model.load_state_dict(state)
    return model


def read_splitfile(splitfile:str) -> tp.List[str]:
    files       = open(splitfile).read().strip().split('\n')
    if files == ['']:
        return []
    
    dirname     = os.path.dirname(splitfile)
    files       = [f if os.path.isabs(f) else os.path.join(dirname, f) for f in files]
    assert all([os.path.exists(f) for f in files])
    return files


def read_splitfiles(images_splitfile:str, annotations_splitfile:str) -> tp.Tuple[tp.List[str], tp.List[str]]:
    imagefiles  = read_splitfile(images_splitfile)
    annotations = read_splitfile(annotations_splitfile)
    assert len(imagefiles) == len(annotations), [len(imagefiles), len(annotations)]
    return imagefiles, annotations


def labelmap_to_areas_output(labelmap:np.ndarray) -> str:
    output        = ''
    labels,counts = np.unique(labelmap[labelmap>0], return_counts=True)
    for l,c in zip(labels, counts):
        output += f'{l}, {c}\n'
    return output

from PIL import Image
def resize_image_using_pil_lib(im_in: np.array, height_output: object, width_output: object) -> np.ndarray:
    """
    Resize image using PIL library.
    @param im_in: input image
    @param height_output: output image height_output
    @param width_output: output image width_output
    @return: matrix with the resized image
    """

    pil_img = Image.fromarray(im_in)
    # Image.ANTIALIAS is deprecated, PIL recommends using Reampling.LANCZOS
    #flag = Image.ANTIALIAS
    flag = Image.Resampling.LANCZOS
    pil_img = pil_img.resize((height_output, width_output), flag)
    im_r = np.array(pil_img)
    return im_r


def polygon_2_labelme_json(chain_list, image_height, image_width, cy, cx, img_orig, image_path,
                         exec_time):
    """
    Converting ch_i list object to labelme format. This format is used to store the coordinates of the rings at the image
    original resolution
    @param chain_list: ch_i list
    @param image_path: image input path
    @param image_height: image hegith
    @param image_width: image width_output
    @param img_orig: input image
    @param exec_time: method execution time
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @return:
    - labelme_json: json in labelme format. Ring coordinates are stored here.
    """
    init_height, init_width, _ = img_orig.shape



    width_cte = init_width / image_width if image_width is not 0 else 1
    height_cte = init_height / image_height if image_height is not 0 else 1
    labelme_json = {"imagePath":image_path, "imageHeight":None,
                    "imageWidth":None, "version":"5.0.1",
                    "flags":{},"shapes":[],"imageData": None, 'exec_time(s)':exec_time,'center':[cy*height_cte, cx*width_cte]}
    for idx, polygon in enumerate(chain_list):

        ring = {"label":str(idx+1)}
        ring["points"] = polygon.tolist()
        ring["shape_type"]="polygon"
        ring["flags"]={}
        labelme_json["shapes"].append(ring)

    return labelme_json

import json
def write_json(dict_to_save: dict, filepath: str) -> None:
    """
    Write dictionary to disk
    :param dict_to_save: serializable dictionary to save
    :param filepath: path where to save
    :return: void
    """
    with open(str(filepath), 'w') as f:
        json.dump(dict_to_save, f)