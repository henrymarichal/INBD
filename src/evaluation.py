import cv2
import numpy as np
import scipy.optimize
import warnings, typing as tp

from . import datasets


def IoU(a:np.ndarray, b:np.ndarray) -> float:
    '''Compute the Intersection over Union of two boolean arrays'''
    return (a & b).sum() / (a | b).sum()

def mIoU(a:np.ndarray, b:np.ndarray) -> float:
    '''Mean IoU for batched inputs'''
    ious = [IoU(a_i, b_i) for a_i,b_i in zip(a,b)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmean( np.asarray(ious) )

def IoU_matrix(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    a_uniques = np.unique(a[a>0])
    b_uniques = np.unique(b[b>0])

    iou_matrix = []
    for l0 in a_uniques:
        for l1 in b_uniques:
            iou = IoU( (a == l0), (b == l1) )
            iou_matrix.append(iou)
    iou_matrix = np.array(iou_matrix).reshape(len(a_uniques), len(b_uniques))
    return iou_matrix

def IoU_matrix_cuda(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    import torch
    a         = torch.as_tensor(a.astype('int32'), device='cuda')
    b         = torch.as_tensor(b.astype('int32'), device='cuda')
    a_uniques = torch.unique(a[a>0])
    b_uniques = torch.unique(b[b>0])

    iou_matrix = []
    for l0 in a_uniques:
        for l1 in b_uniques:
            iou = IoU( (a == l0), (b == l1) )
            iou_matrix.append( float(iou) )
    iou_matrix = np.asarray(iou_matrix).reshape(len(a_uniques), len(b_uniques))
    return iou_matrix


def evaluate_IoU_matrix(iou_matrix:np.ndarray, iou_threshold:float) -> tp.Dict[str, tp.Any]:
    #match highest ious with each other
    ixs0, ixs1 = scipy.optimize.linear_sum_assignment(iou_matrix, maximize=True)
    #check iou values
    ious       = iou_matrix[ixs0, ixs1]
    ious_ok    = (ious >= iou_threshold)
    ixs0, ixs1 = ixs0[ious_ok], ixs1[ious_ok]

    TP         = np.float32(len(ixs0))
    FP         = len(iou_matrix)    - len(ixs0)
    FN         = len(iou_matrix.T)  - len(ixs0)
    return {
        'TP'        : TP,
        'FP'        : FP,
        'FN'        : FN,
        'precision' : TP / (TP+FP),
        'recall'    : TP / (TP+FN),
    }


def evaluate_single_result(labelmap_result:np.ndarray, labelmap_annotation:np.ndarray, iou_threshold=0.5) -> tp.Dict[str, tp.Any]:
    iou_matrix = IoU_matrix(labelmap_result, labelmap_annotation)
    metrics    =  evaluate_IoU_matrix(iou_matrix, iou_threshold)
    return metrics

def evaluate_single_result_at_iou_levels(labelmap_result:np.ndarray, labelmap_annotation:np.ndarray, iou_levels=np.arange(0.5, 1.0, 0.05)) -> dict:
    iou_matrix      = IoU_matrix_cuda(labelmap_result, labelmap_annotation)
    per_iou_metrics = {}
    for th in iou_levels:
        per_iou_metrics[th]     = evaluate_IoU_matrix(iou_matrix, th)
    return per_iou_metrics


def evaluate_single_result_from_annotationfile(labelmap_result:np.ndarray, annotationfile:str, downscale:float=1.0, *a,**kw) -> dict:
    labelmap_annotation = datasets.load_instanced_annotation(annotationfile, downscale)
    return evaluate_single_result(labelmap_result, labelmap_annotation, *a, **kw)

def evaluate_set_of_files(inputfiles:tp.List[str], annotationfiles:tp.List[str], model, process_kw={}, **eval_kw) -> tp.List[dict]:
    all_metrics = []
    for imgf, tgtf in zip(inputfiles, annotationfiles):
        output  = model.process_image(imgf, **process_kw)
        metrics = evaluate_single_result_from_annotationfile(output.labelmap, tgtf, downscale=model.scale, **eval_kw)
        all_metrics.append(metrics)
    return all_metrics


def compute_ARAND(result:np.ndarray, annotation:np.ndarray) -> float:
    import skimage
    annotation = np.where(annotation < 0, 0, annotation)
    ARAND      = skimage.metrics.adapted_rand_error(annotation, result, ignore_labels=[0])[0]
    return ARAND

from src.util import polygon_2_labelme_json, write_json
def evaluate_single_result_from_files_at_iou_levels(resultfile:str, annotationfile:str,
                            iou_levels=np.arange(0.50, 1.00, 0.05), convert2cstrdmetric=True,
                            debug=True) -> tp.Dict[tp.Any, dict]:
    import skimage
    from . import INBD
    from pathlib import Path
    labelmap_annotation = datasets.load_instanced_annotation(annotationfile, downscale=1)
    labelmap_annotation = INBD.remove_boundary_class(labelmap_annotation)
    labelmap_result     = np.load(resultfile)
    if labelmap_result.shape != labelmap_annotation.shape:
        labelmap_result = skimage.transform.resize(labelmap_result, labelmap_annotation.shape, order=0)
    metrics_per_iou             = evaluate_single_result_at_iou_levels(labelmap_result, labelmap_annotation, iou_levels)
    metrics_per_iou['ARAND']    = compute_ARAND(labelmap_result, labelmap_annotation)


    if convert2cstrdmetric:
        image_path = Path(annotationfile).parent.parent / "InputImages" / f"{Path(annotationfile).stem}.jpg"
        image_orig = cv2.imread(str(image_path))
        if debug:
            image_debug = image_orig.copy()
        classes_values = np.unique(labelmap_result)
        #remove background
        classes_values = classes_values[classes_values > 0]
        polygon_list = []
        for ci in classes_values:
            debug_mask = np.zeros_like(labelmap_result)
            debug_mask[labelmap_result == ci] = 255


            #compute external contour in the mask
            contours, _ = cv2.findContours((debug_mask == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            if debug:
                # draw the contour
                #H, W = labelmap_result.shape
                #contour_image = np.zeros((H, W, 3))
                #contour_image[:, :, 0] = debug_mask
                #contour_image[:, :, 1] = debug_mask
                #contour_image[:, :, 2] = debug_mask

                cv2.drawContours(image_debug, contours, -1, (0, 255, 0), 3)
                debug_mask_path = resultfile.replace('.npy', f'_debugmask_{ci}.png')
                #cv2.imwrite(debug_mask_path, contour_image)


            polygon_list.append(contours[0].squeeze())



        labelme_json = polygon_2_labelme_json(polygon_list, labelmap_result.shape[0], labelmap_result.shape[1], 0, 0, image_orig, str(image_path), -1)
        image_name = Path(annotationfile).stem
        json_path = Path(resultfile).parent / f'{image_name}.json'
        write_json(labelme_json, str(json_path) )

        if debug:
            json_debug_path  = str(json_path).replace('.json','_debug.png')
            print(json_debug_path)
            #write image name in image_debug
            cv2.putText(image_debug, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(json_debug_path, image_debug)





    return metrics_per_iou

def evaluate_resultfiles(resultfiles:tp.List[str], annotationfiles:tp.List[str]):
    all_metrics = []
    for resf, annf in zip(resultfiles, annotationfiles):
        metrics = evaluate_single_result_from_files_at_iou_levels(resf, annf)
        print(resf, combine_metrics_at_iou_levels([metrics]))
        all_metrics.append(metrics)
    combined_metrics = combine_metrics_at_iou_levels(all_metrics)
    return combined_metrics, all_metrics

def combine_metrics(metrics:tp.List[dict]) -> dict:
    result = {}
    for name in ['TP', 'FP', 'FN']:
        result[name] = np.sum( [m[name] for m in metrics] )
    
    for newname, name in {'AR': 'recall'}.items():
        result[newname] = np.nanmean( [m[name] for m in metrics] )
    return result

def combine_metrics_at_iou_levels(metrics:tp.List[dict], iou_levels=np.arange(0.50, 1.00, 0.05) ) -> dict:
    all_combined    = []
    for th in iou_levels:
        this_iou_metrics  = [m[th] for m in metrics]
        this_iou_combined = combine_metrics(this_iou_metrics)
        all_combined.append(this_iou_combined)
    
    return {
        'mAR'   :   np.nanmean( [m['AR']    for m in all_combined] ),
        'ARAND' :   np.nanmean( [m['ARAND'] for m in metrics] ),
    }


