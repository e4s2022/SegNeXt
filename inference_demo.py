import mmcv
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
import warnings

PALETTE = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], 
            [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], 
            [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], 
            [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

def show_result(img, seg, seg_gt = None, opacity = 0.5, out_file = None, win_name='', show=False, wait_time=0):
    img = mmcv.imread(img)
    img = img.copy()
    img = mmcv.imresize(img, size = seg.shape[:2])
    
    
    palette = np.array(PALETTE)
   
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img1 = img * (1 - opacity) + color_seg * opacity
    img1 = img1.astype(np.uint8)

    if seg_gt is not None:
        color_seg_gt = np.zeros((seg_gt.shape[0], seg_gt.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg_gt[seg_gt == label, :] = color
        # convert to BGR
        color_seg_gt = color_seg_gt[..., ::-1]

        img2 = img * (1 - opacity) + color_seg_gt * opacity
        img2 = img2.astype(np.uint8)

        # stack the results with seg_gt horizontally
        img1 = np.hstack([img1, img2])

    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

    if show:
        mmcv.imshow(img1, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img1, out_file)

    if not (show or out_file):
        warnings.warn('show==False and out_file is not specified, only '
                        'result image will be returned')
        return img1

config_file = './local_configs/segnext/small/segnext.small.512x512.celebamaskhq.160k.py'
checkpoint_file = './work_dirs/segnext.small.512x512.celebamaskhq.160k/best_mIoU_iter_140000.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

for i in range(20):
    # test a single image and show the results
    img = '/mnt/hdd8T/lza/py_projs/SegNeXt/data/CelebAMaskHQ/CelebA-HQ-img/%d.jpg'%(28000+i) 
    img = mmcv.imread(img)
    
    seg_gt = '/mnt/hdd8T/lza/py_projs/SegNeXt/data/CelebAMaskHQ/CelebA-HQ-mask/%d.png'%(28000+i) 
    seg_gt = mmcv.imread(seg_gt)[:,:,0]

    result = inference_segmentor(model, img)

    
    # you can change the opacity of the painted segmentation map in (0, 1].
    show_result(img, result[0], seg_gt = seg_gt, out_file='vis/%d.jpg'%(28000+i), opacity=0.75)

