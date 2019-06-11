__author__ = "Thai Thien"
__email__ = "tthien@apcs.vn"
from tensorflow.keras.preprocessing import image
import numpy as np
import scipy
from scipy.io import loadmat
from matplotlib import pyplot as plt


def gaussian_filter_density(gt):
    print (gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    pts_copy = pts.copy()
    tree = scipy.spatial.KDTree(pts_copy, leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density

def single_sample_prototype():
    img_path = '/data/dump/ShanghaiTech/part_A/train_data/images/IMG_2.jpg'
    print(img_path)
    mat_path = "/data/dump/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_2.mat"
    mat = scipy.io.loadmat(mat_path)
    imgfile = image.load_img(img_path)
    img = image.img_to_array(imgfile)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)

if __name__ == "__main__":
    """
    TODO: this file will preprocess crowd counting dataset
    """
    image_path = "/data/dump/ShanghaiTech/part_A/train_data/images/"
    mat_path = "/data/dump/ShanghaiTech/part_A/train_data/ground-truth/"
    pass

