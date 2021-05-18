from nilearn import plotting
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


def all_axes(pt_id, x, y, z):
    img = "./hecktor_train/hecktor_nii/" + pt_id + "/" + pt_id + "_ct.nii.gz"
    label = "./hecktor_train/hecktor_nii/" + pt_id + "/" + pt_id + "_ct_gtvt.nii.gz"
    display = plotting.plot_anat(anat_img=img, cut_coords=(x, y, z), draw_cross=False)
    display.add_contours(label, levels=[0.5], colors="r")

def og_slice(pt_id, slc):
    total_slices = 0
    img_path = "./hecktor_train/hecktor_nii/" + pt_id + "/" + pt_id + "_ct.nii.gz"
    gt_path = "./hecktor_train/hecktor_nii/" + pt_id + "/" + pt_id + "_ct_gtvt.nii.gz"
    img_array = nib.load(img_path).get_fdata()
    gt_array = nib.load(gt_path).get_fdata()
    total_slices = img_array.shape[2]

    img = np.zeros((total_slices, 512, 512))
    gt = np.zeros((total_slices, 512, 512))

    for slice_ in range(0, total_slices):
        img_data = img_array[:, :, slice_]
        gt_data = gt_array[:, :, slice_]
        img[slice_] = img_data
        gt[slice_] = gt_data

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img[slc], cmap="gray")
    ax.contour(gt[slc], [.5], colors="red")

def rs_slice(pt_id, slc, x, y, z):
    total_slices = 0
    img_path = "./hecktor_train/hecktor_nii/" + pt_id + "/" + pt_id + "_ct.nii.gz"
    gt_path = "./hecktor_train/hecktor_nii/" + pt_id + "/" + pt_id + "_ct_gtvt.nii.gz"
    img_array = nib.load(img_path).get_fdata()
    gt_array = nib.load(gt_path).get_fdata()
    img_array = resize(img_array, (x, y, z))
    gt_array = resize(gt_array, (x, y, z))
    total_slices = img_array.shape[2]

    rs_img = np.zeros((total_slices, x, y))
    rs_gt = np.zeros((total_slices, x, y))

    for slice_ in range(0, total_slices):
        img_data = img_array[:, :, slice_]
        gt_data = gt_array[:, :, slice_]
        rs_img[slice_] = img_data
        rs_gt[slice_] = gt_data

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rs_img[slc], cmap="gray")
    ax.contour(rs_gt[slc], [.5], colors="red")