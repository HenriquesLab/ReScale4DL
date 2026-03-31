# Functions for morphology analysis of 3D label images

# Import required libraries
import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pypdf  # type: ignore
import re
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import MaxNLocator  # type: ignore
import seaborn as sns  # type: ignore
import skimage as ski  # type: ignore
from skimage.measure._regionprops_utils import perimeter  # type: ignore
from sklearn import metrics as skl  # type: ignore
from time import perf_counter, strftime, gmtime
from scipy import ndimage  # type: ignore
from typing import List, Optional, Tuple, Dict, Literal, Union
from ..utils import find_matching_labels_3d

## Main Function


def morphology_3d(
    main_directory: str,
    skip_directories: Optional[List[str]] = [".DS_Store", "__pycache__"],
    sampling_dir_list: Optional[List[str]] = [
        "upsampling_16",
        "upsampling_8",
        "upsampling_4",
        "upsampling_2",
        "OG",
        "downsampling_2",
        "downsampling_4",
        "downsampling_8",
        "downsampling_16",
    ],
    run_per_object_stats: Optional[bool] = True,
    run_semantic_stats: Optional[bool] = True,
    run_binary_mask_stats: Optional[bool] = True,
) -> None:
    """
    Calculate the properties for each object in each 3D volume in the input
    directory.

    Args:
        main_directory (str): The input directory containing the sub folders
                            containing the image files.

    Expected file arrangement example:
        +-- main_directory
        |  +-- Dataset
        |  |  +-- OG
        |  |  |  +-- GT
        |  |  |  |  +-- volumes.tiff
        |  |  |  +-- Prediction
        |  |  |  |  +-- volumes.tiff
        |  |  +-- downsampling_2
        |  |  |  +-- GT
        |  |  |  |  +-- volumes.tiff
        |  |  |  +-- Prediction
        |  |  |  |  +-- volumes.tiff
    """

    # Get contents of the directory and start timer
    directory_list = os.listdir(main_directory)
    begin_time = perf_counter()

    # Loop through the sub directories
    for sub_dir in directory_list:
        curr_dir = os.path.join(main_directory, sub_dir)

        # Skip misc folders
        if sub_dir in skip_directories:
            continue

        # Skip if not a directory
        elif not os.path.isdir(curr_dir):
            continue

        # Remaining sub directories are the ones to calculate properties for
        else:
            print("Calculating properties for " + sub_dir)
            reset_sampling_dir_list = False
            if sampling_dir_list is None:
                sampling_dir_list = [i for i in os.listdir(curr_dir) if
                                     os.path.isdir(os.path.join(curr_dir, i))]
                reset_sampling_dir_list = True
            # Create folder to store results if it doesn't exist
            result_dir = os.path.join(curr_dir, "Results")
            base_result_dir = result_dir
            count = 1

            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            else:
                while os.path.exists(result_dir):
                    result_dir = base_result_dir + "_" + f"{count:02d}"
                    count += 1
                os.mkdir(result_dir)

            # Calculate properties
            if run_per_object_stats:
                per_object_statistics_3d(
                    directory=curr_dir,
                    result_dir=result_dir,
                    sampling_dir_list=sampling_dir_list,
                )

            if run_semantic_stats:
                semantic_statistics_3d(
                    directory=curr_dir,
                    result_dir=result_dir,
                    sampling_dir_list=sampling_dir_list,
                )

            if run_binary_mask_stats:
                binary_mask_statistics_3d(
                    directory=curr_dir,
                    result_dir=result_dir,
                    sampling_dir_list=sampling_dir_list,
                )
            if reset_sampling_dir_list:
                sampling_dir_list = None

    # Print total time taken
    total_time = strftime("%H:%M:%S", gmtime(perf_counter() - begin_time))
    print(f"Total time: {total_time}")


## Per object Prediction statistics functions


def per_object_statistics_3d(
    directory: str,
    result_dir: str,
    sampling_dir_list: Optional[List[str]] = [
        "upsampling_16",
        "upsampling_8",
        "upsampling_4",
        "upsampling_2",
        "OG",
        "downsampling_2",
        "downsampling_4",
        "downsampling_8",
        "downsampling_16",
    ],
    iou_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the IoU, f1 score, and other statistics for each object in
    3D volumes.

    Matching is done label-wise using find_matching_labels_3d, and
    true positives / false negatives / false positives are computed
    voxel-wise in a way consistent with the 2D implementation.

    Parameters
    ----------
    directory : str
        Directory with folders of sampling folders with GT and Prediction
        folder pairs inside.
    result_dir : str
        Directory to save the results.
    sampling_dir_list : list of str, optional
        List of sampling folders to analyse.
    iou_threshold : float, optional
        Minimum IoU to accept a GT–Prediction pair as a true positive.

    Returns
    -------
    summary_df : pd.DataFrame
        Summary statistics per volume (FOV).
    IoU_per_obj_df : pd.DataFrame
        Per-object statistics.
    """
    # Create dataframes to store the results
    IoU_per_obj_df = pd.DataFrame([])
    summary_df = pd.DataFrame([])
    count_df = pd.DataFrame([])
    start_time = None

    # Lists to store all the data
    GP_folder_list = []
    file_name_list = []
    GT_label_list = []
    pred_label_list = []
    GT_px_cov_list = []
    pred_px_cov_list = []
    IoU_list = []
    f1_score_list = []
    image_dimensions = []

    # Lists to store the per GT volume per object statistics
    GT_min_diameter = []
    GT_max_diameter = []
    GT_mean_diameter = []
    GT_median_diameter = []
    GT_volume_list = []
    GT_volume_filled_list = []
    GT_surface_area_list = []

    # Lists to store the per prediction volume per object statistics
    pred_min_diameter = []
    pred_max_diameter = []
    pred_mean_diameter = []
    pred_median_diameter = []
    pred_volume_list = []
    pred_volume_filled_list = []
    pred_surface_area_list = []

    # Lists to store the per volume statistics
    file_for_count = []
    folder_for_count = []
    true_positives_count = []
    false_negatives_count = []
    false_positives_count = []
    GT_count_count = []
    pred_count_count = []

    # Loop through the parent folders
    for GP_folder in sorted(sampling_dir_list, reverse=True):
        # Create the path variable to the GT and Prediction folders
        GT_path = os.path.join(directory, GP_folder, "GT")
        pred_path = os.path.join(directory, GP_folder, "Prediction")

        if not os.path.exists(GT_path) or not os.path.exists(pred_path):
            continue

        # Create results sub-folder if it doesn't exist
        res_pred_dir = os.path.join(result_dir, GP_folder)
        if not os.path.exists(res_pred_dir):
            os.mkdir(res_pred_dir)

        # Get the list of GT and Prediction .tif files
        GT_file_list = [file for file in os.listdir(GT_path) if file.endswith(".tif")]
        pred_file_list = [file for file in os.listdir(pred_path) if file.endswith(".tif")]

        # Get the list of the paired files
        paired_files = list(set(GT_file_list) & set(pred_file_list))

        # Loop through the paired files
        for file in paired_files:
            GT_vol = ski.io.imread(os.path.join(GT_path, file))
            pred_vol = ski.io.imread(os.path.join(pred_path, file))
            start_time = perf_counter()

            # Check if the shape of GT is bigger than Prediction and pad if so
            if GT_vol.shape > pred_vol.shape:
                print(
                    f"{file} from {GP_folder} has shape {GT_vol.shape} in "
                    f"GT and {pred_vol.shape} in Prediction. Padded "
                    f"Prediction to match GT shape."
                )
                pred_vol = pad_br_with_zeroes_3d(GT_vol, pred_vol)
            image_dimensions.append(list(GT_img.shape))
            # Check if the shape of the GT and Prediction volumes are the same
            if GT_vol.shape == pred_vol.shape:
                # Relabel to consecutive labels
                GT_remap, _, _ = ski.segmentation.relabel_sequential(GT_vol)
                pred_remap, _, _ = ski.segmentation.relabel_sequential(pred_vol)

                GT_count = int(np.max(GT_remap))
                pred_count = int(np.max(pred_remap))

                print(
                    f"{file} from {GP_folder} has {GT_count} objects in GT "
                    f"and {pred_count} objects in Prediction"
                )

                # Get matching labels between GT and Prediction
                # Expected: iterable of (gt_lbl, pred_lbl, iou_score)
                raw_matches = list(find_matching_labels_3d(GT_remap, pred_remap))

                # Filter by IoU threshold (and exclude background if present)
                matching_labels = [
                    (gt_lbl, pred_lbl, iou)
                    for (gt_lbl, pred_lbl, iou) in raw_matches
                    if gt_lbl != 0 and pred_lbl != 0 and iou >= iou_threshold
                ]

                # Make lookup dictionaries for fast access
                # For each GT label, get best matching pred label (highest IoU)
                gt_to_best = {}
                for gt_lbl, pred_lbl, iou in matching_labels:
                    if gt_lbl not in gt_to_best or iou > gt_to_best[gt_lbl][1]:
                        gt_to_best[gt_lbl] = (pred_lbl, iou)

                used_pred_labels = set()

                # Initialize voxel maps
                true_positives = np.zeros_like(GT_vol, dtype=GT_remap.dtype)
                false_positives = np.zeros_like(GT_vol, dtype=GT_remap.dtype)
                false_negatives = np.zeros_like(GT_vol, dtype=GT_remap.dtype)

                # Per-object loop over GT labels (excluding background)
                for gt_lbl in range(1, GT_count + 1):
                    GT_obj = GT_remap == gt_lbl

                    # Calculate GT object stats
                    GT_voxel_coverage = voxel_coverage_percent_3d(GT_obj)
                    gt_min_d, gt_max_d, gt_mean_d, gt_median_d = object_diameter_3d(GT_obj)
                    gt_volume = GT_obj.sum()
                    gt_volume_filled = ndimage.binary_fill_holes(GT_obj).sum()
                    gt_surface_area = surface_area_3d(GT_obj)

                    # Add GT info to lists
                    GP_folder_list.append(GP_folder)
                    file_name_list.append(file)
                    GT_label_list.append(gt_lbl)
                    GT_px_cov_list.append(GT_voxel_coverage)
                    GT_min_diameter.append(gt_min_d)
                    GT_max_diameter.append(gt_max_d)
                    GT_mean_diameter.append(gt_mean_d)
                    GT_median_diameter.append(gt_median_d)
                    GT_volume_list.append(gt_volume)
                    GT_volume_filled_list.append(gt_volume_filled)
                    GT_surface_area_list.append(gt_surface_area)

                    # Case 1: there is no acceptable match for this GT label
                    if gt_lbl not in gt_to_best or pred_count == 0:
                        pred_label_list.append(0)
                        pred_px_cov_list.append(0.0)
                        IoU_list.append(0.0)
                        f1_score_list.append(0.0)
                        pred_min_diameter.append(np.nan)
                        pred_max_diameter.append(np.nan)
                        pred_mean_diameter.append(np.nan)
                        pred_median_diameter.append(np.nan)
                        pred_volume_list.append(np.nan)
                        pred_volume_filled_list.append(np.nan)
                        pred_surface_area_list.append(np.nan)

                        # Mark this GT object as false negative
                        false_negatives[GT_obj] = gt_lbl
                        continue

                    # Case 2: matched prediction label, treat as TP
                    pred_lbl, iou_score = gt_to_best[gt_lbl]
                    used_pred_labels.add(pred_lbl)

                    pred_obj = pred_remap == pred_lbl

                    # F1 score in 3D: flatten to 1D
                    f1_score = skl.f1_score(
                        GT_obj.flatten(), pred_obj.flatten(), average="micro"
                    )

                    # Prediction stats
                    pred_voxel_coverage = voxel_coverage_percent_3d(pred_obj)
                    (
                        pred_min_d,
                        pred_max_d,
                        pred_mean_d,
                        pred_median_d,
                    ) = object_diameter_3d(pred_obj)
                    pred_volume = pred_obj.sum()
                    pred_volume_filled = ndimage.binary_fill_holes(pred_obj).sum()
                    pred_surface = surface_area_3d(pred_obj)

                    pred_label_list.append(pred_lbl)
                    pred_px_cov_list.append(pred_voxel_coverage)
                    IoU_list.append(iou_score)
                    f1_score_list.append(f1_score)
                    pred_min_diameter.append(pred_min_d)
                    pred_max_diameter.append(pred_max_d)
                    pred_mean_diameter.append(pred_mean_d)
                    pred_median_diameter.append(pred_median_d)
                    pred_volume_list.append(pred_volume)
                    pred_volume_filled_list.append(pred_volume_filled)
                    pred_surface_area_list.append(pred_surface)

                    # Mark TP voxels
                    true_positives[pred_obj] = gt_lbl

                # After processing all GT labels, mark FP as any prediction object
                # label that was never used in a TP pair.
                for pl in range(1, pred_count + 1):
                    if pl not in used_pred_labels:
                        fp_mask = pred_remap == pl
                        false_positives[fp_mask] = pl

                # Save the volumes
                ski.io.imsave(
                    os.path.join(
                        res_pred_dir,
                        file.split(".")[0] + "_true_positives.tif",
                    ),
                    true_positives,
                    check_contrast=False,
                )

                ski.io.imsave(
                    os.path.join(
                        res_pred_dir,
                        file.split(".")[0] + "_false_negatives.tif",
                    ),
                    false_negatives,
                    check_contrast=False,
                )

                ski.io.imsave(
                    os.path.join(
                        res_pred_dir,
                        file.split(".")[0] + "_false_positives.tif",
                    ),
                    false_positives,
                    check_contrast=False,
                )

                # Get summary statistics
                file_for_count.append(file)
                folder_for_count.append(GP_folder)
                true_positives_count.append(len(np.unique(true_positives)) - 1)
                false_negatives_count.append(len(np.unique(false_negatives)) - 1)
                false_positives_count.append(len(np.unique(false_positives)) - 1)
                GT_count_count.append(GT_count)
                pred_count_count.append(pred_count)

            else:
                print(
                    f"Error: {file} has different shape in GT and "
                    f"Prediction folders."
                )

            print(
                f"Elapsed time: "
                f"{strftime('%H:%M:%S', gmtime(perf_counter() - start_time))}"
            )

    # Store Object properties in a dataframe
    IoU_per_obj_df["Grand_Parent_Folder"] = GP_folder_list
    IoU_per_obj_df["File_name"] = file_name_list
    IoU_per_obj_df["GT_Label"] = GT_label_list
    IoU_per_obj_df["Prediction_Label"] = pred_label_list
    IoU_per_obj_df["GT_Voxel_Coverage_Percent"] = GT_px_cov_list
    IoU_per_obj_df["Prediction_Voxel_Coverage_Percent"] = pred_px_cov_list
    IoU_per_obj_df["IoU"] = IoU_list
    IoU_per_obj_df["f1_score"] = f1_score_list

    # Store GT volume properties in a dataframe
    IoU_per_obj_df["GT_diameter_min"] = GT_min_diameter
    IoU_per_obj_df["GT_diameter_max"] = GT_max_diameter
    IoU_per_obj_df["GT_diameter_mean"] = GT_mean_diameter
    IoU_per_obj_df["GT_diameter_median"] = GT_median_diameter
    IoU_per_obj_df["GT_volume"] = GT_volume_list
    IoU_per_obj_df["GT_volume_filled"] = GT_volume_filled_list
    IoU_per_obj_df["GT_surface_area"] = GT_surface_area_list

    # Dataframe calculations for GT volumes
    IoU_per_obj_df["GT_Sphericity"] = (
        (np.pi ** (1.0 / 3.0))
        * (6 * IoU_per_obj_df["GT_volume"].astype(float)) ** (2.0 / 3.0)
    ) / IoU_per_obj_df["GT_surface_area"].astype(float)
    IoU_per_obj_df["GT_Filledness"] = IoU_per_obj_df["GT_volume"].astype(float) / (
        IoU_per_obj_df["GT_volume_filled"].astype(float)
    )

    # Store Prediction volume properties in a dataframe
    IoU_per_obj_df["pred_diameter_min"] = pred_min_diameter
    IoU_per_obj_df["pred_diameter_max"] = pred_max_diameter
    IoU_per_obj_df["pred_diameter_mean"] = pred_mean_diameter
    IoU_per_obj_df["pred_diameter_median"] = pred_median_diameter
    IoU_per_obj_df["pred_volume"] = pred_volume_list
    IoU_per_obj_df["pred_volume_filled"] = pred_volume_filled_list
    IoU_per_obj_df["pred_surface_area"] = pred_surface_area_list

    # Dataframe calculations for Prediction volumes
    IoU_per_obj_df["pred_Sphericity"] = (
        (np.pi ** (1.0 / 3.0))
        * (6 * IoU_per_obj_df["pred_volume"].astype(float)) ** (2.0 / 3.0)
    ) / IoU_per_obj_df["pred_surface_area"].astype(float)
    IoU_per_obj_df["pred_Filledness"] = IoU_per_obj_df["pred_volume"].astype(
        float
    ) / IoU_per_obj_df["pred_volume_filled"].astype(float)

    # Summary statistics per file
    if len(IoU_per_obj_df) > 0:
        summary_df = (
            IoU_per_obj_df.groupby(["Grand_Parent_Folder", "File_name"])
            .agg("mean")
            .reset_index()
        )
        summary_df.drop(["GT_Label", "Prediction_Label"], axis=1, inplace=True)
        summary_df["Dimensions"] = image_dimensions

        count_df["Grand_Parent_Folder"] = folder_for_count
        count_df["File_name"] = file_for_count
        count_df["GT_count"] = GT_count_count
        count_df["pred_count"] = pred_count_count
        count_df["true_positives_count"] = true_positives_count
        count_df["false_negatives_count"] = false_negatives_count
        count_df["false_positives_count"] = false_positives_count

        summary_df = summary_df.merge(
            count_df, on=["Grand_Parent_Folder", "File_name"], how="left"
        )

        # Calculate summary Sensitivity/Recall and Accuracy
        summary_df["Sensitivity"] = summary_df["true_positives_count"] / (
            summary_df["true_positives_count"]
            + summary_df["false_negatives_count"]
        )
        summary_df["Accuracy"] = summary_df["true_positives_count"] / (
            summary_df["true_positives_count"]
            + summary_df["false_positives_count"]
            + summary_df["false_negatives_count"]
        )

        # Save summary statistics in csv file
        summary_df.to_csv(
            os.path.join(
                result_dir, directory.split(os.sep)[-1] + "_summary_stats.csv"
            )
        )

        # Save IoU per object statistics in csv file
        IoU_per_obj_df.to_csv(
            os.path.join(
                result_dir, directory.split(os.sep)[-1] + "_IoU_per_obj_stats.csv"
            )
        )

        print("Done.")

    return summary_df, IoU_per_obj_df



def semantic_statistics_3d(
    directory: str,
    result_dir: str,
    sampling_dir_list: Optional[List[str]] = [
        "upsampling_16",
        "upsampling_8",
        "upsampling_4",
        "upsampling_2",
        "OG",
        "downsampling_2",
        "downsampling_4",
        "downsampling_8",
        "downsampling_16",
    ],
) -> pd.DataFrame:
    """
    Calculate the IoU, f1 score, and other statistics for each label in the
    semantic segmentation GT and Prediction 3D volumes. Only for 2 labels +
    background.

    Args:
        directory (str): Directory with folders of sampling folders with GT
                        and Prediction folder pairs inside.
        result_dir (str): Directory to save the results.

    Returns:
        pd.DataFrame: A dataframe containing per semantic label and average
                     IoU and f1 score statistics.
    """

    # Create dataframes to store the results
    IoU_per_SS_df = pd.DataFrame([])
    start_time = None

    # Lists to store all the data
    GP_folder_list = []
    file_name_list = []
    GT_label_list = []
    pred_label_list = []
    IoU_list = []
    f1_score_list = []

    # Loop through the parent folders
    for GP_folder in sorted(sampling_dir_list, reverse=True):
        # Create the path variable to the GT and Prediction folders
        GT_path = os.path.join(directory, GP_folder, "GT")
        pred_path = os.path.join(directory, GP_folder, "Prediction")

        # Check if the GT and Prediction folders exist
        if not os.path.exists(GT_path) or not os.path.exists(pred_path):
            continue

        # Create results sub-folder if it doesn't exist
        res_pred_dir = os.path.join(result_dir, GP_folder)
        if not os.path.exists(res_pred_dir):
            os.mkdir(res_pred_dir)

        # Get the list of GT and Prediction .tif files
        GT_file_list = [
            file for file in os.listdir(GT_path) if file.endswith(".tif")
        ]
        pred_file_list = [
            file for file in os.listdir(pred_path) if file.endswith(".tif")
        ]

        # Get the list of the paired files
        paired_files = list(set(GT_file_list) & set(pred_file_list))

        # Loop through the paired files
        for file in paired_files:
            print(
                f"Calculating semantic segmentation statistics for {file} in {GP_folder}"
            )
            GT_vol = ski.io.imread(os.path.join(GT_path, file))
            pred_vol = ski.io.imread(os.path.join(pred_path, file))
            start_time = perf_counter()

            # Check if shape of GT is bigger than Prediction and pad if so
            if GT_vol.shape > pred_vol.shape:
                print(
                    f"{file} from {GP_folder} has shape {GT_vol.shape} in "
                    f"GT and {pred_vol.shape} in Prediction. Padded "
                    f"Prediction to match GT shape."
                )
                pred_vol = pad_br_with_zeroes_3d(GT_vol, pred_vol)

            # Check if the shape of the GT and Prediction volumes are same
            if GT_vol.shape == pred_vol.shape:
                # Get unique labels that exist in both GT and Prediction
                GT_labels = np.unique(GT_vol)
                pred_labels = np.unique(pred_vol)

                # Only compute for labels that exist in both
                # (semantic segmentation assumes matching label values)
                common_labels = np.intersect1d(GT_labels, pred_labels)

                # Calculate IoU and f1 score for each matching label
                for lbl in common_labels:
                    # Create binary masks for current label
                    GT_mask = GT_vol == lbl
                    pred_mask = pred_vol == lbl

                    # Calculate IoU
                    intersection = np.sum(GT_mask & pred_mask)
                    union = np.sum(GT_mask | pred_mask)
                    iou_score = intersection / union if union > 0 else 0.0

                    # Calculate F1 score (faster than sklearn for binary)
                    tp = intersection
                    fp = np.sum(pred_mask & ~GT_mask)
                    fn = np.sum(GT_mask & ~pred_mask)
                    f1_score = (
                        (2 * tp) / (2 * tp + fp + fn)
                        if (2 * tp + fp + fn) > 0
                        else 0.0
                    )

                    # Add to lists
                    GP_folder_list.append(GP_folder)
                    file_name_list.append(file)
                    GT_label_list.append(lbl)
                    pred_label_list.append(lbl)
                    IoU_list.append(iou_score)
                    f1_score_list.append(f1_score)

                # Calculate overall statistics (all labels combined)
                # Use direct computation instead of sklearn for speed
                GT_flat = GT_vol.ravel()
                pred_flat = pred_vol.ravel()

                # Overall IoU (Jaccard)
                intersection_all = np.sum(GT_flat == pred_flat)
                total = GT_flat.size
                iou_score_all = intersection_all / total

                # Overall F1 (micro-averaged)
                f1_score_all = skl.f1_score(
                    GT_flat, pred_flat, average="micro"
                )

                # Add to lists
                GP_folder_list.append(GP_folder)
                file_name_list.append(file)
                GT_label_list.append("ALL")
                pred_label_list.append("ALL")
                IoU_list.append(iou_score_all)
                f1_score_list.append(f1_score_all)

            else:
                print(
                    f"Error: {file} has different shape in GT and "
                    f"Prediction folders."
                )

            print(
                f"Elapsed time: "
                f"{strftime('%H:%M:%S', gmtime(perf_counter() - start_time))}"
            )

    # Store Object properties in a dataframe
    IoU_per_SS_df["Grand_Parent_Folder"] = GP_folder_list
    IoU_per_SS_df["File_name"] = file_name_list
    IoU_per_SS_df["GT_Label"] = GT_label_list
    IoU_per_SS_df["Prediction_Label"] = pred_label_list
    IoU_per_SS_df["IoU"] = IoU_list
    IoU_per_SS_df["f1_score"] = f1_score_list

    # Check if the dataframe is empty if not save the results as csv
    if len(GP_folder_list) != 0:
        # Save semantic segmentation statistics in csv file
        IoU_per_SS_df.to_csv(
            os.path.join(
                result_dir,
                directory.split(os.sep)[-1] + "_semantic_stats.csv",
            )
        )

        print("Done.")

    else:
        print("No semantic segmentation files found.")

    return IoU_per_SS_df


def binary_mask_statistics_3d(
    directory: str,
    result_dir: str,
    sampling_dir_list: Optional[List[str]] = [
        "upsampling_16",
        "upsampling_8",
        "upsampling_4",
        "upsampling_2",
        "OG",
        "downsampling_2",
        "downsampling_4",
        "downsampling_8",
        "downsampling_16",
    ],
    max_samples_to_save: int = 3,
) -> pd.DataFrame:
    """
    Calculate the IoU, f1 score, and other statistics for a binary mask 3D
    volume from the semantic segmentation GT and Prediction volumes.

    In addition to IoU and F1, this function computes voxel-level
    true positives (TP), false positives (FP), and false negatives (FN)
    for each volume, and saves TP/FP/FN volumes for up to
    `max_samples_to_save` examples per sampling folder.

    Args
    ----
    directory : str
        Directory with folders of sampling folders with GT and Prediction
        folder pairs inside.
    result_dir : str
        Directory to save the results.
    sampling_dir_list : list of str, optional
        List of sampling folders to analyse.
    max_samples_to_save : int, optional
        Maximum number of volumes per sampling folder for which TP/FP/FN
        3D masks are saved.

    Returns
    -------
    IoU_per_BN_df : pd.DataFrame
        A dataframe containing the binary mask IoU, f1 score and
        TP/FP/FN counts per volume.
    """
    # Create dataframes to store the results
    IoU_per_BN_df = pd.DataFrame([])
    start_time = None

    # Lists to store all the data
    GP_folder_list = []
    file_name_list = []
    GT_label_list = []
    pred_label_list = []
    IoU_list = []
    f1_score_list = []

    # New: voxel-level confusion counts
    TP_vox_list = []
    FP_vox_list = []
    FN_vox_list = []

    # Track how many examples we have saved per sampling folder
    saved_examples_per_folder: Dict[str, int] = {}

    # Loop through the parent folders
    for GP_folder in sorted(sampling_dir_list):
        # Create the path variable to the GT and Prediction folders
        GT_path = os.path.join(directory, GP_folder, "GT")
        pred_path = os.path.join(directory, GP_folder, "Prediction")

        # Check if the GT and Prediction folders exist
        if not os.path.exists(GT_path) or not os.path.exists(pred_path):
            continue

        # Create results sub-folder if it doesn't exist
        res_pred_dir = os.path.join(result_dir, GP_folder)
        if not os.path.exists(res_pred_dir):
            os.mkdir(res_pred_dir)

        # Get the list of GT and Prediction .tif files
        GT_file_list = [file for file in os.listdir(GT_path) if file.endswith(".tif")]
        pred_file_list = [file for file in os.listdir(pred_path) if file.endswith(".tif")]

        # Get the list of the paired files
        paired_files = list(set(GT_file_list) & set(pred_file_list))

        # Init saved counter for this folder
        saved_examples_per_folder.setdefault(GP_folder, 0)

        # Loop through the paired files
        for file in paired_files:
            GT_vol = ski.io.imread(os.path.join(GT_path, file))
            pred_vol = ski.io.imread(os.path.join(pred_path, file))
            start_time = perf_counter()

            # Check if shape of GT is bigger than Prediction and pad if so
            if GT_vol.shape > pred_vol.shape:
                print(
                    f"{file} from {GP_folder} has shape {GT_vol.shape} in "
                    f"GT and {pred_vol.shape} in Prediction. Padded "
                    f"Prediction to match GT shape."
                )
                pred_vol = pad_br_with_zeroes_3d(GT_vol, pred_vol)

            # Check if the shape of the GT and Prediction volumes are same
            if GT_vol.shape == pred_vol.shape:
                # Convert to binary masks
                GT_binary = GT_vol > 0
                pred_binary = pred_vol > 0

                # Calculate IoU
                intersection = np.logical_and(GT_binary, pred_binary)
                union = np.logical_or(GT_binary, pred_binary)
                iou_score = np.sum(intersection) / np.sum(union)

                # Calculate F1 score
                f1_score = skl.f1_score(
                    GT_binary.flatten(), pred_binary.flatten(), average="micro"
                )

                # Voxel-level confusion:
                # TP: GT = 1, Pred = 1
                # FP: GT = 0, Pred = 1
                # FN: GT = 1, Pred = 0
                TP_mask = GT_binary & pred_binary
                FP_mask = (~GT_binary) & pred_binary
                FN_mask = GT_binary & (~pred_binary)

                TP_vox = int(TP_mask.sum())
                FP_vox = int(FP_mask.sum())
                FN_vox = int(FN_mask.sum())

                # Add to lists
                GP_folder_list.append(GP_folder)
                file_name_list.append(file)
                GT_label_list.append("BINARY")
                pred_label_list.append("BINARY")
                IoU_list.append(iou_score)
                f1_score_list.append(f1_score)
                TP_vox_list.append(TP_vox)
                FP_vox_list.append(FP_vox)
                FN_vox_list.append(FN_vox)

                # Save example TP/FP/FN volumes for a few samples
                if saved_examples_per_folder[GP_folder] < max_samples_to_save:
                    base_name = file.split(".")[0]
                    ski.io.imsave(
                        os.path.join(
                            res_pred_dir, base_name + "_TP_binary.tif"
                        ),
                        TP_mask.astype(np.uint8),
                        check_contrast=False,
                    )
                    ski.io.imsave(
                        os.path.join(
                            res_pred_dir, base_name + "_FP_binary.tif"
                        ),
                        FP_mask.astype(np.uint8),
                        check_contrast=False,
                    )
                    ski.io.imsave(
                        os.path.join(
                            res_pred_dir, base_name + "_FN_binary.tif"
                        ),
                        FN_mask.astype(np.uint8),
                        check_contrast=False,
                    )
                    saved_examples_per_folder[GP_folder] += 1

            else:
                print(
                    f"Error: {file} has different shape in GT and "
                    f"Prediction folders."
                )

            if start_time is not None:
                print(
                    f"Elapsed time: "
                    f"{strftime('%H:%M:%S', gmtime(perf_counter() - start_time))}"
                )

    # Store results in dataframe
    IoU_per_BN_df["Grand_Parent_Folder"] = GP_folder_list
    IoU_per_BN_df["File_name"] = file_name_list
    IoU_per_BN_df["GT_Label"] = GT_label_list
    IoU_per_BN_df["Prediction_Label"] = pred_label_list
    IoU_per_BN_df["IoU"] = IoU_list
    IoU_per_BN_df["f1_score"] = f1_score_list
    IoU_per_BN_df["TP_voxels"] = TP_vox_list
    IoU_per_BN_df["FP_voxels"] = FP_vox_list
    IoU_per_BN_df["FN_voxels"] = FN_vox_list

    # If dataframe is not empty save the results as csv
    if len(GP_folder_list) != 0:
        IoU_per_BN_df.to_csv(
            os.path.join(
                result_dir, directory.split(os.sep)[-1] + "_binary_mask_stats.csv"
            )
        )
        print("Done.")
    else:
        print("No binary mask files found.")

    return IoU_per_BN_df

## Miscellaneous functions


def voxel_coverage_percent_3d(vol_array: np.ndarray) -> float:
    """
    Calculate the voxel coverage percentage of the input 3D volume array,
    how much of the whole object is covered by a single voxel.

    Args:
        vol_array (np.ndarray): A numpy array volume with a single object
                               label.

    Returns:
        voxel_coverage_percent (float): The percentage of the object that each
                                       voxel covers, as a float.
    """

    # Calculate the percentage of the object that each voxel covers
    voxel_coverage = (1 / np.count_nonzero(vol_array)) * 100

    return voxel_coverage


def bbox_points_for_crop_3d(
    bbox: List[int], zmax: int, xmax: int, ymax: int
) -> Tuple[int, int, int, int, int, int]:
    """
    Using the bounding box coordinates for each object, new bbox coordinates
    for the padded crop region are calculated for 3D volumes.

    Args:
        bbox (List[int]): A list containing the z, x, and y coordinates of the
                         top left and bottom right points of the bounding box.
        zmax (int): The maximum z value of the volume.
        xmax (int): The maximum x value of the volume.
        ymax (int): The maximum y value of the volume.

    Returns:
        Tuple[int, int, int, int, int, int]: A tuple containing the z, x, and
                                             y coordinates of the top left and
                                             bottom right points of the bbox.
    """
    # Unpack the bounding box coordinates
    z1, x1, y1, z2, x2, y2 = bbox

    # Calculate half the edge length of the box for padding
    z_radius = (z2 - z1 + 2) // 2
    x_radius = (x2 - x1 + 2) // 2
    y_radius = (y2 - y1 + 2) // 2

    # Calculate new bounding box coordinates for padded crop region
    # Top Left
    z1 = (z1 - z_radius) if (z1 - z_radius) > 0 else 0
    x1 = (x1 - x_radius) if (x1 - x_radius) > 0 else 0
    y1 = (y1 - y_radius) if (y1 - y_radius) > 0 else 0

    # Bottom Right
    z2 = (z2 + z_radius) if (z2 + z_radius) < zmax else zmax
    x2 = (x2 + x_radius) if (x2 + x_radius) < xmax else xmax
    y2 = (y2 + y_radius) if (y2 + y_radius) < ymax else ymax

    return z1, x1, y1, z2, x2, y2


def object_diameter_3d(
    volume_array: np.array,
) -> Tuple[float, float, float, float]:
    """
    Calculate the diameter of the object in the 3D volume array.

    Args:
        volume_array: A numpy array volume with a single object

    Returns:
        min_diameter: The minimum diameter of the object in the volume array.
        max_diameter: The maximum diameter of the object in the volume array.
        mean_diameter: The mean diameter of the object in the volume array.
        median_diameter: The median diameter of the object in the volume array.
    """
    # Calculate the object skeleton and Euclidean distance transform
    obj_skeleton = ski.morphology.skeletonize(volume_array)
    obj_edt = ndimage.distance_transform_edt(volume_array)

    # Get the EDT values for the object skeleton
    obj_skeleton_edt = obj_skeleton * obj_edt

    # Get non-zero skeleton EDT values
    skeleton_values = obj_skeleton_edt[np.nonzero(obj_skeleton_edt)]

    # Check if skeleton is empty (can happen with very small objects)
    if len(skeleton_values) == 0:
        # Fallback: use EDT directly
        edt_values = obj_edt[np.nonzero(obj_edt)]
        if len(edt_values) == 0:
            # Object is empty or single voxel
            return 0.0, 0.0, 0.0, 0.0
        min_diameter = np.min(edt_values) * 2
        max_diameter = np.max(edt_values) * 2
        mean_diameter = np.mean(edt_values) * 2
        median_diameter = np.median(edt_values) * 2
    else:
        # Calculate min, max, mean, and median radius excluding zero values
        # of background, multiply by 2 for diameter
        min_diameter = np.min(skeleton_values) * 2
        max_diameter = np.max(skeleton_values) * 2
        mean_diameter = np.mean(skeleton_values) * 2
        median_diameter = np.median(skeleton_values) * 2

    return min_diameter, max_diameter, mean_diameter, median_diameter


def surface_area_3d(volume_array: np.array) -> float:
    """
    Calculate the surface area of a 3D object using marching cubes.

    Args:
        volume_array: A numpy array volume with a single object

    Returns:
        surface_area: The surface area of the object
    """
    try:
        # Use marching cubes to get surface mesh
        verts, faces, _, _ = ski.measure.marching_cubes(
            volume_array.astype(float), level=0.5
        )

        # Calculate surface area from mesh
        surface_area = ski.measure.mesh_surface_area(verts, faces)

        return surface_area
    except Exception:
        # If marching cubes fails, return NaN
        return np.nan


def pad_br_with_zeroes_3d(gt_vol: np.array, pred_vol: np.array) -> np.array:
    """
    Calculate the padding size between the GT and Prediction 3D volumes.

    Args:
        gt_vol: A numpy array containing the GT volume.
        pred_vol: A numpy array containing the Prediction volume.

    Returns:
        padded_pred: The Prediction volume padded with zeroes to match the
                    GT volume size.
    """
    # Pad the Prediction volume with zeroes to match the GT volume shape
    padded_pred = np.pad(
        pred_vol,
        (
            (0, gt_vol.shape[0] - pred_vol.shape[0]),
            (0, gt_vol.shape[1] - pred_vol.shape[1]),
            (0, gt_vol.shape[2] - pred_vol.shape[2]),
        ),
        "constant",
        constant_values=0,
    )

    return padded_pred


## Region properties functions for 3D


def object_props_3d(
    directory: str,
    properties: Optional[List[str]] = [
        "label",
        "area",
        "equivalent_diameter_area",
        "axis_major_length",
        "axis_minor_length",
        "extent",
    ],
    spacing: Optional[Tuple[float, float, float]] = None,
    folder_sampling_dict: Optional[Dict[str, float]] = {
        "upsampling_16": 16,
        "upsampling_8": 8,
        "upsampling_4": 4,
        "upsampling_2": 2,
        "OG": 1,
        "downsampling_2": 1 / 2,
        "downsampling_4": 1 / 4,
        "downsampling_8": 1 / 8,
        "downsampling_16": 1 / 16,
    },
) -> pd.DataFrame:
    """
    Calculate the properties for each object in each 3D volume in the input
    directory.

    Args:
        directory (str): The input directory containing the volume files.
        properties (List[str]): A list of properties to calculate.
        spacing (Optional[Tuple[float, float, float]]): The spacing between
                                                        voxels in z, y, x.
        folder_sampling_dict (Dict[str, float]): Dictionary mapping folder
                                                names to sampling multipliers.

    Returns:
        pd.DataFrame: A dataframe containing the properties for each object.
    """

    # Initialize empty dataframe
    IoU_per_obj_df = pd.DataFrame([])

    # Walk through the directory structure
    for root, dirs, files in os.walk(directory):
        for given_dir in dirs:
            for file in os.listdir(os.path.join(root, given_dir)):
                if file.endswith(".tif"):
                    # Read the volume
                    label_vol = ski.io.imread(
                        os.path.join(root, given_dir, file)
                    )

                    # Calculate region properties
                    props_df = region_properties_3d(
                        label_vol, properties, spacing
                    )

                    # Add file name and parent folder information
                    props_df = add_file_name_to_dataframe(file, props_df)
                    props_df = add_parent_folder(
                        props_df, given_dir, root, folder_sampling_dict
                    )

                    # Add extra properties
                    props_df = extra_properties(props_df)

                    # Normalize to sampling
                    props_df = normalize_to_sampling(props_df, properties)

                    # Concatenate to main dataframe
                    IoU_per_obj_df = pd.concat(
                        [IoU_per_obj_df, props_df], ignore_index=True
                    )

    return IoU_per_obj_df


def region_properties_3d(
    label_volume: np.ndarray,
    properties: List[str] = [
        "label",
        "area",
        "equivalent_diameter_area",
        "axis_major_length",
        "axis_minor_length",
        "extent",
    ],
    spacing: Optional[Tuple[float, float, float]] = None,
) -> pd.DataFrame:
    """
    Calculate the region properties for each object in a 3D label volume.

    Args:
        label_volume (np.ndarray): A 3D numpy array containing labeled objects.
        properties (List[str]): A list of properties to calculate.
        spacing (Optional[Tuple[float, float, float]]): The spacing between
                                                        voxels in z, y, x.

    Returns:
        pd.DataFrame: A dataframe containing the properties for each object.
    """

    # Calculate region properties
    if spacing is not None:
        props_df = pd.DataFrame(
            ski.measure.regionprops_table(
                label_volume, properties=properties, spacing=spacing
            )
        )
    else:
        props_df = pd.DataFrame(
            ski.measure.regionprops_table(label_volume, properties=properties)
        )

    return props_df


def add_file_name_to_dataframe(
    file: str, IoU_per_obj_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add the file name to the dataframe.

    Args:
        file (str): The file name.
        IoU_per_obj_df (pd.DataFrame): The dataframe to add the file name to.

    Returns:
        pd.DataFrame: The dataframe with the file name added.
    """

    IoU_per_obj_df["File_name"] = file

    return IoU_per_obj_df


def extra_properties(IoU_per_obj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate extra properties for the dataframe.

    Args:
        IoU_per_obj_df (pd.DataFrame): The dataframe to calculate extra
                                       properties for.

    Returns:
        pd.DataFrame: The dataframe with extra properties added.
    """

    # Calculate circularity if perimeter exists (2D)
    if "perimeter" in IoU_per_obj_df.columns:
        IoU_per_obj_df["Circularity"] = (
            4 * np.pi * IoU_per_obj_df["area"].astype(float)
        ) / IoU_per_obj_df["perimeter"].astype(float) ** 2

    # Calculate filledness if area_filled exists
    if "area_filled" in IoU_per_obj_df.columns:
        IoU_per_obj_df["Filledness"] = IoU_per_obj_df["area"].astype(
            float
        ) / IoU_per_obj_df["area_filled"].astype(float)

    return IoU_per_obj_df


def add_parent_folder(
    IoU_per_obj_df: pd.DataFrame,
    given_dir: str,
    root: str,
    folder_sampling_dict: Dict[str, float],
) -> pd.DataFrame:
    """
    Add the parent folder information to the dataframe.

    Args:
        IoU_per_obj_df (pd.DataFrame): The dataframe to add parent folder
                                       info to.
        given_dir (str): The directory name.
        root (str): The root path.
        folder_sampling_dict (Dict[str, float]): Dictionary mapping folder
                                                names to sampling multipliers.

    Returns:
        pd.DataFrame: The dataframe with parent folder information added.
    """

    # Get the parent folder name
    parent_folder = os.path.basename(
        os.path.dirname(os.path.join(root, given_dir))
    )

    # Add parent folder to dataframe
    IoU_per_obj_df["Parent_Folder"] = parent_folder

    # Add sampling multiplier if available
    if parent_folder in folder_sampling_dict:
        IoU_per_obj_df["Sampling_Multiplier"] = folder_sampling_dict[
            parent_folder
        ]

    return IoU_per_obj_df


def normalize_to_sampling(
    IoU_per_obj_df: pd.DataFrame, properties: List[str]
) -> pd.DataFrame:
    """
    Normalize properties to sampling multiplier.

    Args:
        IoU_per_obj_df (pd.DataFrame): The dataframe to normalize.
        properties (List[str]): The properties that were calculated.

    Returns:
        pd.DataFrame: The dataframe with normalized properties.
    """

    # Check if Sampling_Multiplier exists
    if "Sampling_Multiplier" in IoU_per_obj_df.columns:
        # Normalize area/volume properties
        if "area" in properties:
            IoU_per_obj_df["area_normalized"] = (
                IoU_per_obj_df["area"]
                / IoU_per_obj_df["Sampling_Multiplier"] ** 2
            )

        # Normalize length properties
        length_props = [
            "axis_major_length",
            "axis_minor_length",
            "equivalent_diameter_area",
        ]
        for prop in length_props:
            if prop in properties:
                IoU_per_obj_df[f"{prop}_normalized"] = (
                    IoU_per_obj_df[prop]
                    / IoU_per_obj_df["Sampling_Multiplier"]
                )

    return IoU_per_obj_df
