# Functions for morphology analysis of label images

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
import ast
import math

## Main Function


def morphology(
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
) -> None:
    """
    Calculate the properties for each object in each image in the input directory.

    Args:
        main_directory (str): The input directory containing the sub folders contating the image files.

    Expected file arrangement example:
        +-- main_directory
        |  +-- Saureus
        |  |  +-- OG
        |  |  |  +-- GT
        |  |  |  |  +-- images.tiff
        |  |  |  +-- Prediction
        |  |  |  |  +-- images.tiff
        |  |  +-- downsampling_2
        |  |  |  +-- GT
        |  |  |  |  +-- images.tiff
        |  |  |  +-- Prediction
        |  |  |  |  +-- images.tiff
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

            # Create folder to store results if it doesn't exist, if it exists make new one
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
            per_object_statistics(
                directory=curr_dir,
                result_dir=result_dir,
                sampling_dir_list=sampling_dir_list,
            )

            semantic_statistics(
                directory=curr_dir,
                result_dir=result_dir,
                sampling_dir_list=sampling_dir_list,
            )

            binary_mask_statistics(
                directory=curr_dir,
                result_dir=result_dir,
                sampling_dir_list=sampling_dir_list,
            )

    # Print total time taken
    total_time = strftime("%H:%M:%S", gmtime(perf_counter() - begin_time))
    print(f"Total time: {total_time}")


## Per object Prediction statistics functions


def per_object_statistics(
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the IoU, f1 score, and other statistics for each object in the image.

    Args:
        directory (str): Directory with folders of sampling folders with GT and Prediction folder pairs inside.
        result_dir (str): Directory to save the results.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - The first dataframe contains per object IoU and f1 score statistics, for Ground Truth and Prediction.
            - The second dataframe contains a summary of the statistics per field of view, including Sensitivity and Accuracy, for Ground Truth and Prediction.
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

    # Lists to store the per GT image per object statistics
    GT_min_diameter = []
    GT_max_diameter = []
    GT_mean_diameter = []
    GT_median_diameter = []
    GT_area_list = []
    GT_area_filled_list = []
    GT_perimeter_list = []

    # Lists to store the per prediction image per object statistics
    pred_min_diameter = []
    pred_max_diameter = []
    pred_mean_diameter = []
    pred_median_diameter = []
    pred_area_list = []
    pred_area_filled_list = []
    pred_perimeter_list = []

    # Lists to store the per image statistics
    file_for_count = []
    folder_for_count = []
    true_positives_count = []
    false_negatives_count = []
    false_positives_count = []
    GT_count_count = []
    pred_count_count = []

    # Loop through the parent folders
    for GP_folder in sorted(sampling_dir_list):
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
        GT_file_list = [
            file for file in os.listdir(GT_path) if file.endswith(".tif")
        ]
        pred_file_list = [
            file for file in os.listdir(pred_path) if file.endswith(".tif")
        ]

        # Get the list of the paired files (both GT and Prediction .tif files)
        paired_files = list(set(GT_file_list) & set(pred_file_list))

        # Loop through the paired files
        for file in paired_files:
            GT_img = ski.io.imread(os.path.join(GT_path, file))
            pred_img = ski.io.imread(os.path.join(pred_path, file))
            start_time = perf_counter()

            # Check if the shape of the GT is bigger than the Prediction are the same and pad Prediction if not
            if GT_img.shape > pred_img.shape:
                print(
                    f"{file} from {GP_folder} has shape {GT_img.shape} in GT and {pred_img.shape} in Prediction. Padded Prediction to match GT shape."
                )
                pred_img = pad_br_with_zeroes(GT_img, pred_img)

            # Check if the shape of the GT and Prediction images are the same
            if GT_img.shape == pred_img.shape:
                # Calculate the number of objects in each image and remap the labels
                GT_remap, _, _ = ski.segmentation.relabel_sequential(GT_img)
                pred_remap, _, _ = ski.segmentation.relabel_sequential(
                    pred_img
                )
                GT_count = np.max(GT_remap)
                pred_count = np.max(pred_remap)

                # Print the number of objects in each image
                print(
                    f"{file} from {GP_folder} has {GT_count} objects in GT and {pred_count} objects in Prediction"
                )

                # Get Bounding Boxes coords for each GT object
                bbox_list = pd.DataFrame(
                    ski.measure.regionprops_table(
                        GT_remap, properties=["label", "bbox"], spacing=(1, 1)
                    )
                )

                # Initialize the true positives, false positives, and false negatives arrays
                true_positives = np.zeros_like(GT_img)
                false_positives = np.zeros_like(GT_img)
                false_negatives = np.zeros_like(GT_img)

                # Loop through all objects in GT and Prediction
                # For each object in GT
                for obj in range(1, GT_count + 1):
                    # Get the bounding box for the current object
                    bbox_index = bbox_list.loc[
                        bbox_list["label"] == obj
                    ].index[0]
                    bbox = bbox_list.loc[
                        bbox_index, ["bbox-0", "bbox-1", "bbox-2", "bbox-3"]
                    ]

                    # Get the coordinates of the bounding box
                    x1, y1, x2, y2 = bbox_points_for_crop(
                        bbox, bbox["bbox-2"].max(), bbox["bbox-3"].max()
                    )

                    # Copy the remaped GT image and remap the current object in GT to 1 and make all others 0
                    GT_obj = GT_remap[x1:x2, y1:y2]
                    GT_obj = GT_obj == obj

                    # Calculate the pixel coverage and object diameter values
                    GT_pixel_coverage = pixel_coverage_percent(GT_obj)
                    gt_min_w, gt_max_w, gt_mean_w, gt_median_w = (
                        object_diameter(GT_obj)
                    )
                    gt_area = GT_obj.sum()
                    gt_area_filled = ndimage.binary_fill_holes(GT_obj).sum()
                    gt_perimeter = perimeter(GT_obj)

                    # Add object information to lists
                    GP_folder_list.append(GP_folder)
                    file_name_list.append(file)
                    GT_label_list.append(obj)
                    GT_px_cov_list.append(GT_pixel_coverage)
                    GT_min_diameter.append(gt_min_w)
                    GT_max_diameter.append(gt_max_w)
                    GT_mean_diameter.append(gt_mean_w)
                    GT_median_diameter.append(gt_median_w)
                    GT_area_list.append(gt_area)
                    GT_area_filled_list.append(gt_area_filled)
                    GT_perimeter_list.append(gt_perimeter)

                    # If the current object is not present in Prediction set values to 0 or NaN
                    if pred_count == 0:
                        pred_label_list.append(0)
                        pred_px_cov_list.append(0)
                        IoU_list.append(0)
                        f1_score_list.append(0)

                        pred_min_diameter.append(np.nan)
                        pred_max_diameter.append(np.nan)
                        pred_mean_diameter.append(np.nan)
                        pred_median_diameter.append(np.nan)
                        pred_area_list.append(np.nan)
                        pred_area_filled_list.append(np.nan)
                        pred_perimeter_list.append(np.nan)

                        continue

                    # Copy the remaped Prediction image
                    pred_obj_main = pred_remap[x1:x2, y1:y2]

                    # If the current FOV has no objects in Prediction set values to 0 or NaN and add to false negatives
                    if len(np.unique(pred_obj_main)) == 1:
                        false_negatives[GT_remap == obj] = obj

                        # Add object information to lists
                        pred_label_list.append(0)
                        pred_px_cov_list.append(0)
                        IoU_list.append(0)
                        f1_score_list.append(0)

                        pred_min_diameter.append(np.nan)
                        pred_max_diameter.append(np.nan)
                        pred_mean_diameter.append(np.nan)
                        pred_median_diameter.append(np.nan)
                        pred_area_list.append(np.nan)
                        pred_area_filled_list.append(np.nan)
                        pred_perimeter_list.append(np.nan)

                        continue

                    # If there is an object in the current FOV
                    # For each object in the current FOV
                    else:
                        for p_obj in np.unique(pred_obj_main):
                            pred_obj = pred_obj_main == p_obj

                            # Calculate IoU
                            intersection = np.logical_and(GT_obj, pred_obj)
                            union = np.logical_or(GT_obj, pred_obj)
                            iou_score = np.sum(intersection) / np.sum(union)

                            if iou_score > 0.5:
                                # Calculate F1 score
                                f1_score = skl.f1_score(
                                    GT_obj, pred_obj, average="micro"
                                )

                                # Calculate pixel coverage percentage and object diameter for Prediction Label
                                pred_pixel_coverage = pixel_coverage_percent(
                                    pred_obj
                                )
                                (
                                    pred_min_d,
                                    pred_max_d,
                                    pred_mean_d,
                                    pred_median_d,
                                ) = object_diameter(pred_obj)
                                pred_area = pred_obj.sum()
                                pred_area_filled = ndimage.binary_fill_holes(
                                    pred_obj
                                ).sum()
                                pred_perimeter = perimeter(pred_obj)

                                # Add object to the true positives array and remove object from the remaped Prediction image
                                true_positives[pred_remap == p_obj] = obj
                                pred_remap[pred_remap == p_obj] = 0

                                # Add object information to lists
                                pred_label_list.append(p_obj)
                                pred_px_cov_list.append(pred_pixel_coverage)
                                IoU_list.append(iou_score)
                                f1_score_list.append(f1_score)
                                pred_min_diameter.append(pred_min_d)
                                pred_max_diameter.append(pred_max_d)
                                pred_mean_diameter.append(pred_mean_d)
                                pred_median_diameter.append(pred_median_d)
                                pred_area_list.append(pred_area)
                                pred_area_filled_list.append(pred_area_filled)
                                pred_perimeter_list.append(pred_perimeter)

                                # Once a true positive is found, break out of the loop
                                break

                            # If no Prediction object matches the GT object
                            if p_obj == np.unique(pred_obj_main)[-1]:
                                # Add object information to lists
                                pred_label_list.append(0)
                                pred_px_cov_list.append(0)
                                IoU_list.append(0)
                                f1_score_list.append(0)

                                pred_min_diameter.append(np.nan)
                                pred_max_diameter.append(np.nan)
                                pred_mean_diameter.append(np.nan)
                                pred_median_diameter.append(np.nan)
                                pred_area_list.append(np.nan)
                                pred_area_filled_list.append(np.nan)
                                pred_perimeter_list.append(np.nan)

                # Store false positives in the array image
                false_positives[pred_remap != 0] = pred_remap[pred_remap != 0]

                # Save the images
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
                false_negatives_count.append(
                    len(np.unique(false_negatives)) - 1
                )
                false_positives_count.append(
                    len(np.unique(false_positives)) - 1
                )
                GT_count_count.append(GT_count)
                pred_count_count.append(pred_count)

            else:
                print(
                    f"Error: {file} has different shape in GT and Prediction folders."
                )

            print(
                f'Elapsed time: {strftime("%H:%M:%S", gmtime(perf_counter() - start_time))}'
            )

    # Store Object properties in a dataframe
    IoU_per_obj_df["Grand_Parent_Folder"] = GP_folder_list
    IoU_per_obj_df["File_name"] = file_name_list
    IoU_per_obj_df["GT_Label"] = GT_label_list
    IoU_per_obj_df["Prediction_Label"] = pred_label_list
    IoU_per_obj_df["GT_Pixel_Coverage_Percent"] = GT_px_cov_list
    IoU_per_obj_df["Prediction_Pixel_Coverage_Percent"] = pred_px_cov_list
    IoU_per_obj_df["IoU"] = IoU_list
    IoU_per_obj_df["f1_score"] = f1_score_list

    # Store GT image properties in a dataframe
    IoU_per_obj_df["GT_diameter_min"] = GT_min_diameter
    IoU_per_obj_df["GT_diameter_max"] = GT_max_diameter
    IoU_per_obj_df["GT_diameter_mean"] = GT_mean_diameter
    IoU_per_obj_df["GT_diameter_median"] = GT_median_diameter
    IoU_per_obj_df["GT_area"] = GT_area_list
    IoU_per_obj_df["GT_area_filled"] = GT_area_filled_list
    IoU_per_obj_df["GT_perimeter"] = GT_perimeter_list

    # Dataframe calculations for GT images
    IoU_per_obj_df["GT_Circularity"] = (
        4 * np.pi * IoU_per_obj_df["GT_area"].astype(float)
    ) / IoU_per_obj_df["GT_perimeter"].astype(float) ** 2
    IoU_per_obj_df["GT_Filledness"] = IoU_per_obj_df["GT_area"].astype(
        float
    ) / IoU_per_obj_df["GT_area_filled"].astype(float)

    # Store Prediction image properties in a dataframe#
    IoU_per_obj_df["pred_diameter_min"] = pred_min_diameter
    IoU_per_obj_df["pred_diameter_max"] = pred_max_diameter
    IoU_per_obj_df["pred_diameter_mean"] = pred_mean_diameter
    IoU_per_obj_df["pred_diameter_median"] = pred_median_diameter
    IoU_per_obj_df["pred_area"] = pred_area_list
    IoU_per_obj_df["pred_area_filled"] = pred_area_filled_list
    IoU_per_obj_df["pred_perimeter"] = pred_perimeter_list

    # Dataframe calculations for Prediction images
    # Calculate the circularity of the predicted objects
    # Circularity is a measure of how close the shape of an object is to a perfect circle.
    # It is calculated as (4 * π * Area) / Perimeter^2.
    IoU_per_obj_df["pred_Circularity"] = (
        4 * np.pi * IoU_per_obj_df["pred_area"].astype(float)
    ) / IoU_per_obj_df["pred_perimeter"].astype(float) ** 2
    # Calculate the filledness of the predicted objects
    # Filledness is a measure of how much of the object's bounding box is filled by the object.
    # It is calculated as Area / Filled Area.
    IoU_per_obj_df["pred_Filledness"] = IoU_per_obj_df["pred_area"].astype(
        float
    ) / IoU_per_obj_df["pred_area_filled"].astype(float)

    # Dataframe calculations for GT

    # Summary statistics per file
    summary_df = (
        IoU_per_obj_df.groupby(["Grand_Parent_Folder", "File_name"])
        .agg("mean")
        .reset_index()
    )

    summary_df.drop(["GT_Label", "Prediction_Label"], axis=1, inplace=True)

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


def semantic_statistics(
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
    Calculate the IoU, f1 score, and other statistics for each label in the semantic segmentation GT and Prediction images. Only for 2 labels + background.

    Args:
        directory (str): Directory with folders of sampling folders with GT and Prediction folder pairs inside.
        result_dir (str): Directory to save the results.

    Returns:
        pd.DataFrame: A dataframe containing per semantic label and average IoU and f1 score statistics.
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
        GT_file_list = [
            file for file in os.listdir(GT_path) if file.endswith(".tif")
        ]
        pred_file_list = [
            file for file in os.listdir(pred_path) if file.endswith(".tif")
        ]

        # Get the list of the paired files (both GT and Prediction .tif files)
        paired_files = list(set(GT_file_list) & set(pred_file_list))

        # Loop through the paired files
        for file in paired_files:
            GT_img = ski.io.imread(os.path.join(GT_path, file))
            pred_img = ski.io.imread(os.path.join(pred_path, file))
            start_time = perf_counter()

            # Check if the shape of the GT is bigger than the Prediction are the same and pad Prediction if not
            if GT_img.shape > pred_img.shape:
                print(
                    f"{file} from {GP_folder} has shape {GT_img.shape} in GT and {pred_img.shape} in Prediction. Padded Prediction to match GT shape."
                )
                pred_img = pad_br_with_zeroes(GT_img, pred_img)

            # Check if the shape of the GT and Prediction images are the same
            if GT_img.shape == pred_img.shape:
                # Calculate the number of objects in each image and remap the labels
                GT_count = np.max(GT_img)
                pred_count = np.max(pred_img)

                # Check the number of labels in GT and Prediction, if above 2 skip semantic analysis
                if GT_count != 2 or pred_count != 2 or GT_count != pred_count:
                    continue

                # Print the number of objects in each image
                print(
                    f"Calculating Semantic Segmentation statistics for {file} from {GP_folder}."
                )

                # Loop through the 2 labels
                for obj in range(1, 3):
                    # Extract the object from the GT
                    GT_obj = GT_img == obj
                    pred_obj = pred_img == obj

                    # Calculate IoU and f1 score
                    intersection = np.logical_and(GT_obj, pred_obj)
                    union = np.logical_or(GT_obj, pred_obj)
                    iou_score = np.sum(intersection) / np.sum(union)

                    f1_score = skl.f1_score(GT_obj, pred_obj, average="micro")

                    # Add object information to lists
                    GP_folder_list.append(GP_folder)
                    file_name_list.append(file)
                    GT_label_list.append(obj)
                    pred_label_list.append(obj)
                    IoU_list.append(iou_score)
                    f1_score_list.append(f1_score)

                    # Store the IoU and f1 score from label 1
                    if obj == 1:
                        iou_1 = iou_score
                        f1_1 = f1_score

                    if obj == 2:
                        # Add object information to lists
                        GP_folder_list.append(GP_folder)
                        file_name_list.append(file)
                        GT_label_list.append("ALL")
                        pred_label_list.append("ALL")
                        IoU_list.append((iou_1 + iou_score) / 2)
                        f1_score_list.append((f1_1 + f1_score) / 2)

            else:
                print(
                    f"Error: {file} has different shape in GT and Prediction folders."
                )

            if start_time is not None:
                print(
                    f'Elapsed time: {strftime("%H:%M:%S", gmtime(perf_counter() - start_time))}'
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
        # Save IoU per semantic label statistics in csv file
        IoU_per_SS_df.to_csv(
            os.path.join(
                result_dir,
                directory.split(os.sep)[-1]
                + "_semantic_segmentation_stats.csv",
            )
        )

        print("Done.")

        return IoU_per_SS_df

    else:
        print("No semantic segmentation images found.")
        return None


def binary_mask_statistics(
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
    Calculate the IoU, f1 score, and other statistics for a binary mask image from the semantic segmentation GT and Prediction images.

    Args:
        directory (str): Directory with folders of sampling folders with GT and Prediction folder pairs inside.
        result_dir (str): Directory to save the results.

    Returns:
        pd.DataFrame: A dataframe containing the binary mask IoU and f1 score statistics.
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
        GT_file_list = [
            file for file in os.listdir(GT_path) if file.endswith(".tif")
        ]
        pred_file_list = [
            file for file in os.listdir(pred_path) if file.endswith(".tif")
        ]

        # Get the list of the paired files (both GT and Prediction .tif files)
        paired_files = list(set(GT_file_list) & set(pred_file_list))

        # Loop through the paired files
        for file in paired_files:
            GT_img = ski.io.imread(os.path.join(GT_path, file))
            pred_img = ski.io.imread(os.path.join(pred_path, file))
            start_time = perf_counter()

            # Check if the shape of the GT is bigger than the Prediction are the same and pad Prediction if not
            if GT_img.shape > pred_img.shape:
                print(
                    f"{file} from {GP_folder} has shape {GT_img.shape} in GT and {pred_img.shape} in Prediction. Padded Prediction to match GT shape."
                )
                pred_img = pad_br_with_zeroes(GT_img, pred_img)

            # Check if the shape of the GT and Prediction images are the same
            if GT_img.shape == pred_img.shape:
                # Calculate the number of objects in each image and remap the labels
                GT_count = np.max(GT_img)
                pred_count = np.max(pred_img)

                # Check the number of labels in GT and Prediction, if above 2 skip semantic analysis
                if GT_count != 2 or pred_count != 2 or GT_count != pred_count:
                    continue

                # Print the number of objects in each image
                print(
                    f"Calculating Binary Mask statistics for {file} from {GP_folder}."
                )

                # Extract the object from the GT
                GT_obj = GT_img != 0
                pred_obj = pred_img != 0

                # Calculate IoU and f1 score
                intersection = np.logical_and(GT_obj, pred_obj)
                union = np.logical_or(GT_obj, pred_obj)
                iou_score = np.sum(intersection) / np.sum(union)

                f1_score = skl.f1_score(GT_obj, pred_obj, average="micro")

                # Add object information to lists
                GP_folder_list.append(GP_folder)
                file_name_list.append(file)
                GT_label_list.append("mask")
                pred_label_list.append("mask")
                IoU_list.append(iou_score)
                f1_score_list.append(f1_score)

            else:
                print(
                    f"Error: {file} has different shape in GT and Prediction folders."
                )

            if start_time is not None:
                print(
                    f'Elapsed time: {strftime("%H:%M:%S", gmtime(perf_counter() - start_time))}'
                )

    # Store Object properties in a dataframe
    IoU_per_BN_df["Grand_Parent_Folder"] = GP_folder_list
    IoU_per_BN_df["File_name"] = file_name_list
    IoU_per_BN_df["GT_Label"] = GT_label_list
    IoU_per_BN_df["Prediction_Label"] = pred_label_list
    IoU_per_BN_df["IoU"] = IoU_list
    IoU_per_BN_df["f1_score"] = f1_score_list

    # If dataframe is not empty save the results as csv
    if len(GP_folder_list) != 0:
        # Save IoU per object statistics in csv file
        IoU_per_BN_df.to_csv(
            os.path.join(
                result_dir,
                directory.split(os.sep)[-1] + "_binary_mask_stats.csv",
            )
        )

        print("Done.")

        return IoU_per_BN_df

    else:
        return None


## Plot generating functions


def generate_binary_semantic_box_plot(
    folder_path: str,
    dataset_SS: str,
    dataset_name: str,
    fig_name: str,
    y_axis: str,
    thoughput_plot: Optional[bool] = False,
    metrics_csv_path: Optional[str] = None,
    y_axis_2: Optional[str] = None,
    output_path: Optional[str] = None,
    color_line: Optional[str] = "#d62728",
    palette: Optional[list] = ["#1f77b4", "#ff9f9b"],
    fig_width: Optional[int] = 4.2,
    aspect_ratio: Optional[float] = 1.5,
) -> None:
    """
    Generate a box plot of the IoU of the binary mask and semantic segmentation images.
    It will have no title and no legend.
    x axis is the % Diameter per Pixel.

    Args:
        folder_path (str): The path to the folder containing the csv files.
        dataset_SS (str): The dataset name for the semantic segmentation csv files.
        dataset_name (str): The dataset name for the instance segmentation csv files.
        fig_name (str): The name of the figure.
        y_axis (str): The column to use for the y-axis.
        output_path (str): The path to the folder to save the figures.
        palette (Optional[list]): The color palette for the plot, list of hexcodes.
        fig_width (Optional[int]): The width of the figure.
        aspect_ratio (Optional[float]): The aspect ratio of the plot.
    """

    # Input variables
    x_axis = "% Diameter per Pixel"

    # Get the csv files
    csv_dict = get_csv_dict(folder_path)

    # Import CSVs
    csv_BN = pd.read_csv(csv_dict[dataset_SS][1])
    csv_SS = pd.read_csv(csv_dict[dataset_SS][2])
    csv_instance_summary = pd.read_csv(csv_dict[dataset_name][-1])

    # Calculate mean diameter per sampling and use it to calculate % Diameter per Pixel
    mean_diam_sampling = mean_obj_diam_dict(dataset_name, csv_dict)

    csv_BN["Mean_diameter_per_sampling_GT"] = csv_BN[
        "Grand_Parent_Folder"
    ].map(mean_diam_sampling)
    csv_SS["Mean_diameter_per_sampling_GT"] = csv_SS[
        "Grand_Parent_Folder"
    ].map(mean_diam_sampling)

    csv_BN["% Diameter per Pixel"] = (
        (100 / csv_BN["Mean_diameter_per_sampling_GT"]).round(0).astype(int)
    )
    csv_SS["% Diameter per Pixel"] = (
        (100 / csv_SS["Mean_diameter_per_sampling_GT"]).round(0).astype(int)
    )

    # Get % diameter per pixel of original image
    og_percent = csv_SS[csv_SS["Grand_Parent_Folder"] == "OG"][
        "% Diameter per Pixel"
    ].values[0]

    # Filter the dataframe
    csv_SS = csv_SS[csv_SS["GT_Label"] == "ALL"]

    # Add a column to identify the source of the data
    csv_instance_summary["Source"] = "Instance Summary"
    csv_BN["Source"] = "Binary Mask"
    csv_SS["Source"] = "Semantic\nSegmentation"

    # Concatenate the dataframes
    dataframe = pd.concat([csv_BN, csv_SS], axis=0, ignore_index=True)

    # If adding throughput line to plot
    if thoughput_plot:
        csv_instance_summary["Mean_diameter_per_sampling_GT"] = (
            csv_instance_summary["Grand_Parent_Folder"].map(mean_diam_sampling)
        )
        csv_instance_summary["% Diameter per Pixel"] = (
            (100 / csv_instance_summary["Mean_diameter_per_sampling_GT"])
            .round(0)
            .astype(int)
            .astype(str)
        )

    sns.set_context("talk")
    fig, ax1 = plt.subplots()

    # Arguments for plotting
    plot_args_box = {
        "data": dataframe,
        "x": x_axis,
        "y": y_axis,
        "hue": "Source",
        "palette": palette,
        "dodge": True,
        "linecolor": "black",
        "linewidth": 2,
        "whis": 1.5,  # 1.5 IQR
        "legend": False,
        "ax": ax1,
    }

    # Plot
    plot = sns.boxplot(**plot_args_box)

    # Identify the original sampling
    plt.axvline(str(og_percent), color="black", dashes=(2, 5))

    # Set fixed figure width
    plt.gcf().set_size_inches(fig_width, fig_width / aspect_ratio)

    # Add major grid lines, x label and y top limit
    plt.grid(axis="y", which="major")
    plt.ylim(top=1)
    plt.xlabel("Pixel Diameter [%]")

    if thoughput_plot:
        # Create a secondary y-axis
        ax2 = ax1.twinx()

        # Calculate microscopeFOV from original resolution dataset
        mic_FOV_area = microscope_FOV_area(metrics_csv_path, dataset_name)

        # Calculate the objects per FOV for each sampling
        objs_per_FOV_df = obj_per_microscope_FOV(
            mic_FOV_area, folder_path, dataset_name
        )

        # Merge the dataframes
        csv_instance_summary = pd.merge(
            csv_instance_summary,
            objs_per_FOV_df,
            on=["Grand_Parent_Folder", "File_name"],
            how="left",
        )

        plot_args_line = {
            "data": csv_instance_summary,
            "x": x_axis,
            "y": y_axis_2,
            "color": color_line,
            "linewidth": 2,
            "errorbar": ("ci", 95),
            "ax": ax2,
        }

        sns.lineplot(**plot_args_line)

        # y-axis log scale and labels
        plt.yscale("log")
        plt.ylabel("Throughput [N/\u03c4]")

    # Save the plot
    if output_path is not None:
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_{y_axis}.svg",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_{y_axis}.png",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_{y_axis}.pdf",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )


def generate_semantic_gt_pred_bar_plot(
    folder_path: str,
    dataset_name: str,
    fig_name: str,
    output_path: Optional[str] = None,
    palette: Optional[list] = ["#7f7f7f", "#ff9f9b"],
    fig_width: Optional[int] = 5.5,
    aspect_ratio: Optional[Union[int, float]] = 1.5,
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
) -> None:
    """
    Generate a bar plot comparing the estimated median diameter of the ground truth and the prediction for a dataset.
    x axis is the % Diameter per Pixel and y axis is the mean median diameter of the objects per FOV.

    Args:
        folder_path (str): The path to the folder containing the csv files.
        dataset_name (str): The dataset name for the instance segmentation csv files.
        fig_name (str): The name of the figure.
        output_path (str): The path to the folder to save the figures.
        palette (Optional[list]): The color palette for the plot, list of hexcodes.
        fig_width (Optional[int]): The width of the figure.
        aspect_ratio (Optional[float]): The aspect ratio of the plot.
        folder_sampling_dict (Optional[Dict[str, float]]): The dictionary identifying sampling multipliers according to folder
    """

    # Input variables
    x_axis = "% Diameter per Pixel"
    y_axis = "Diameter"

    # Get the csv files
    csv_dict = get_csv_dict(folder_path)

    # Import CSVs
    csv_instance_summary = pd.read_csv(csv_dict[dataset_name][-1])

    # Calculate mean diameter per sampling and use it to calculate % Diameter per Pixel
    mean_diam_sampling = mean_obj_diam_dict(dataset_name, csv_dict)

    csv_instance_summary["Mean_diameter_per_sampling_GT"] = (
        csv_instance_summary["Grand_Parent_Folder"].map(mean_diam_sampling)
    )

    csv_instance_summary["% Diameter per Pixel"] = (
        (100 / csv_instance_summary["Mean_diameter_per_sampling_GT"])
        .round(0)
        .astype(int)
    )

    # Get % diameter per pixel of original image
    og_percent = csv_instance_summary[
        csv_instance_summary["Grand_Parent_Folder"] == "OG"
    ]["% Diameter per Pixel"].values[0]

    # Normalize GT and Prediction median diameter from sampling
    csv_instance_summary["GT_diameter_median_norm"] = csv_instance_summary[
        "GT_diameter_median"
    ] / csv_instance_summary["Grand_Parent_Folder"].map(folder_sampling_dict)
    csv_instance_summary["Prediction_diameter_median_norm"] = (
        csv_instance_summary["pred_diameter_median"]
        / csv_instance_summary["Grand_Parent_Folder"].map(folder_sampling_dict)
    )

    # Create a dataframe for the plot
    gt_df = csv_instance_summary[
        ["GT_diameter_median_norm", "% Diameter per Pixel"]
    ].rename(columns={"GT_diameter_median_norm": y_axis})
    gt_df["Source\nSegmentation"] = "Ground Truth"
    pred_df = csv_instance_summary[
        ["Prediction_diameter_median_norm", "% Diameter per Pixel"]
    ].rename(columns={"Prediction_diameter_median_norm": y_axis})
    pred_df["Source\nSegmentation"] = "Prediction"

    dataframe = pd.concat([gt_df, pred_df], axis=0, ignore_index=True)

    sns.set_context("talk")

    # Arguments for plotting
    plot_args_box = {
        "data": dataframe,
        "x": x_axis,
        "y": y_axis,
        "hue": "Source\nSegmentation",
        "palette": palette,
        "kind": "bar",
        "height": 3.5,
        "aspect": aspect_ratio,
        "dodge": True,
        "linewidth": 2,
        "errorbar": ("pi", 95),
        "capsize": 0.2,
        "err_kws": {"color": "black", "linewidth": 1},
        "edgecolor": "black",
        "zorder": 2,
        "legend": False,
    }

    # Plot
    plot = sns.catplot(**plot_args_box)

    plt.axvline(str(og_percent), color="black", dashes=(2, 5))

    # Set fixed figure width
    plt.gcf().set_size_inches(fig_width, plt.gcf().get_size_inches()[1])

    # Force y-axis to round to the next major grid point
    max_y_value = dataframe[y_axis].max()
    rounded_max_y = math.ceil(max_y_value / 10.0) * 10
    plt.ylim(top=rounded_max_y)

    plt.grid(axis="y", which="major")
    plt.xlabel("Pixel Diameter [%]")

    # Save the plot
    if output_path is not None:
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_GT_pred_{y_axis}.svg",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_GT_pred_{y_axis}.png",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_GT_pred_{y_axis}.pdf",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )


def generate_instance_box_plot(
    folder_path: str,
    dataset_name: str,
    fig_name: str,
    y_axis: str,
    thoughput_plot: Optional[bool] = False,
    y_axis_2: Optional[str] = None,
    metrics_csv_path: Optional[str] = None,
    color_line: Optional[str] = "#d62728",
    subset_filenames_to_exclude: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    color: Optional[str] = "#1f77b4",
    fig_width: Optional[Union[int, float]] = 8,
    aspect_ratio: Optional[float] = 2.3,
    is_round_obj: Optional[bool] = True,
) -> None:
    """
    Generate a box plot of the instance segmentation images.
    It will have no title and no legend.
    x axis is the % Diameter per Pixel.

    Args:
        folder_path (str): The path to the folder containing the csv files.
        dataset_name (str): The dataset name for the instance segmentation csv files.
        fig_name (str): The name of the figure.
        y_axis (str): The column to use for the y-axis.
        subset_filenames_to_exclude (Optional[list]): The list of filenames to exclude from the plot.
        output_path (Optional[str]): The path to the folder to save the figures.
        color (Optional[list]): The color palette for the plot, list of hexcodes.
        fig_width (Optional[int]): The width of the figure.
        aspect_ratio (Optional[float]): The aspect ratio of the plot.
        folder_sampling_dict (Optional[Dict[str, float]]): The dictionary identifying sampling multipliers according to folder.

    """
    # Input variables
    x_axis = "% Diameter per Pixel"

    # Get the csv files
    csv_dict = get_csv_dict(folder_path)

    # Import CSVs
    csv_instance_summary = pd.read_csv(csv_dict[dataset_name][-1])

    # Calculate mean diameter per sampling and use it to calculate % Diameter per Pixel
    mean_diam_sampling = mean_obj_diam_dict(
        dataset_name, csv_dict, is_round_obj
    )

    # Assign the mean diameter per sampling to the dataframe based on sampling
    csv_instance_summary["Mean_diameter_per_sampling_GT"] = (
        csv_instance_summary["Grand_Parent_Folder"].map(mean_diam_sampling)
    )

    # Calculate % diameter per pixel
    csv_instance_summary["% Diameter per Pixel"] = (
        (100 / csv_instance_summary["Mean_diameter_per_sampling_GT"])
        .round(1)
        .astype(float)
    )

    # If thoughput plot is true
    if thoughput_plot:
        order = sorted(csv_instance_summary["% Diameter per Pixel"].unique())
        csv_instance_summary["% Diameter per Pixel"] = (
            (100 / csv_instance_summary["Mean_diameter_per_sampling_GT"])
            .round(1)
            .astype(float)
            .astype(str)
        )

    # If a subset is given, filter the dataframe
    if subset_filenames_to_exclude is not None:
        if any(
            file in csv_instance_summary["File_name"].unique()
            for file in subset_filenames_to_exclude
        ):
            csv_instance_summary = csv_instance_summary[
                ~csv_instance_summary["File_name"].isin(
                    subset_filenames_to_exclude
                )
            ]

    # Get % diameter per pixel of original image
    og_percent = csv_instance_summary[
        csv_instance_summary["Grand_Parent_Folder"] == "OG"
    ]["% Diameter per Pixel"].values[0]

    sns.set_context(
        "talk",
        rc={
            "font.size": 25,
            "axes.titlesize": 22,
            "axes.labelsize": 25,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "legend.title_fontsize": 20,
        },
    )
    fig, ax1 = plt.subplots()

    # Arguments for plotting
    plot_args_box = {
        "data": csv_instance_summary,
        "x": x_axis,
        "y": y_axis,
        "color": color,
        "dodge": True,
        "linecolor": "black",
        "linewidth": 2,
        "whis": 1.5,  # 1.5 IQR
        "legend": False,
    }

    if thoughput_plot:
        plot_args_box["ax"] = ax1
        plot_args_box["order"] = order

    # Plot
    plot = sns.boxplot(**plot_args_box)

    # Identify the original sampling
    plt.axvline(str(og_percent), color="black", dashes=(2, 5))

    # Set fixed figure width
    plt.gcf().set_size_inches(fig_width, fig_width / aspect_ratio)

    # Major gridlines, x label and y top limit
    plt.grid(axis="y", which="major")
    plt.ylim(top=1)
    plt.xlabel("Pixel Diameter [%]")

    if thoughput_plot:
        # Create a secondary y-axis
        ax2 = ax1.twinx()

        # Calculate microscopeFOV from original resolution dataset
        mic_FOV_area = microscope_FOV_area(metrics_csv_path, dataset_name)

        # Calculate the objects per FOV for each sampling
        objs_per_FOV_df = obj_per_microscope_FOV(
            mic_FOV_area, folder_path, dataset_name
        )

        # Merge the dataframes
        csv_instance_summary = pd.merge(
            csv_instance_summary,
            objs_per_FOV_df,
            on=["Grand_Parent_Folder", "File_name"],
            how="left",
        )

        plot_args_line = {
            "data": csv_instance_summary,
            "x": x_axis,
            "y": y_axis_2,
            "color": color_line,
            "linewidth": 2,
            "errorbar": ("ci", 95),
            "ax": ax2,
        }

        sns.lineplot(**plot_args_line)

        # y-axis log scale and labels
        plt.yscale("log")
        plt.ylabel("Throughput [N/\u03c4]")

    # Save the plot
    if output_path is not None:
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_{y_axis}.svg",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_{y_axis}.png",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_{y_axis}.pdf",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )


def generate_instance_gt_pred_bar_plot(
    folder_path: str,
    dataset_name: str,
    fig_name: str,
    subset_filenames_to_exclude: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    palette: Optional[List[str]] = ["#7f7f7f", "#ff9f9b"],
    fig_width: Optional[Union[int, float]] = 8,
    aspect_ratio: Optional[Union[int, float]] = 2,
    is_round_obj: Optional[bool] = True,
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
) -> None:
    """
    Generate a bar plot comparing the estimated median diameter of the ground truth and the prediction for a dataset.
    x axis is the % Diameter per Pixel and y axis is the mean median diameter of the objects per FOV.

    Args:
        folder_path (str): The path to the folder containing the csv files.
        dataset_name (str): The dataset name for the instance segmentation csv files.
        fig_name (str): The name of the figure.
        y_axis (str): The column to use for the y-axis.
        subset_filenames_to_exclude (Optional[list]): The list of filenames to exclude from the plot.
        output_path (Optional[str]): The path to the folder to save the figures.
        color (Optional[list]): The color palette for the plot, list of hexcodes.
        fig_width (Optional[int]): The width of the figure.
        aspect_ratio (Optional[float]): The aspect ratio of the plot.
        folder_sampling_dict (Optional[Dict[str, float]]): The dictionary identifying sampling multipliers according to folder.


    """
    # Input variables
    x_axis = "% Diameter per Pixel"
    y_axis = "Diameter"

    # Get the csv files
    csv_dict = get_csv_dict(folder_path)

    # Calculate mean diameter per sampling and use it to calculate % Diameter per Pixel
    mean_diam_sampling = mean_obj_diam_dict(
        dataset_name, csv_dict, is_round_obj
    )

    # Import CSVs
    csv_instance_summary = pd.read_csv(csv_dict[dataset_name][-1])
    csv_instance_per_obj = pd.read_csv(csv_dict[dataset_name][0])

    # Calculate object diameter from area, assuming objects are circular
    csv_instance_per_obj["GT_diameter_from_area"] = 2 * np.sqrt(
        csv_instance_per_obj["GT_area"] / np.pi
    )
    csv_instance_per_obj["pred_diameter_from_area"] = 2 * np.sqrt(
        csv_instance_per_obj["pred_area"] / np.pi
    )

    # Calculate median values of objecter diameter for summary table
    csv_instance_summary["Median_GT_diameter_from_area"] = (
        csv_instance_per_obj.groupby(["Grand_Parent_Folder", "File_name"])[
            "GT_diameter_from_area"
        ]
        .median()
        .reset_index(drop=True)
    )
    csv_instance_summary["Median_pred_diameter_from_area"] = (
        csv_instance_per_obj.groupby(["Grand_Parent_Folder", "File_name"])[
            "pred_diameter_from_area"
        ]
        .median()
        .reset_index(drop=True)
    )

    # Assign the mean diameter per sampling to the dataframe based on sampling
    csv_instance_summary["Mean_diameter_per_sampling_GT"] = (
        csv_instance_summary["Grand_Parent_Folder"].map(mean_diam_sampling)
    )

    # Calculate % diameter per pixel
    csv_instance_summary["% Diameter per Pixel"] = (
        100 / csv_instance_summary["Mean_diameter_per_sampling_GT"]
    ).round(1)

    # If a subset is given, filter the dataframe
    if subset_filenames_to_exclude is not None:
        if any(
            file in csv_instance_summary["File_name"].unique()
            for file in subset_filenames_to_exclude
        ):
            csv_instance_summary = csv_instance_summary[
                ~csv_instance_summary["File_name"].isin(
                    subset_filenames_to_exclude
                )
            ]

    # Get % diameter per pixel of original image
    og_percent = csv_instance_summary[
        csv_instance_summary["Grand_Parent_Folder"] == "OG"
    ]["% Diameter per Pixel"].values[0]

    # Normalize GT and Prediction median diameter from sampling
    csv_instance_summary["GT_diameter_median_norm"] = csv_instance_summary[
        "Median_GT_diameter_from_area"
    ] / csv_instance_summary["Grand_Parent_Folder"].map(folder_sampling_dict)
    csv_instance_summary["Prediction_diameter_median_norm"] = (
        csv_instance_summary["Median_pred_diameter_from_area"]
        / csv_instance_summary["Grand_Parent_Folder"].map(folder_sampling_dict)
    )

    # Create a dataframe for the plot
    gt_df = csv_instance_summary[
        ["GT_diameter_median_norm", "% Diameter per Pixel"]
    ].rename(columns={"GT_diameter_median_norm": y_axis})
    gt_df["Source\nSegmentation"] = "Ground Truth"
    pred_df = csv_instance_summary[
        ["Prediction_diameter_median_norm", "% Diameter per Pixel"]
    ].rename(columns={"Prediction_diameter_median_norm": y_axis})
    pred_df["Source\nSegmentation"] = "Prediction"

    # Concatenate the dataframes
    dataframe = pd.concat([gt_df, pred_df], axis=0, ignore_index=True)

    # Set the context for the plot
    sns.set_context(
        "talk",
        rc={
            "font.size": 25,
            "axes.titlesize": 22,
            "axes.labelsize": 25,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "legend.title_fontsize": 20,
        },
    )

    # Arguments for plotting
    plot_args_box = {
        "data": dataframe,
        "x": x_axis,
        "y": y_axis,
        "hue": "Source\nSegmentation",
        "palette": palette,
        "kind": "bar",
        "height": 3.5,
        "aspect": aspect_ratio,
        "dodge": True,
        "linewidth": 2,
        "errorbar": ("pi", 95),
        "capsize": 0.2,
        "err_kws": {"color": "black", "linewidth": 1},
        "edgecolor": "black",
        "zorder": 2,
        "legend": False,
    }

    # Plot
    plot = sns.catplot(**plot_args_box)

    plt.axvline(str(og_percent), color="black", dashes=(2, 5))

    # Set fixed figure width
    plt.gcf().set_size_inches(fig_width, plt.gcf().get_size_inches()[1])

    # Force y-axis top to round to the next major grid point
    max_y_value = dataframe[y_axis].max()
    rounded_max_y = math.ceil(max_y_value / 10.0) * 10
    plt.ylim(top=rounded_max_y)

    # Force y-axis bottom to round to the next major grid point
    min_y_value = dataframe[y_axis].min()
    rounded_min_y = math.floor(min_y_value / 10.0) * 10
    plt.ylim(bottom=rounded_min_y)

    plt.grid(axis="y", which="major")
    plt.xlabel("Pixel Diameter [%]")

    # Save the plot
    if output_path is not None:
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_GT_pred_{y_axis}.svg",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_GT_pred_{y_axis}.png",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_GT_pred_{y_axis}.pdf",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )


def generate_instance_wt_treatment_bar_plot(
    folder_path: str,
    dataset_name: str,
    fig_name: str,
    subset_filenames_treatment: List[str],
    output_path: Optional[str] = None,
    palette: Optional[List[str]] = [
        "#1f77b4",
        "#a1c9f4",
        "#ff7f0e",
        "#ffb482",
    ],
    fig_width: Optional[Union[int, float]] = 8,
    aspect_ratio: Optional[Union[int, float]] = 2.2,
    is_round_obj: Optional[bool] = True,
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
) -> None:
    """
    Generate a bar plot comparing the estimated median diameter of the ground truth and the prediction for the wt and treatment subsets of a dataset.
    x axis is the % Diameter per Pixel and y axis is the mean median diameter of the objects per FOV.

    Args:
        folder_path (str): The path to the folder containing the csv files.
        dataset_name (str): The dataset name for the instance segmentation csv files.
        fig_name (str): The name of the figure.
        y_axis (str): The column to use for the y-axis.
        subset_filenames_treatment (list): The list of filenames that belong to the wt subset.
        output_path (Optional[str]): The path to the folder to save the figures.
        color (Optional[list]): The color palette for the plot, list of hexcodes.
        fig_width (Optional[int]): The width of the figure.
        aspect_ratio (Optional[float]): The aspect ratio of the plot.
        folder_sampling_dict (Optional[Dict[str, float]]): The dictionary identifying sampling multipliers according to folder.


    """
    # Input variables
    x_axis = "% Diameter per Pixel"
    y_axis = "Diameter"

    # Get the csv files
    csv_dict = get_csv_dict(folder_path)

    # Calculate mean diameter per sampling and use it to calculate % Diameter per Pixel
    mean_diam_sampling = mean_obj_diam_dict(
        dataset_name, csv_dict, is_round_obj
    )

    # Import CSVs
    csv_instance_summary = pd.read_csv(csv_dict[dataset_name][-1])
    csv_instance_per_obj = pd.read_csv(csv_dict[dataset_name][0])

    # ID the treatment CSVs
    csv_instance_summary["Subset"] = csv_instance_summary["File_name"].map(
        lambda x: "Treatment" if x in subset_filenames_treatment else "WT"
    )
    csv_instance_per_obj["Subset"] = csv_instance_per_obj["File_name"].map(
        lambda x: "Treatment" if x in subset_filenames_treatment else "WT"
    )

    # Calculate object diameter from area, assuming objects are circular
    csv_instance_per_obj["GT_diameter_from_area"] = 2 * np.sqrt(
        csv_instance_per_obj["GT_area"] / np.pi
    )
    csv_instance_per_obj["pred_diameter_from_area"] = 2 * np.sqrt(
        csv_instance_per_obj["pred_area"] / np.pi
    )

    # Calculate median values of objecter diameter for summary table
    csv_instance_summary["Median_GT_diameter_from_area"] = (
        csv_instance_per_obj.groupby(["Grand_Parent_Folder", "File_name"])[
            "GT_diameter_from_area"
        ]
        .median()
        .reset_index(drop=True)
    )
    csv_instance_summary["Median_pred_diameter_from_area"] = (
        csv_instance_per_obj.groupby(["Grand_Parent_Folder", "File_name"])[
            "pred_diameter_from_area"
        ]
        .median()
        .reset_index(drop=True)
    )

    # Assign the mean diameter per sampling to the dataframe based on sampling
    csv_instance_summary["Mean_diameter_per_sampling_GT"] = (
        csv_instance_summary["Grand_Parent_Folder"].map(mean_diam_sampling)
    )

    # Calculate % diameter per pixel
    csv_instance_summary["% Diameter per Pixel"] = (
        100 / csv_instance_summary["Mean_diameter_per_sampling_GT"]
    ).round(1)

    # Get % diameter per pixel of original image
    og_percent = csv_instance_summary[
        csv_instance_summary["Grand_Parent_Folder"] == "OG"
    ]["% Diameter per Pixel"].values[0]

    # Normalize GT and Prediction median diameter from sampling
    csv_instance_summary["GT_diameter_median_norm"] = csv_instance_summary[
        "Median_GT_diameter_from_area"
    ] / csv_instance_summary["Grand_Parent_Folder"].map(folder_sampling_dict)
    csv_instance_summary["Prediction_diameter_median_norm"] = (
        csv_instance_summary["Median_pred_diameter_from_area"]
        / csv_instance_summary["Grand_Parent_Folder"].map(folder_sampling_dict)
    )

    # Create a dataframe for the plot
    gt_df = csv_instance_summary[
        ["GT_diameter_median_norm", "% Diameter per Pixel", "Subset"]
    ].rename(columns={"GT_diameter_median_norm": y_axis})
    gt_df["Source\nSegmentation"] = "Ground Truth " + gt_df["Subset"]
    pred_df = csv_instance_summary[
        ["Prediction_diameter_median_norm", "% Diameter per Pixel", "Subset"]
    ].rename(columns={"Prediction_diameter_median_norm": y_axis})
    pred_df["Source\nSegmentation"] = "Prediction " + pred_df["Subset"]

    # Concatenate the dataframes
    dataframe = pd.concat([gt_df, pred_df], axis=0, ignore_index=True)

    # Set the context for the plot
    sns.set_context(
        "talk",
        rc={
            "font.size": 25,
            "axes.titlesize": 22,
            "axes.labelsize": 25,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "legend.title_fontsize": 20,
        },
    )

    # Arguments for plotting
    plot_args_box = {
        "data": dataframe,
        "x": x_axis,
        "y": y_axis,
        "hue": "Source\nSegmentation",
        "palette": palette,
        "kind": "bar",
        "height": 3.5,
        "aspect": aspect_ratio,
        "dodge": True,
        "linewidth": 2,
        "errorbar": ("pi", 95),
        "capsize": 0.2,
        "err_kws": {"color": "black", "linewidth": 1},
        "edgecolor": "black",
        "zorder": 2,
        "hue_order": [
            "Ground Truth WT",
            "Prediction WT",
            "Ground Truth Treatment",
            "Prediction Treatment",
        ],
        "legend": False,
    }

    # Plot
    plot = sns.catplot(**plot_args_box)

    plt.axvline(str(og_percent), color="black", dashes=(2, 5))

    # Set fixed figure width
    plt.gcf().set_size_inches(fig_width, plt.gcf().get_size_inches()[1])

    # Force y-axis top to round to the next major grid point
    max_y_value = dataframe[y_axis].max()
    rounded_max_y = math.ceil(max_y_value / 5) * 5
    plt.ylim(top=rounded_max_y)

    # Force y-axis bottom to round to the next major grid point
    min_y_value = dataframe[y_axis].min()
    rounded_min_y = math.floor(min_y_value / 10.0) * 10
    plt.ylim(bottom=rounded_min_y)

    plt.grid(axis="y", which="major")
    plt.xlabel("Pixel Diameter [%]")

    # Save the plot
    if output_path is not None:
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_GT_pred_{y_axis}.svg",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_GT_pred_{y_axis}.png",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_{dataset_name}_GT_pred_{y_axis}.pdf",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )


def generate_throughput_line_plot(
    folder_path: str,
    dataset_name_list: list,
    fig_name: str,
    metrics_csv_path: str,
    round_datasets: Optional[list] = [],
    output_path: Optional[str] = None,
    palette: Optional[list] = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
    fig_width: Optional[int] = 8,
):
    """
    Generate a line plot comparing the throughput of the different datasets.

    Args:
        folder_path (str): The path to the folder containing the csv files.
        dataset_name_list (list): The list of dataset names.
        fig_name (str): The name of the figure.
        metrics_csv_path (str): The path to the metrics csv file.
        round_datasets (Optional[list]): The list of datasets that have round objects.
        output_path (Optional[str]): The path to the folder to save the figures.
        palette (Optional[list]): The color palette for the plot, list of hexcodes.
        fig_width (Optional[int]): The width of the figure.

    Returns:

    """
    # variables
    color_count = 0
    x_axis = "% Diameter per Pixel"
    y_axis = "Obj_per_FOV_mean"

    # Get dictionary of CSVs in folder
    csv_dict = get_csv_dict(folder_path)

    # Read CSVs
    for dataset_name in dataset_name_list:
        if dataset_name in csv_dict.keys():
            # Load dataset summary csv
            csv_instance_summary = pd.read_csv(csv_dict[dataset_name][0])

            # Calculate mean diameter per sampling and use it to calculate % Diameter per Pixel
            if dataset_name in round_datasets:
                is_round_obj = True
            else:
                is_round_obj = False

            mean_diam_sampling = mean_obj_diam_dict(
                dataset_name, csv_dict, is_round_obj
            )

            # Assign the mean diameter per sampling to the dataframe based on sampling
            csv_instance_summary["Mean_diameter_per_sampling_GT"] = (
                csv_instance_summary["Grand_Parent_Folder"].map(
                    mean_diam_sampling
                )
            )

            # Calculate % diameter per pixel
            csv_instance_summary["% Diameter per Pixel"] = (
                100 / csv_instance_summary["Mean_diameter_per_sampling_GT"]
            ).round(1)

            # Calculate microscopeFOV from original resolution dataset
            mic_FOV_area = microscope_FOV_area(metrics_csv_path, dataset_name)

            # Calculate the objects per FOV for each sampling
            objs_per_FOV_df = obj_per_microscope_FOV(
                mic_FOV_area, folder_path, dataset_name
            )

            # Merge the dataframes
            csv_instance_summary = pd.merge(
                csv_instance_summary,
                objs_per_FOV_df,
                on=["Grand_Parent_Folder", "File_name"],
                how="left",
            )

            sns.set_context(
                "talk",
                rc={
                    "font.size": 25,
                    "axes.titlesize": 22,
                    "axes.labelsize": 25,
                    "xtick.labelsize": 20,
                    "ytick.labelsize": 20,
                    "legend.fontsize": 20,
                    "legend.title_fontsize": 20,
                },
            )

            plot_args_line = {
                "data": csv_instance_summary,
                "x": x_axis,
                "y": y_axis,
                "color": palette[color_count],
                "linewidth": 2,
                "label": dataset_name,
                "errorbar": ("pi", 95),
            }

            sns.lineplot(**plot_args_line)

            color_count += 1

    plt.yscale("log")

    # Place the legend outside the plot
    plt.legend(bbox_to_anchor=(0, -1), loc="lower left")

    # Set fixed figure width
    plt.gcf().set_size_inches(fig_width, plt.gcf().get_size_inches()[1])

    plt.ylabel("Throughput [N/\u03c4]")
    plt.xlabel("Pixel Diameter [%]")

    plt.grid(axis="y", which="major")

    # Save the plot
    if output_path is not None:
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_Troughput_{len(dataset_name_list)}_datasets.svg",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_Troughput_{len(dataset_name_list)}_datasets.png",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )
        plt.savefig(
            f"{output_path}/Fig_{fig_name}_Troughput_{len(dataset_name_list)}_datasets.pdf",
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=300,
            transparent=True,
        )


## Get data from PDFs


def get_metrics_from_pdfs(model_main_dir: str) -> pd.DataFrame:
    """
    Extract the metrics from the training report PDFs in the model directories.

    Args:
        model_main_dir (str): The path to the main directory containing the model directories.

    Returns:
        pd.DataFrame: A dataframe containing the metrics extracted from the PDFs.
        Save a csv file with the extracted metrics in the provided directory.
    """

    # Define variables
    pdf_ends_with = "_training_report.pdf"

    # Get list of model folders in main directory
    model_list = os.listdir(model_main_dir)

    # Create dataframe to store results
    pdf_metrics_df = pd.DataFrame()

    # Create lists to store results during loop
    notebook = []
    sample = []
    sampling = []
    n_epochs_list = []
    n_paired_image_patches_list = []
    img_dimensions_list = []
    patch_size_list = []
    training_time_list = []

    # Loop through all models folders
    for model in model_list:
        # Skip cache directories
        if model in [".DS_Store", "__pycache__"]:
            continue

        else:
            model_dir = os.path.join(model_main_dir, model)

            # Skip if not directory
            if not os.path.isdir(model_dir):
                continue

            # Generate theoretical path to pdf
            pdf_path = os.path.join(model_dir, model + pdf_ends_with)

            # Check if pdf exists
            if os.path.exists(pdf_path):
                # Load pdf and extract text from
                pdf = pypdf.PdfReader(pdf_path)
                page = pdf.pages[0].extract_text()

                # Extract epochs
                epochs = int(re.search(r"(\d+) epochs", page).group(1))
                n_epochs_list.append(epochs)

                # Extract paired image patches
                paired_image_patches = int(
                    re.search(r"(\d+) paired image patches", page).group(1)
                )
                n_paired_image_patches_list.append(paired_image_patches)

                # Extract dimensions
                dimensions_match = re.search(
                    r"image\s*dimensions:\s*\((\d+),\s*(\d+)\)", page
                )
                if dimensions_match:
                    img_dimensions = (
                        int(dimensions_match.group(1)),
                        int(dimensions_match.group(2)),
                    )
                else:
                    img_dimensions = (0, 0)

                img_dimensions_list.append(img_dimensions)

                # Extract patch size
                patch_size_match = re.search(r"\((\d+),(\d+)\)\)", page)
                if patch_size_match:
                    patch_size = (
                        int(patch_size_match.group(1)),
                        int(patch_size_match.group(2)),
                    )
                else:
                    patch_size = (0, 0)

                patch_size_list.append(patch_size)

                # Extract training time
                for line in page.split("\n"):
                    if "Training time: " in line:
                        time_str = line.split(": ")[-1]
                        time = compact_time_string(time_str)
                        training_time_list.append(time)

                # Extract info from model name
                model_components = model.split("_")
                notebook.append(model_components[1])

                # Extract sampling
                if model_components[3] == "mix":
                    sampling.append(model_components[4])
                    sample.append(
                        model_components[2] + "_" + model_components[3]
                    )
                else:
                    sampling.append(model_components[3])
                    sample.append(model_components[2])

            else:
                continue

    # Add lists to dataframe
    pdf_metrics_df["notebook"] = notebook
    pdf_metrics_df["sample"] = sample
    pdf_metrics_df["sampling"] = sampling
    pdf_metrics_df["n_epochs"] = n_epochs_list
    pdf_metrics_df["n_paired_image_patches"] = n_paired_image_patches_list
    pdf_metrics_df["img_dimensions"] = img_dimensions_list
    pdf_metrics_df["patch_size"] = patch_size_list
    pdf_metrics_df["training_time"] = training_time_list

    # Sort dataframe
    pdf_metrics_df.sort_values(
        by=["sample", "img_dimensions"],
        ascending=True,
        ignore_index=True,
        inplace=True,
    )

    # Save dataframe to csv
    pdf_metrics_df.to_csv(os.path.join(model_main_dir, "pdf_metrics.csv"))

    print("DONE!")

    return pdf_metrics_df


## Miscellaneous functions


def pixel_coverage_percent(img_array: np.ndarray) -> float:
    """
    Calculate the pixel coverage percentage of the input image array, how much of the whole object is covered by a single pixel.

    Args:
        img_array (np.ndarray): A numpy array image array with a single object label.

    Returns:
        pixel_coverage_percent (float): The percentage of the object that each pixel covers, as a float.

    """

    # Calculate the percentage of the object that each pixel covers
    pixel_coverage = (1 / np.count_nonzero(img_array)) * 100

    return pixel_coverage


def bbox_points_for_crop(
    bbox: List[int], xmax: int, ymax: int
) -> Tuple[int, int, int, int]:
    """
    Using the bouding box coordinates for each object, new bbox coordinates for the padded crop region are calculated.

    Args:
        bbox (List[int]): A list containing the x and y coordinates of the top left and bottom right points of the bounding box.
        xmax (int): The maximum x value of the image.
        ymax (int): The maximum y value of the image.

    Returns:
        Tuple[int, int, int, int]: A tuple containing the x and y coordinates of the top left and bottom right points of the bounding box.
    """
    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox

    # Calculate the half the edge length of the box for padding
    x_radius = (x2 - x1 + 2) // 2
    y_radius = (y2 - y1 + 2) // 2

    # Calculate the new bounding box coordinates for the padded crop region but only if they are within the image bounds
    # Top Left
    x1 = (x1 - x_radius) if (x1 - x_radius) > 0 else 0
    y1 = (y1 - y_radius) if (y1 - y_radius) > 0 else 0

    # Bottom Right
    x2 = (x2 + x_radius) if (x2 + x_radius) < xmax else xmax
    y2 = (y2 + y_radius) if (y2 + y_radius) < ymax else ymax

    return x1, y1, x2, y2


def object_diameter(
    image_array: np.array,
) -> Tuple[float, float, float, float]:
    """
    Calculate the diameter of the object in the image array.

    Args:
        image_array: A numpy/dataframe image array with a single object

    Returns:
        min_diameter: The minimum diameter of the object in the image array.
        max_diameter: The maximum diameter of the object in the image array.
        mean_diameter: The mean diameter of the object in the image array.
        median_diameter: The median diameter of the object in the image array.
    """
    # Calculate the object skeleton and Euclidean distance transform
    obj_skeleton = ski.morphology.skeletonize(image_array)
    obj_edt = ndimage.distance_transform_edt(image_array)

    # Get the EDT values for the object skeleton
    obj_skeleton_edt = obj_skeleton * obj_edt

    # Calculate the min, max, mean, and median radius excluding the zero values of the background, multiply by 2 for diameter
    min_diameter = np.min(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)]) * 2
    max_diameter = np.max(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)]) * 2
    mean_diameter = np.mean(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)]) * 2
    median_diameter = (
        np.median(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)]) * 2
    )

    return min_diameter, max_diameter, mean_diameter, median_diameter


def pad_br_with_zeroes(gt_img: np.array, pred_img: np.array) -> np.array:
    """
    Calculate the padding size between the GT and Prediction images.

    Args:
        gt_img: A numpy array containing the GT image.
        pred_img: A numpy array containing the Prediction image.

    Returns:
        pad_with_zero: The padding size between the GT and Prediction images.
    """
    # Pad the Prediction image with zeroes to match the GT image shape
    padded_pred = np.pad(
        pred_img,
        (
            (0, gt_img.shape[0] - pred_img.shape[0]),
            (0, gt_img.shape[1] - pred_img.shape[1]),
        ),
        "constant",
        constant_values=0,
    )

    return padded_pred


def get_csv_dict(
    main_directory: str,
    skiped_folders: Optional[List[str]] = [".DS_Store", "__pycache__"],
) -> Dict[str, List[str]]:
    """
    Find all csv files in the latest Results folder of each folder in the input directory.

    Args:
        directory (str): The input directory containing the sub folders contating the image files.
                            directory |----> Dataset_folder|----> Grandparent_Folder |----> Parent_Folder |----> Files
        skiped_folders (Optional[List[str]]): A list of folders to skip when searching for csv files.

    Returns:
        csv_dict (Dict[str, List[str]]): A dictionary containing the csv files in the Results folder of each folder in the input directory.
    """

    # Initialize a dictionary to store the csv files names
    csv_dict = {}

    # Get the list of directories in the main directory
    directory_list = os.listdir(main_directory)

    for sub_dir in directory_list:
        if sub_dir in skiped_folders:
            continue
        else:
            curr_dir = os.path.join(main_directory, sub_dir)

            # Create the Results folder path for the current directory
            results_dir = os.path.join(curr_dir, "Results")
            base_results_dir = results_dir
            count = 1

            if not os.path.exists(results_dir):
                continue

            else:
                while os.path.exists(results_dir):
                    prev_results_dir = results_dir
                    results_dir = base_results_dir + "_" + f"{count:02d}"
                    count += 1

                results_dir = prev_results_dir

            # Find all csv files in the Results folder in the current directory
            csv_dict[sub_dir] = [
                os.path.join(results_dir, f)
                for f in sorted(os.listdir(results_dir))
                if f.endswith(".csv")
            ]

    print("DONE!")

    return csv_dict


def compact_time_string(
    time_str: str,
) -> str:
    """
    Convert a string like 'x hour(s) y min(s) z sec(s)' to a HH:MM:SS string.

    Args:
        time_str: A string containing the time in the format 'x hour(s) y min(s) z sec(s)'

    Returns:
        A str representing the time in 'HH:MM:SS' format
    """

    # Split the time string into hours, minutes, and seconds, and remove the word 'hour(s)', 'min(s)', and 'sec(s)'
    parts = time_str.split()
    hours = float(parts[0].replace("hour(s)", ""))
    minutes = float(parts[1].replace("min(s)", ""))
    seconds = float(parts[2].replace("sec(s)", ""))

    return str(f"{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")


def percentage_variation_metrics(
    folder_path: str,
    dataset_name: str,
    instance_segmentation: Optional[bool] = False,
) -> None:
    """
    Function to calculate the percentage variation of the IoU metric between the original image and the other samplings.

    Parameters:
        folder_path (str): Path to the folder containing the analysis folders.
        dataset_name (str): Name of the dataset instance to be analyzed.
        instance_segmentation (bool): If True, the dataset is an instance segmentation dataset. Default is False.

    Returns:
        None.
        Save a csv in the instance sub-folder.
    """
    # Make new dictionary to store the percentage variation of the IoU metric
    percent_var_dict = {}

    # Get dictionary of CSVs in folder
    csv_dict = get_csv_dict(folder_path)

    # Import CSVs
    csv_instance_summary = pd.read_csv(csv_dict[dataset_name][-1])

    # If it is a semantic segmentation dataset
    if not instance_segmentation:
        # Import CSVs
        csv_BN = pd.read_csv(csv_dict[dataset_name][1])
        csv_SS = pd.read_csv(csv_dict[dataset_name][2])

        # Filter the csv semantic segmentation dataframe to only include the 'ALL' GT label metrics
        csv_SS = csv_SS[csv_SS["GT_Label"] == "ALL"]

        # Calculate median IoU per sampling and use it to calculate % Diameter per Pixel
        median_IoU_samp_BN = (
            csv_BN.groupby("Grand_Parent_Folder")["IoU"].median().to_dict()
        )
        median_IoU_samp_SS = (
            csv_SS.groupby("Grand_Parent_Folder")["IoU"].median().to_dict()
        )

        # Calculate the percentage variation of the IoU metric for Binary Mask
        for key in median_IoU_samp_BN.keys():
            difference = median_IoU_samp_BN[key] - median_IoU_samp_BN["OG"]
            percent_var_dict["BN OG vs " + key] = (
                difference / median_IoU_samp_BN["OG"]
            ) * 100

        # Calculate the percentage variation of the IoU metric for Semantic Segmentation
        for key in median_IoU_samp_SS.keys():
            difference = median_IoU_samp_SS[key] - median_IoU_samp_SS["OG"]
            percent_var_dict["SS OG vs " + key] = (
                difference / median_IoU_samp_SS["OG"]
            ) * 100

        # Convert dictionary to DataFrame
        percent_var_df = pd.DataFrame(
            list(percent_var_dict.items()),
            columns=["Comparison", "Percentage Difference"],
        )

    # If it is an instance segmentation dataset
    elif instance_segmentation:
        # Import CSVs
        csv_instance_summary = pd.read_csv(csv_dict[dataset_name][-1])

        # Calculate median IoU per sampling and use it to calculate % Diameter per Pixel
        median_IoU_samp = (
            csv_instance_summary.groupby("Grand_Parent_Folder")["IoU"]
            .median()
            .to_dict()
        )

        # Calculate the percentage variation of the IoU metric for Instance Segmentation
        for key in median_IoU_samp.keys():
            difference = median_IoU_samp[key] - median_IoU_samp["OG"]
            percent_var_dict["OG vs " + key] = (
                difference / median_IoU_samp["OG"]
            ) * 100

        # Convert dictionary to DataFrame
        percent_var_df = pd.DataFrame(
            list(percent_var_dict.items()),
            columns=["Comparison", "Percentage Difference"],
        )

    # Round the 'Percentage Difference' column to 2 decimal points
    percent_var_df["Percentage Difference"] = percent_var_df[
        "Percentage Difference"
    ].round(2)

    # Save DataFrame to CSV
    percent_var_df.to_csv(
        folder_path
        + "/"
        + dataset_name
        + "/"
        + dataset_name
        + "_percent_var_dict.csv",
        index=False,
    )

    return print(
        "Percentage variation metrics saved as csv in "
        + folder_path
        + "/"
        + dataset_name
        + "/"
        + dataset_name
        + "_percent_var_dict.csv"
    )


def microscope_FOV_area(
    path_metrics_csv: str,
    dataset_name: str,
    dataset_name_match_dict: Optional[dict] = {
        "Deepbacs_instance": "deepbacs",
        "Saureus": "saureus",
        "Saureus_WT_PC190723": "saureus_mix",
        "Worm_instance": "worm",
    },
) -> float:
    """
    Function to calculate the area of the microscope FOV.

    Args:
        path_metrics_csv (str): path to the csv with the pdf metrics.
        dataset_name (str): dataset instance name.
        dataset_name_match_dict (Optional[dict]): dictionary to match dataset names to the sample name from the model in the csv.

    Returns:
        FOV_area (float): area in pixels of the microscope FOV.
    """
    # Read CSVs
    pdf_metrics_csv = pd.read_csv(path_metrics_csv)

    # Filter dataframe for specific dataset and OG sampling
    pdf_metrics_csv = pdf_metrics_csv[
        pdf_metrics_csv["sample"] == dataset_name_match_dict[dataset_name]
    ]
    pdf_metrics_csv = pdf_metrics_csv[pdf_metrics_csv["sampling"] == "og"]

    # Convert string to values and calculate FOV area from Image dimensions
    pdf_metrics_csv["img_dimensions"] = pdf_metrics_csv[
        "img_dimensions"
    ].apply(ast.literal_eval)
    pdf_metrics_csv["FOV_area"] = pdf_metrics_csv["img_dimensions"].apply(
        lambda x: x[0] * x[1]
    )

    return pdf_metrics_csv["FOV_area"].values[0]


def obj_per_microscope_FOV(
    microscope_FOV: float,
    folder_path: str,
    dataset_name: str,
    save_csv: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Function to calculate the number of objects per microscope FOV.

    Args:
        path_metrics_csv (str): path to the csv with the metrics.
        folder_path (str): path to the folder with the csvs.
        dataset_name (str): dataset instance name.
        save_csv (bool): whether to save the values in a csv.

    Returns:
        dataframe with the median and mean number of objects per microscope FOV.
    """
    # Empty Dataframe
    px_per_obj = pd.DataFrame()

    # Get dictionary of CSVs in folder
    csv_dict = get_csv_dict(folder_path)

    # Read CSVs
    per_obj_csv = pd.read_csv(csv_dict[dataset_name][0])

    # Calculate median and mean values of GT area per FOV
    px_per_obj["GT_area_median"] = per_obj_csv.groupby(
        ["Grand_Parent_Folder", "File_name"]
    )["GT_area"].median()
    px_per_obj["GT_area_mean"] = (
        per_obj_csv.groupby(["Grand_Parent_Folder", "File_name"])["GT_area"]
        .mean()
        .round(2)
    )

    # Calculate the number of objects per FOV
    px_per_obj["Obj_per_FOV_median"] = (
        microscope_FOV / px_per_obj["GT_area_median"]
    ).round(0)
    px_per_obj["Obj_per_FOV_mean"] = (
        microscope_FOV / px_per_obj["GT_area_mean"]
    ).round(0)

    if save_csv == True:
        # Save DataFrame to CSV in the dataset folder
        px_per_obj.to_csv(
            folder_path
            + "/"
            + dataset_name
            + "/"
            + dataset_name
            + "_obj_per_FOV.csv"
        )

        print(
            "CSV saved in "
            + folder_path
            + "/"
            + dataset_name
            + "/"
            + dataset_name
            + "_obj_per_FOV.csv"
        )

    return px_per_obj[["Obj_per_FOV_mean", "Obj_per_FOV_median"]]


def mean_obj_diam_dict(
    dataset_name: str,
    csv_dict: Dict[str, List[str]],
    is_round_obj: bool = False,
) -> Dict[str, float]:
    """
    Function to calculate the mean object diameter for each dataset instance.

    Args:
        csv_dict (dict): dictionary with the csvs.
        round_obj (bool): Is the object circular? Default is False.

    Returns:
        mean_obj_diam_dict (dict): dictionary with the mean object diameter for each dataset instance.
    """
    # For non-circular objects
    if is_round_obj == False:
        # Load csv
        csv_instance_summary = pd.read_csv(csv_dict[dataset_name][-1])

        # Calculate mean diameter per sampling
        mean_diam_sampling = (
            csv_instance_summary.groupby("Grand_Parent_Folder")[
                "GT_diameter_median"
            ]
            .mean()
            .to_dict()
        )

    # For circular objects
    elif is_round_obj == True:
        # Load csv
        csv_instance_summary = pd.read_csv(csv_dict[dataset_name][-1])
        csv_per_obj = pd.read_csv(csv_dict[dataset_name][0])

        # Calculate mean diameter per sampling
        csv_per_obj["GT_diameter_from_area"] = 2 * np.sqrt(
            csv_per_obj["GT_area"] / np.pi
        )
        csv_instance_summary["Median_GT_diameter_from_area"] = (
            csv_per_obj.groupby(["Grand_Parent_Folder", "File_name"])[
                "GT_diameter_from_area"
            ]
            .median()
            .reset_index(drop=True)
        )
        mean_diam_sampling = (
            csv_instance_summary.groupby("Grand_Parent_Folder")[
                "Median_GT_diameter_from_area"
            ]
            .mean()
            .to_dict()
        )

    return mean_diam_sampling


## Region properties function and sub functions


def object_props(
    directory: str,
    properties: Optional[List[str]] = [
        "label",
        "area",
        "eccentricity",
        "perimeter",
        "equivalent_diameter_area",
        "axis_major_length",
        "axis_minor_length",
        "area_filled",
    ],
    spacing: Optional[Tuple[float, float]] = None,
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
    Calculate the properties for each object in each image inthe input directory.

    Args:
        directory (str): The input directory containing the image files.
        properties (Optional[List[str]]): A list of the properties to calculated for each region. Default is ['label', 'area', 'eccentricity',  'perimeter', 'equivalent_diameter_area', 'axis_major_length', 'axis_minor_length', 'area_filled']
        spacing (Optional[Tuple[float, float]]): The physical spacing of the image files. Default is None.
        folder_sampling_dict (Optional[Dict[str, float]]): A dictionary of grandparent folder names and their sampling multipliers. Default is {'upsampling_16': 16, 'upsampling_8': 8, 'upsampling_4': 4,'upsampling_2': 2, 'OG': 1, 'downsampling_2': 1/2, 'downsampling_4': 1/4, 'downsampling_8': 1/8, 'downsampling_16': 1/16}

    Returns:
        obj_props_df (pd.DataFrame): A table containing the calculated properties for each region, extraproperties, and including the file name and parent folder, normalized values by sampling for 'area', 'area_filled', 'equivalent_diameter_area', 'perimeter', 'axis_major_length', 'axis_minor_length'.
    """

    # Create a list to store the properties DataFrames
    props_list = []

    # Loop through all files in all subdirectories
    for root, dirs, files in os.walk(directory):
        if "Results" not in root:

            # Loop through all files and add region properties/prediction statistics to a dataframe
            for file in files:
                if file.endswith(".tif"):
                    # Open image file
                    img_file = ski.io.imread(os.path.join(root, file))

                    start_time = perf_counter()  # Start timer

                    # Calculate region properties
                    props = region_properties(
                        img_file, properties=properties, spacing=spacing
                    )
                    extra_properties(props)
                    add_file_name_to_dataframe(file, props)
                    add_parent_folder(
                        props,
                        given_dir=directory,
                        root=root,
                        folder_sampling_dict=folder_sampling_dict,
                    )
                    normalize_to_sampling(props, properties)

                    # Append to the list of properties DataFrames
                    props_list.append(pd.DataFrame(props))

                    print(
                        f"Properties calculated for {file} from {root.split(os.sep)[-1]} in {root.split(os.sep)[-2]} in {strftime('%H:%M:%S', gmtime(perf_counter() - start_time))} seconds"
                    )

    # Create the dataframe by concatenating the list of properties DataFrames
    obj_props_df = pd.concat(props_list, ignore_index=True)

    return obj_props_df


def region_properties(
    label_image: np.ndarray,
    properties: List[str] = [
        "label",
        "area",
        "eccentricity",
        "perimeter",
        "equivalent_diameter_area",
        "axis_major_length",
        "axis_minor_length",
        "area_filled",
    ],
    spacing: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Calculate properties of regions in the input file and return the results.

    Args:
        label_image: The input file path for region property calculation.
        properties: A list of properties to calculate for each region. Defaults to ['label', 'area', 'eccentricity',  'perimeter', 'equivalent_diameter_area', 'axis_major_length', 'axis_minor_length', 'area_filled']
        spacing: The physical spacing of the image files. Default is None.

    Returns:
        IoU_per_obj_df: A table containing the calculated properties for each region.
    """

    # Calculate the properties of the regions in the input file
    IoU_per_obj_df = ski.measure.regionprops_table(
        label_image, properties=properties, spacing=spacing
    )

    return pd.DataFrame(IoU_per_obj_df)


def add_file_name_to_dataframe(
    file: str, IoU_per_obj_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add the file name to the dataframe.

    Args:
        file (str): The input file name used for region property calculation.
        IoU_per_obj_df (pd.DataFrame): A table containing the calculated properties for each region.

    Returns:
        IoU_per_obj_df (pd.DataFrame): Original table with added column for file name.
    """
    IoU_per_obj_df["File_name"] = file

    return IoU_per_obj_df


def extra_properties(IoU_per_obj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate roundness and expected roundness of regions in the input file and return the results, only if the 'area', 'equivalent_diameter_area' and 'perimeter' columns exist in the input table.

    Args:
        IoU_per_obj_df (pd.DataFrame): A table containing the calculated properties for each region.

    Returns:
        IoU_per_obj_df (pd.DataFrame:) Original table with added columns for circularity(4*pi*area/perimeter^2), roundness (minor axis/major axis), filledness (area/area_filled). If the required original columns are present.

    """
    if (
        "area" in IoU_per_obj_df.columns
        and "perimeter" in IoU_per_obj_df.columns
    ):
        IoU_per_obj_df["Circularity"] = (
            4
            * np.pi
            * IoU_per_obj_df["area"].astype(float)
            / IoU_per_obj_df["perimeter"].astype(float) ** 2
        )

    if (
        "axis_major_length" in IoU_per_obj_df.columns
        and "axis_minor_length" in IoU_per_obj_df.columns
    ):
        IoU_per_obj_df["Roundness"] = IoU_per_obj_df[
            "axis_minor_length"
        ].astype(float) / IoU_per_obj_df["axis_major_length"].astype(float)

    if (
        "area" in IoU_per_obj_df.columns
        and "area_filled" in IoU_per_obj_df.columns
    ):
        IoU_per_obj_df["Filledness"] = IoU_per_obj_df["area"].astype(
            float
        ) / IoU_per_obj_df["area_filled"].astype(float)

    if "area" in IoU_per_obj_df.columns:
        IoU_per_obj_df["Pixel_Coverage_Percent"] = (
            1 / IoU_per_obj_df["area"].astype(float)
        ) * 100

    return IoU_per_obj_df


def add_parent_folder(
    IoU_per_obj_df: pd.DataFrame,
    given_dir: str,
    root: str,
    folder_sampling_dict: Dict[str, float],
) -> pd.DataFrame:
    """
    Loop through the parent folders and add them to the dataframe.

    Args:
        IoU_per_obj_df (pd.DataFrame): A table containing the calculated properties for each region.
        given_dir (str): Directory containing the image files.
        root (str): Current os.walk directory
        folder_sampling_dict (Dict[str, float]): A dictionary of grandparent folders and their sampling multipliers.

    Returns:
        IoU_per_obj_df (pd.DataFrame): Original table with added column(s) for parent folder(s) and sampling multiplier.
    """
    # Default function variables
    depth_count: int = 0
    folder_col_name: str = "Parent_Folder"
    root_depth: int = len(root.split(os.sep))
    dir_depth: int = len(given_dir.split(os.sep))

    # loop through the parent folders and add them to the dataframe
    while root_depth > (dir_depth + depth_count):
        IoU_per_obj_df[folder_col_name] = root.split(os.sep)[-depth_count - 1]

        # Add the sampling multiplier based on folder name
        for folder, sampling in folder_sampling_dict.items():
            if root.split(os.sep)[-depth_count - 1] == folder:
                IoU_per_obj_df["sampling_multiplier"] = sampling

        depth_count += 1
        folder_col_name = "Grand_" + folder_col_name

    return IoU_per_obj_df


def normalize_to_sampling(
    IoU_per_obj_df: pd.DataFrame, properties: List[str]
) -> pd.DataFrame:
    """
    Normalize the dataframe to the given folder sampling.

    Args:
        IoU_per_obj_df (pd.DataFrame): A table containing the calculated properties for each region.
        properties (List[str]): A list of the properties to calculated for each region.

    Returns:
        IoU_per_obj_df (pd.DataFrame): Original table with normalized columns for 'area', 'area_filled', 'equivalent_diameter_area', 'perimeter', 'axis_major_length', 'axis_minor_length'.
    """
    for property in properties:
        if property == "area" or property == "area_filled":
            IoU_per_obj_df["norm_" + property] = (
                IoU_per_obj_df[property]
                / IoU_per_obj_df["sampling_multiplier"] ** 2
            )

        if any(
            x in property
            for x in [
                "perimeter",
                "axis_major_length",
                "axis_minor_length",
                "equivalent_diameter_area",
            ]
        ):
            IoU_per_obj_df["norm_" + property] = (
                IoU_per_obj_df[property]
                / IoU_per_obj_df["sampling_multiplier"]
            )

    return IoU_per_obj_df
