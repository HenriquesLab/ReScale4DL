# Imports
import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import skimage as ski  # type: ignore
from time import perf_counter, strftime, gmtime
from skimage.measure._regionprops_utils import perimeter  # type: ignore

from typing import List, Tuple, Dict, Optional

from .properties import pixel_coverage_percent, object_diameter
from ..utils import incremental_dir_creation, pad_with_zeroes, find_matching_labels


def main_function(
    main_dir: str,
    sampling_dir_modifier_dict: Dict[str, float],
    extra_metrics_func_list: Optional[List[callable]] = None,
):
    """
    Main function to iterate over datasets and process them.

    Parameters
    ----------
    main_dir : str
        Path to the main directory containing dataset folders.
    sampling_dir_modifier_dict : dict
        Dictionary mapping sampling folder names to their modifiers.
    extra_metrics_func_list : Optional[list of metrics function names]
        List of metrics function names to calculate.

    Returns
    -------
    None
    """

    dataset_folders = [
        f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))
    ]

    for dataset in dataset_folders:
        dataset_dir = os.path.join(main_dir, dataset)
        print(f"Processing dataset dataset: {dataset}")
        begin_time = perf_counter()

        # Call per_dataset function
        per_dataset(dataset_dir, sampling_dir_modifier_dict, extra_metrics_func_list)

        end_time = perf_counter()
        elapsed_time = end_time - begin_time
        elapsed_time_str = strftime("%H:%M:%S", gmtime(elapsed_time))
        print(f"Time taken for dataset {dataset}: {elapsed_time_str}")
        print("--------------------------------------------------")


def per_dataset(
    dataset_dir: str,
    sampling_dir_modifier_dict: Dict[str, float],
    extra_metrics_func_list: Optional[List[callable]] = None,
):
    """
    Process each dataset folder.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset folder.
    sampling_dir_modifier_dict : dict
        Dictionary mapping sampling folder names to their modifiers.
    extra_metrics_func_list : Optional[list of metrics function names]
        List of metrics function names to calculate.

    Returns
    -------
    None
    """
    # Create an empty DataFrame to store results
    dataset_df = pd.DataFrame()

    # Check for sampling folders
    sampling_folders = [
        f
        for f in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, f))
    ]

    # Check if sampling folders are in the sampling_dir_modifier_dict
    sampling_dir_list = list(sampling_dir_modifier_dict.keys())

    sampling_folders = [f for f in sampling_folders if f in sampling_dir_list]

    # Check if any sampling folders were found
    if not sampling_folders:
        print(
            "No sampling folders matching the nomenclature given in the sampling directory and multiplier dictionary were found in the dataset directory."
        )

    # Iterate over found sampling folders
    for sampling_dir in sampling_folders:
        # Call per_sampling function
        per_sampling_df = per_sampling(
            os.path.join(dataset_dir, sampling_dir),
            sampling_dir_modifier_dict[sampling_dir],
            extra_metrics_func_list,
        )

        # Check if per_sampling_df is not empty
        if per_sampling_df.empty:
            print(
                f"No results found for sampling folder: {sampling_dir}. Skipping this folder."
            )
            continue

        dataset_df = (
            pd.concat([dataset_df, per_sampling_df], ignore_index=True)
            if "dataset_df" in locals()
            else per_sampling_df
        )

    # Check for pre-existing results folder, create new one if it exists
    result_dir = incremental_dir_creation(dataset_dir, "results")

    # Save concatenated results to CSV
    dataset_results_csv = os.path.join(
        result_dir, f"{os.path.basename(dataset_dir)}_raw_results.csv"
    )
    dataset_df.to_csv(dataset_results_csv, index=False)

    # Calculate summary dataframe per file within the dataset
    """df_headers = dataset_df.columns.tolist()
    summary_df = (
        dataset_df.groupby("file_name")
        .agg(
            {
                header: "mean"
                for header in df_headers
                if header
                != ["file_name", "sampling_dir", "sampling_modifier", "dataset_dir"]
            },
        )
        .reset_index()
    )

    # Save summary dataframe to CSV
    summary_csv = os.path.join(
        result_dir, f"{os.path.basename(dataset_dir)}_summary.csv"
    )
    summary_df.to_csv(summary_csv, index=False)
    """


def per_sampling(
    sampling_dir: str,
    sampling_modifier: float,
    extra_metrics_func_list: Optional[List[callable]] = None,
) -> pd.DataFrame:
    """
    Process each sampling folder.

    Parameters
    ----------
    sampling_dir : str
        Path to the sampling folder.
    sampling_modifier : float
        Modifier for the sampling folder.
    extra_metrics_func_list : Optional[list of metrics function names]
        List of metrics function names to calculate.

    Returns
    -------
    pd.DataFrame
        DataFrame containing results for all images in the sampling folder.
    """

    # Check for GT and Prediction folders
    gt_dir = os.path.join(sampling_dir, "GT")
    pred_dir = os.path.join(sampling_dir, "Prediction")

    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(
            f"GT or Prediction folder not found in {os.path.basename(sampling_dir)}. Skipping this sampling folder."
        )
        return pd.DataFrame()

    # Get list of GT and Prediction files
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".tif")]
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".tif")]

    # Check if GT and Prediction files are paired
    paired_files = set(gt_files) & set(pred_files)

    # Check if any paired files were found
    if not paired_files:
        print(
            f"No paired GT and Prediction files found in {os.path.basename(sampling_dir)}. Skipping this sampling folder."
        )
        return pd.DataFrame()

    # Initialize an empty dataframe to store results
    results_df = pd.DataFrame()

    # Iterate over paired files
    for file_name in paired_files:
        per_image_pair_df = per_image_pair(
            file_name,
            sampling_dir,
            extra_metrics_func_list,
        )

        # Add per pair df to the results dataframe
        results_df = (
            pd.concat([results_df, per_image_pair_df], ignore_index=True)
            if not per_image_pair_df.empty
            else results_df
        )

    # add metadata to results dataframe
    results_df["dataset_dir"] = os.path.basename(os.path.dirname(sampling_dir))
    results_df["sampling_dir"] = os.path.basename(sampling_dir)
    results_df["file_name"] = results_df["file_name"].str.replace(".tif", "")
    results_df["sampling_modifier"] = sampling_modifier

    # normalize results in reference to sampling modifier dict
    results_df["normalized_value"] = (
        results_df["value"] * results_df["sampling_modifier"]
        if "value" in results_df.columns
        else results_df["value"]
    )  # Example normalization

    """
    INPUT:
    - sampling_dir - path to the folder containing the GT/Prediction folders
    - sampling modifier dict - dictionary with sampling modifiers for each sampling
    - extra_metrics_func_list - list of extra_metrics_func_list to use in analysis


    Main function to run the code
    - check for GT and Prediction folders
        if not present, skip sampling folder
    - os.listdir - get list of GT and Prediction files
        - check if GT and Prediction files are paired
            - if paired, run per_image_pair function
        - if not, skip unpaired files

    - concat all per_image_pair outputs in one df

    -  calculate normalized values in reference to sampling modifier dict
        - calculate normalized values for each metric
        - add normalized values to dataframe

    RETURN:
    dataframe with results ffrom all images

    """


def per_image_pair(
    file_name: str,
    sampling_dir: str,
    extra_metrics_func_list: Optional[List[callable]] = None,
) -> pd.DataFrame:
    """
    Process each image pair (GT and Prediction).
    Parameters
    ----------
    file_name : str
        Name of the image file.
    sampling_dir : str
        Path to the sampling folder.
    extra_metrics_func_list : Optional[list of metrics function names]
        List of metrics function names to calculate.
    Returns
    -------
    pd.DataFrame
        DataFrame containing results for the image pair.
    """
    # Default metrics to calculate
    default_metrics_func_list = [
        pixel_coverage_percent,
        object_diameter,
        perimeter,
        # Add other default metrics functions here
    ]

    # Use default metrics if extra_metrics_func_list is not provided
    metrics_func_list = extra_metrics_func_list or default_metrics_func_list

    # Get GT and Prediction file paths
    gt_file = os.path.join(sampling_dir, "GT", file_name)
    pred_file = os.path.join(sampling_dir, "Prediction", file_name)

    # Load GT and Prediction images
    gt_image = ski.io.imread(gt_file)
    pred_image = ski.io.imread(pred_file)

    # Check if the shape of the GT is bigger than the Prediction are the same and pad Prediction if not
    if gt_image.shape > pred_image.shape:
        print(
            f"{file_name} from {os.path.basename(sampling_dir)} has shape {gt_image.shape} in GT and {pred_image.shape} in Prediction. Padded Prediction to match GT shape."
        )
        pred_image = pad_with_zeroes(gt_image, pred_image)

    # If the shapes are still not equal, skip the image pair
    if gt_image.shape != pred_image.shape:
        print(
            f"GT and Prediction images have different shapes: {gt_image.shape} and {pred_image.shape}. Skipping {file_name} image pair."
        )
        return pd.DataFrame()

    # Get the paired labels from GT and Prediction with IoU
    matching_labels_list = find_matching_labels(gt_image, pred_image)

    column_names_list = ["GT_label", "Pred_label", "IoU_score"]

    for matched_lbls in matching_labels_list:
        gt_lbl, pred_lbl, score = matched_lbls

        # Get the bounding box for the GT label

    return pd.DataFrame()
    """
    INPUT:
    - GT file - path to the GT file
    - Prediction file - path to the Prediction file
    - extra_metrics_func_list - list of extra_metrics_func_list to use in analysis

    Main function to run the code
    - load paired images
    - check if images are the same size
        - if not, run resize function (pad prediction)
        - if yes, continue
    - np.unique to get number of objects in GT and Prediction
    - iterate over each GT object
        - use bounding box per obj to check for IOU
            - compute_labels_matching_scores
            - find_matching_labels
            - store results in a list/df
        - get extra_metrics_func_list from extra_metrics_func_list list
            - run each metric function
            - add results to a list/df

    matric functions:
    - Pixel coverage
    - object diameter
    - object area
    - optional from region props table
    - user defined extra_metrics_func_list from other sources?

    save results in a dataframe


    RETURN:
    dataframe with results for one image
    """
