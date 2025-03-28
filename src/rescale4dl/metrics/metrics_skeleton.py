# Imports
import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import skimage as ski  # type: ignore
from skimage.measure import regionprops_table  # type: ignore
from skimage.segmentation import relabel_sequential  # type: ignore
from scipy import ndimage  # type: ignore
from time import perf_counter, strftime, gmtime

from ..utils import incremental_dir_creation


def main_function(main_dir, sampling_dir_modifier_dict, metrics):
    """
    Main function to iterate over datasets and process them.

    Parameters
    ----------
    main_dir : str
        Path to the main directory containing dataset folders.
    sampling_dir_modifier_dict : dict
        Dictionary mapping sampling folder names to their modifiers.
    metrics : list of str
        List of metrics to calculate.

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
        per_dataset(dataset_dir, sampling_dir_modifier_dict, metrics)

        end_time = perf_counter()
        elapsed_time = end_time - begin_time
        elapsed_time_str = strftime("%H:%M:%S", gmtime(elapsed_time))
        print(f"Time taken for dataset {dataset}: {elapsed_time_str}")
        print("--------------------------------------------------")


def per_dataset(dataset_dir, sampling_dir_modifier_dict, metrics):
    """
    Process each dataset folder.

    Parameters
    ----------
    dataset_dir : str
        Path to the main directory containing dataset folders.
    sampling_dir_modifier_dict : dict
        Dictionary mapping sampling folder names to their modifiers.
    metrics : list of str
        List of metrics to calculate.

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
            metrics,
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
    df_headers = dataset_df.columns.tolist()
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


def per_sampling(sampling_dir, sampling_modifier, metrics):
    """
    Process each sampling folder.

    Parameters
    ----------
    sampling_dir : str
        Path to the main directory containing dataset folders.
    sampling_dir_modifier_dict : dict
        Dictionary mapping sampling folder names to their modifiers.
    metrics : list of str
        List of metrics to calculate.

    Returns
    -------
    None
    """
    # Check for GT and Prediction folders
    gt_dir = os.path.join(sampling_dir, "GT")
    pred_dir = os.path.join(sampling_dir, "Prediction")

    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(
            f"GT or Prediction folder not found in {sampling_dir}. Skipping this sampling folder."
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
            f"No paired GT and Prediction files found in {sampling_dir}. Skipping this sampling folder."
        )
        return pd.DataFrame()

    # Initialize an empty list to store results
    results = []

    # Iterate over paired files
    for file_name in paired_files:
        pass

    """
    INPUT:
    - sampling_dir - path to the folder containing the GT/Prediction folders
    - sampling modifier dict - dictionary with sampling modifiers for each sampling
    - metrics - list of metrics to use in analysis


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


def per_image_pair():
    """
    INPUT:
    - GT file - path to the GT file
    - Prediction file - path to the Prediction file
    - metrics - list of metrics to use in analysis

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
        - get metrics from metrics list
            - run each metric function
            - add results to a list/df

    matric functions:
    - Pixel coverage
    - object diameter
    - object area
    - optional from region props table
    - user defined metrics from other sources?

    save results in a dataframe


    RETURN:
    dataframe with results for one image
    """
