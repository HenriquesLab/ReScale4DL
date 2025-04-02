# imports
import ast
import numpy as np
import pandas as pd
from typing import Optional
from ..utils import get_csv_dict


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
    pdf_metrics_csv["img_dimensions"] = pdf_metrics_csv["img_dimensions"].apply(
        ast.literal_eval
    )
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

    if save_csv:
        # Save DataFrame to CSV in the dataset folder
        px_per_obj.to_csv(
            folder_path + "/" + dataset_name + "/" + dataset_name + "_obj_per_FOV.csv"
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
