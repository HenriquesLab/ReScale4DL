# imports
import numpy as np
import pandas as pd
from typing import List, Dict


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
    if not is_round_obj:
        # Load csv
        csv_instance_summary = pd.read_csv(csv_dict[dataset_name][-1])

        # Calculate mean diameter per sampling
        mean_diam_sampling = (
            csv_instance_summary.groupby("Grand_Parent_Folder")["GT_diameter_median"]
            .mean()
            .to_dict()
        )

    # For circular objects
    elif is_round_obj:
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
