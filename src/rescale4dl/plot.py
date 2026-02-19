# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
import os
from typing import List, Optional, Tuple, Dict, Literal, Union
from rescale4dl.utils import get_csv_dict
from rescale4dl.metrics.fov_estimates import microscope_FOV_area, obj_per_microscope_FOV


def _load_2d_or_3d_slice(path: str, slice_axis: int = 0, slice_index: int | None = None):
    """Load an image/volume and return a 2D slice for plotting."""
    arr = imread(path)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr

    if arr.ndim == 3:
        if slice_axis not in (0, 1, 2):
            raise ValueError("slice_axis must be 0, 1, or 2 for 3D data.")
        vol = np.moveaxis(arr, slice_axis, 0)
        if slice_index is None:
            slice_index = vol.shape[0] // 2
        if not (0 <= slice_index < vol.shape[0]):
            raise ValueError(
                f"slice_index must be in [0, {vol.shape[0] - 1}] for axis {slice_axis}."
            )
        return vol[slice_index]

    raise ValueError(f"Unsupported array ndim={arr.ndim} for file {path}.")


def plot_segmentation_example(
        input_dir: str,
        dataset: str,
        scaling: str,
        slice_axis: int = 0,
        slice_index: int | None = None,
        cmap: str = "gray",
):
    """
    Plot BOTH object-level AND binary segmentation TP/FP/FN images.
    10-panel layout: [GT, Pred, Obj-TP, Obj-FP, Obj-FN] + [Bin-TP, Bin-FP, Bin-FN]
    """
    example_images_dir = os.path.join(input_dir, dataset, scaling)
    example_images_dir_GT = os.path.join(example_images_dir, "GT")
    example_images_dir_Prediction = os.path.join(example_images_dir, "Prediction")
    example_images_results_dir = os.path.join(input_dir, dataset, "Results", scaling)

    # Binary results directory (same Results folder)
    binary_results_dir = example_images_results_dir

    # Pick first .tif in GT folder
    example_image = [f for f in os.listdir(example_images_dir_GT) if f.endswith(".tif")][0]
    example_stem = example_image.rsplit(".tif", 1)[0]

    # OBJECT-LEVEL paths (your original 5)
    gt_path = os.path.join(example_images_dir_GT, f"{example_stem}.tif")
    pred_path = os.path.join(example_images_dir_Prediction, f"{example_stem}.tif")
    obj_tp_path = os.path.join(binary_results_dir, f"{example_stem}_true_positives.tif")
    obj_fp_path = os.path.join(binary_results_dir, f"{example_stem}_false_positives.tif")
    obj_fn_path = os.path.join(binary_results_dir, f"{example_stem}_false_negatives.tif")

    # BINARY SEGMENTATION paths (the 5 new ones from binary_mask_stats)
    bin_tp_path = os.path.join(binary_results_dir, f"{example_stem}_TP_binary.tif")
    bin_fp_path = os.path.join(binary_results_dir, f"{example_stem}_FP_binary.tif")
    bin_fn_path = os.path.join(binary_results_dir, f"{example_stem}_FN_binary.tif")

    # Load all 8 images (automatically handles 2D/3D)
    gt_img = _load_2d_or_3d_slice(gt_path, slice_axis, slice_index)
    pred_img = _load_2d_or_3d_slice(pred_path, slice_axis, slice_index)
    obj_tp_img = _load_2d_or_3d_slice(obj_tp_path, slice_axis, slice_index)
    obj_fp_img = _load_2d_or_3d_slice(obj_fp_path, slice_axis, slice_index)
    obj_fn_img = _load_2d_or_3d_slice(obj_fn_path, slice_axis, slice_index)

    bin_tp_img = _load_2d_or_3d_slice(bin_tp_path, slice_axis, slice_index)
    bin_fp_img = _load_2d_or_3d_slice(bin_fp_path, slice_axis, slice_index)
    bin_fn_img = _load_2d_or_3d_slice(bin_fn_path, slice_axis, slice_index)
    if slice_index is None:
        slice_index_plot = "middle"
    else:
        slice_index_plot = str(slice_index)

    # 2-row layout: Object-level (top) + Binary-level (bottom)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    # Row 1: Object-level segmentation
    images_row1 = [gt_img, pred_img, obj_tp_img, obj_fp_img, obj_fn_img]
    titles_row1 = [f"Ground Truth z={slice_index_plot}", "Prediction", "Obj-TP", "Obj-FP", "Obj-FN"]

    for i, (img, title) in enumerate(zip(images_row1, titles_row1)):
        axes[0, i].imshow(img, cmap=cmap)
        axes[0, i].set_axis_off()
        axes[0, i].set_title(title, fontsize=12, fontweight="bold")

    # Row 2: Binary segmentation
    images_row2 = [gt_img, pred_img, bin_tp_img, bin_fp_img, bin_fn_img]  # Reuse GT/Pred
    titles_row2 = ["", "", "Bin-TP", "Bin-FP", "Bin-FN"]

    for i, (img, title) in enumerate(zip(images_row2, titles_row2)):
        axes[1, i].imshow(img, cmap=cmap)
        axes[1, i].set_axis_off()
        axes[1, i].set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()




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

