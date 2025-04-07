import numpy as np
import os
from functools import lru_cache
from typing import List, Dict, Optional


def check_crop_img(arr, bin_factor):
    """
    Crop the image if any of the dimensions is not divisible by the bin factor.

    Parameters
    ----------
    arr : np.array
        Input image array.
    bin_factor : int
        Factor by which dimensions should be divisible.

    Returns
    -------
    np.array
        Cropped image array.
    """
    r, c = arr.shape

    if c % bin_factor != 0:
        c = int(c / bin_factor) * bin_factor
    if r % bin_factor != 0:
        r = int(r / bin_factor) * bin_factor

    return arr[:r, :c]


def compute_labels_matching_scores(gt: np.array, pred: np.array):
    """
    Compute matching scores between ground truth and predicted labels.

    Parameters
    ----------
    gt : np.array
        Ground truth labels.
    pred : np.array
        Predicted labels.

    Returns
    -------
    dict
        Dictionary with gt_label as keys and a list of tuples (pred_label, score) as values.
    """
    scores = {}
    gt_labels = np.unique(gt)

    for lbl in gt_labels[1:]:  # skips the background label
        scores[lbl] = []
        rows_idx, cols_idx = np.nonzero(gt == lbl)
        min_row, max_row, min_col, max_col = (
            np.min(rows_idx),
            np.max(rows_idx),
            np.min(cols_idx),
            np.max(cols_idx),
        )
        pred_box = pred[min_row : max_row + 1, min_col : max_col + 1]
        pred_labels_in_box = np.unique(pred_box)
        for pred_lbl in pred_labels_in_box:
            score = score_label_overlap(gt, pred, lbl, pred_lbl)
            scores[lbl].append([pred_lbl, score])

        scores[lbl] = sorted(scores[lbl], key=lambda x: x[1], reverse=True)

    return scores


def score_label_overlap(gt: np.array, pred: np.array, gt_label, pred_label):
    """
    Calculate the score of label overlap between ground truth and prediction.

    Parameters
    ----------
    gt : np.array
        Ground truth labels.
    pred : np.array
        Predicted labels.
    gt_label : int
        Label in ground truth.
    pred_label : int
        Label in prediction.

    Returns
    -------
    float
        Score of label overlap.
    """
    gt_mask = gt == gt_label
    pred_mask = pred == pred_label

    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)

    if union == 0:
        score = 0.0
    else:
        score = intersection / union

    return score


def remove_duplicates(scores, pred_labels):
    """
    Resolve conflicts in the scores dictionary by ensuring each pred_label
    is assigned to the gt_label with the highest score. If a pred_label has no
    assignment in the ground truth, assign it to 0.

    Parameters
    ----------
    scores : dict
        Dictionary with gt_label as keys and a list of tuples (pred_label, score) as values.
    pred_labels : np.array
        Array of unique predicted labels.

    Returns
    -------
    list
        List of tuples (gt_label, pred_label, score) with resolved conflicts.
    """
    assigned_pred_labels = set()
    result = []

    # Sort gt_labels by their highest score to prioritize them
    sorted_gt_labels = sorted(
        scores.keys(),
        key=lambda lbl: scores[lbl][0][1] if scores[lbl] else 0,
        reverse=True,
    )

    for gt_label in sorted_gt_labels:
        for pred_label, score in scores[gt_label]:
            if pred_label not in assigned_pred_labels:
                result.append((gt_label, pred_label, score))
                assigned_pred_labels.add(pred_label)
                break

    # Add unmatched pred_labels with gt_label = 0
    for pred_label in pred_labels:
        if pred_label not in assigned_pred_labels:
            result.append((0, pred_label, 0.0))

    return result


def find_matching_labels(gt: np.array, pred: np.array):
    """
    Find the matching labels between ground truth and prediction. If a pred_label
    has no assignment in the ground truth, assign it to 0.

    Parameters
    ----------
    gt : np.array
        Ground truth labels.
    pred : np.array
        Predicted labels.

    Returns
    -------
    list
        List of tuples (gt_label, pred_label, score).
    """
    if np.unique(pred).shape[0] == 1:
        return ((gt_lbl, 0, 0) for gt_lbl in np.unique(gt))

    scores = compute_labels_matching_scores(gt, pred)
    pred_labels = np.unique(pred)

    # Process scores to resolve conflicts and get final matching labels
    matching_labels = remove_duplicates(scores, pred_labels)
    return matching_labels


def incremental_dir_creation(parent_dir, incr_dir):
    """
    Create a new directory with an incremented name if it already exists.
    If the directory does not exist, create it.
    Parameters
    ----------
    parent_dir : str
        Path to the parent directory.
    incr_dir : str
        Name of the directory to create.
    Returns
    -------
    str
        Path to the created directory.
    """
    new_dir = os.path.join(parent_dir, incr_dir)

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    else:
        count = 1
        base_new_dir = new_dir
        while os.path.exists(new_dir):
            new_dir = base_new_dir + f"_{count:02d}"
            count += 1
        os.mkdir(new_dir)

    return new_dir


def pad_with_zeroes(gt_image: np.array, pred_image: np.array) -> np.array:
    """
    Calculate the padding size between the GT and Prediction images.

    Args:
        gt_image: A numpy array containing the GT image.
        pred_image: A numpy array containing the Prediction image.

    Returns:
        pred_image_padded: The prediction image padded with zeroes to match the GT image size.
    """
    # Pad the Prediction image with zeroes to match the GT image shape
    pred_image_padded = np.pad(
        pred_image,
        (
            (0, gt_image.shape[0] - pred_image.shape[0]),
            (0, gt_image.shape[1] - pred_image.shape[1]),
        ),
        "constant",
        constant_values=0,
    )

    return pred_image_padded


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




def crop_with_padding(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center-crop with padding when needed"""
    im = np.squeeze(image)
    output = np.zeros(shape=target_shape, dtype=image.dtype)
    # Handle vertical and horizontal dimension
    if im.shape[0] < target_shape[0] and im.shape[1] < target_shape[1]:
        h = np.floor((target_shape[0] - im.shape[0]) / 2)
        w = np.floor((target_shape[1] - im.shape[1]) / 2)
        w = np.uint16(w)
        h = np.uint16(h)
        output[h:h + im.shape[0], w:w + im.shape[1]] = im
    # Handle vertical dimension
    elif im.shape[0] < target_shape[0]:
        h = np.floor((target_shape[0] - im.shape[0]) / 2)
        w = np.floor(im.shape[1] / 2) - np.floor(shape[1] / 2)
        w = np.uint16(w)
        h = np.uint16(h)
        output[h:h + im.shape[0]] = im[:, w:w + target_shape[1]]
    # Handle horizontal dimension
    elif im.shape[1] < target_shape[1]:
        w = np.floor((target_shape[1] - im.shape[1]) / 2)
        h = np.floor(im.shape[0] / 2) - np.floor(target_shape[0] / 2)
        w = np.uint16(w)
        h = np.uint16(h)
        output[:, w:w + im.shape[1]] = im[h:h + target_shape[0], :]
    else:
        w = np.floor(im.shape[1] / 2) - np.floor(target_shape[1] / 2)
        h = np.floor(im.shape[0] / 2) - np.floor(target_shape[0] / 2)
        w = np.uint16(w)
        h = np.uint16(h)
        output = im[h:h + target_shape[0], w:w + target_shape[1]]
    return output
