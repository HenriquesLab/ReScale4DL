import numpy as np
import os
from functools import lru_cache


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