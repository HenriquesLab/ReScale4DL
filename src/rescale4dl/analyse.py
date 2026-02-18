from typing import List, Optional
from .metrics.metrics import morphology as morphology_2d
from .metrics.metrics3d import morphology_3d


def analyse(
    main_directory: str,
    is_3d: Optional[bool] = None,
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
    run_per_object_stats: bool = True,
    run_semantic_stats: bool = True,
    run_binary_mask_stats: bool = True,
) -> None:
    """
    High-level entry point to run morphology analysis on 2D or 3D data.
    If is_3d is True, run 3D analysis (metrics3d.morphology_3d).
    Otherwise (is_3d is False or None), run 2D analysis (metrics.morphology).

    Parameters
    ----------
    main_directory : str
        Root directory containing dataset subfolders.
    is_3d : bool
        If True, call 3D analysis (metrics3d.morphology_3d),
        otherwise call 2D analysis (metrics.morphology).
    Other parameters are forwarded to the underlying morphology function.
    """

    if is_3d:
        morphology_3d(
            main_directory=main_directory,
            skip_directories=skip_directories,
            sampling_dir_list=sampling_dir_list,
            run_per_object_stats=run_per_object_stats,
            run_semantic_stats=run_semantic_stats,
            run_binary_mask_stats=run_binary_mask_stats,
        )
    else:
        morphology_2d(
            main_directory=main_directory,
            skip_directories=skip_directories,
            sampling_dir_list=sampling_dir_list,
            run_per_object_stats=run_per_object_stats,
            run_semantic_stats=run_semantic_stats,
            run_binary_mask_stats=run_binary_mask_stats,
        )
