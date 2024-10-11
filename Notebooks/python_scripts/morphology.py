# Functions for morphology analysis of label images

#Import required libraries
import os
import numpy as np
import pandas as pd
import pypdf
import re
import matplotlib.pyplot as plt
import seaborn as sns
import skimage as ski
from skimage.measure._regionprops_utils import perimeter
from sklearn import metrics as skl
from time import perf_counter, strftime, gmtime
from scipy import ndimage
from typing import List, Optional, Tuple, Dict, Literal


## Main Function


def morphology(main_directory: str) -> None:
    """
    Calculate the properties for each object in each image in the input directory.
    
    Args:
        directory (str): The input directory containing the sub folders contating the image files.
        
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
                                                    
    directory_list = os.listdir(main_directory)
    begin_time = perf_counter()

    for dir in directory_list:
        if dir in ['.DS_Store', '__pycache__', 'blank', 'main_folder']:
            continue
        else:
            curr_dir = os.path.join(main_directory, dir)
            print('Calculating properties for ' + dir)

            # Create folder to store results if it doesn't exist, if it exists make new one
            res_dir = os.path.join(curr_dir, 'Results')
            base_res_dir = res_dir
            count = 1

            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            
            else:
                while os.path.exists(res_dir):
                    res_dir = base_res_dir + '_' + f"{count:02d}"
                    count += 1
                os.mkdir(res_dir)

            per_object_statistics(directory = curr_dir,
                                  res_dir=res_dir)

            semantic_statistics(directory = curr_dir,
                                res_dir=res_dir)

    print(f'Total time: {strftime("%H:%M:%S", gmtime(perf_counter() - begin_time))}')


## Per object Prediction statistics functions


def per_object_statistics(
        directory: str,
        res_dir: str,
    ) -> Tuple:
    """
    Calculate the IoU, f1 score, and other statistics for each object in the image.
    
    Args:
        directory (str): Directory with folders of sampling folders with GT and Prediction folder pairs inside.
        res_dir (str): Directory to save the results.
        obj_props_df (pd.DataFrame): A table containing the calculated properties for each region.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two dataframes. The first contains per object IoU and f1 score statistics. The second contains a summary of the statistics.
    """

    # Create a dictionary of parent folders and grand parent folders
    parent_folder_dict = {'Grand_Parent_Folder': ['upsampling_16', 'upsampling_8', 'upsampling_4', 'upsampling_2', 'OG', 'downsampling_2', 'downsampling_4', 'downsampling_8', 'downsampling_16'], 'Parent_Folder': ['GT', 'Prediction']}
    GP_dict = parent_folder_dict['Grand_Parent_Folder']

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
    for GP_folder in sorted(GP_dict):
        # Create the path variable to the GT and Prediction folders
        GT_path = os.path.join(directory, GP_folder, 'GT')
        pred_path = os.path.join(directory, GP_folder, 'Prediction')

        if not os.path.exists(GT_path) or not os.path.exists(pred_path):
            continue

        #Create results sub-folder if it doesn't exist
        res_pred_dir = os.path.join(res_dir, GP_folder)
        if not os.path.exists(res_pred_dir):
            os.mkdir(res_pred_dir)

        # Get the list of GT and Prediction .tif files
        GT_file_list = [file for file in os.listdir(GT_path) if file.endswith('.tif')]
        pred_file_list = [file for file in os.listdir(pred_path) if file.endswith('.tif')]

        # Get the list of the paired files (both GT and Prediction .tif files)
        paired_files = list(set(GT_file_list) & set(pred_file_list))

        # Loop through the paired files
        for file in paired_files:
            GT_img = ski.io.imread(os.path.join(GT_path, file))
            pred_img = ski.io.imread(os.path.join(pred_path, file))
            start_time = perf_counter()
            
            # Check if the shape of the GT is bigger than the Prediction are the same and pad Prediction if not
            if GT_img.shape > pred_img.shape:
                print(f'{file} from {GP_folder} has shape {GT_img.shape} in GT and {pred_img.shape} in Prediction. Padded Prediction to match GT shape.')
                pred_img = pad_br_with_zeroes(GT_img, pred_img)

            # Check if the shape of the GT and Prediction images are the same
            if GT_img.shape == pred_img.shape:
                # Calculate the number of objects in each image and remap the labels
                GT_remap, _, _ = ski.segmentation.relabel_sequential(GT_img)
                pred_remap, _, _ = ski.segmentation.relabel_sequential(pred_img)
                GT_count = np.max(GT_remap)
                pred_count = np.max(pred_remap)
                
                # Print the number of objects in each image
                print(f'{file} from {GP_folder} has {GT_count} objects in GT and {pred_count} objects in Prediction')
                

                # Get Bounding Boxes coords for each GT object
                bbox_list = pd.DataFrame(ski.measure.regionprops_table(GT_remap, properties=['label', 'bbox'], spacing=(1,1)))

                # Initialize the true positives, false positives, and false negatives arrays
                true_positives = np.zeros_like(GT_img)
                false_positives = np.zeros_like(GT_img)
                false_negatives = np.zeros_like(GT_img)

                # Loop through all objects in GT and Prediction
                # For each object in GT
                for obj in range(1, GT_count+1):
                    # Get the bounding box for the current object
                    bbox_index = bbox_list.loc[bbox_list['label'] == obj].index[0]
                    bbox = bbox_list.loc[bbox_index, ['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']]

                    # Get the coordinates of the bounding box    
                    x1, y1, x2, y2 = bbox_points_for_crop(bbox, bbox['bbox-2'].max(), bbox['bbox-3'].max())

                    #Copy the remaped GT image and remap the current object in GT to 1 and make all others 0
                    GT_obj = GT_remap[x1:x2, y1:y2] 
                    GT_obj = GT_obj == obj

                    #Calculate the pixel coverage and object diameter values
                    GT_pixel_coverage = pixel_coverage_percent(GT_obj)
                    gt_min_w, gt_max_w, gt_mean_w, gt_median_w = object_diameter(GT_obj)
                    gt_area = GT_obj.sum()
                    gt_area_filled = ndimage.binary_fill_holes(GT_obj).sum()
                    gt_perimeter = perimeter(GT_obj)

                    #Add object information to lists
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

                    if len(np.unique(pred_obj_main)) == 1:                                    
                        false_negatives[GT_remap == obj] = obj

                        #Add object information to lists
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

                    else:
                        for p_obj in np.unique(pred_obj_main):
                            pred_obj = pred_obj_main == p_obj

                            intersection = np.logical_and(GT_obj, pred_obj)
                            union = np.logical_or(GT_obj, pred_obj)
                            iou_score =  np.sum(intersection) / np.sum(union)

                            if iou_score > 0.5:
                                #Calculate F1 score
                                f1_score = skl.f1_score(GT_obj, pred_obj, average='micro')

                                #Calculate pixel coverage percentage and object diameter for Prediction Label
                                pred_pixel_coverage = pixel_coverage_percent(pred_obj)
                                pred_min_d, pred_max_d, pred_mean_d, pred_median_d = object_diameter(pred_obj)
                                pred_area = pred_obj.sum()
                                pred_area_filled = ndimage.binary_fill_holes(pred_obj).sum()
                                pred_perimeter = perimeter(pred_obj)

                                #Add object to the true positives array and remove object from the remaped Prediction image
                                true_positives[pred_remap == p_obj] = obj
                                pred_remap[pred_remap == p_obj] = 0

                                #Add object information to lists
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

                                #Once a true positive is found, break out of the loop
                                break
                            
                            if p_obj == np.unique(pred_obj_main)[-1]:
                                #Add object information to lists
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


                #Store false positives in the array image
                false_positives[pred_remap != 0] = pred_remap[pred_remap != 0]

                #Save the images
                ski.io.imsave(os.path.join(res_pred_dir, file.split('.')[0] + '_true_positives.tif'), true_positives, check_contrast=False)
                ski.io.imsave(os.path.join(res_pred_dir, file.split('.')[0] + '_false_negatives.tif'), false_negatives, check_contrast=False)
                ski.io.imsave(os.path.join(res_pred_dir, file.split('.')[0] + '_false_positives.tif'), false_positives, check_contrast=False)

                #Get summary statistics
                file_for_count.append(file)
                folder_for_count.append(GP_folder)
                true_positives_count.append(len(np.unique(true_positives))-1)
                false_negatives_count.append(len(np.unique(false_negatives))-1)
                false_positives_count.append(len(np.unique(false_positives))-1)
                GT_count_count.append(GT_count)
                pred_count_count.append(pred_count)

            else:
                print(f'Error: {file} has different shape in GT and Prediction folders.')

            if start_time is not None:
                    print(f'Elapsed time: {strftime("%H:%M:%S", gmtime(perf_counter() - start_time))}')
                
    # Store Object properties in a dataframe
    IoU_per_obj_df['Grand_Parent_Folder'] = GP_folder_list
    IoU_per_obj_df['File_name'] = file_name_list
    IoU_per_obj_df['GT_Label'] = GT_label_list
    IoU_per_obj_df['Prediction_Label'] = pred_label_list
    IoU_per_obj_df['GT_Pixel_Coverage_Percent'] = GT_px_cov_list
    IoU_per_obj_df['Prediction_Pixel_Coverage_Percent'] = pred_px_cov_list
    IoU_per_obj_df['IoU'] = IoU_list
    IoU_per_obj_df['f1_score'] = f1_score_list

    # Store GT image properties in a dataframe
    IoU_per_obj_df['GT_diameter_min'] = GT_min_diameter
    IoU_per_obj_df['GT_diameter_max'] = GT_max_diameter
    IoU_per_obj_df['GT_diameter_mean'] = GT_mean_diameter
    IoU_per_obj_df['GT_diameter_median'] = GT_median_diameter
    IoU_per_obj_df['GT_area'] = GT_area_list
    IoU_per_obj_df['GT_area_filled'] = GT_area_filled_list
    IoU_per_obj_df['GT_perimeter'] = GT_perimeter_list

    # Dataframe calculations for GT images
    IoU_per_obj_df['GT_Circularity'] = ((4 * np.pi * IoU_per_obj_df['GT_area'].astype(float)) / IoU_per_obj_df['GT_perimeter'].astype(float) **2)
    IoU_per_obj_df['GT_Filledness'] = IoU_per_obj_df['GT_area'].astype(float) / IoU_per_obj_df['GT_area_filled'].astype(float)

    # Store Prediction image properties in a dataframe#
    IoU_per_obj_df['pred_diameter_min'] = pred_min_diameter
    IoU_per_obj_df['pred_diameter_max'] = pred_max_diameter
    IoU_per_obj_df['pred_diameter_mean'] = pred_mean_diameter
    IoU_per_obj_df['pred_diameter_median'] = pred_median_diameter
    IoU_per_obj_df['pred_area'] = pred_area_list
    IoU_per_obj_df['pred_area_filled'] = pred_area_filled_list
    IoU_per_obj_df['pred_perimeter'] = pred_perimeter_list

    # Dataframe calculations for Prediction images
    IoU_per_obj_df['pred_Circularity'] = ((4 * np.pi * IoU_per_obj_df['pred_area'].astype(float)) / IoU_per_obj_df['pred_perimeter'].astype(float) **2)
    IoU_per_obj_df['pred_Filledness'] = IoU_per_obj_df['pred_area'].astype(float) / IoU_per_obj_df['pred_area_filled'].astype(float)

    # Dataframe calculations for GT

    # Summary statistics per file
    summary_df = IoU_per_obj_df.groupby(['Grand_Parent_Folder', 'File_name']).agg('mean').reset_index()

    summary_df.drop(['GT_Label', 'Prediction_Label'], axis=1, inplace=True)

    count_df['Grand_Parent_Folder'] = folder_for_count
    count_df['File_name'] = file_for_count
    count_df['GT_count'] = GT_count_count
    count_df['pred_count'] = pred_count_count
    count_df['true_positives_count'] = true_positives_count
    count_df['false_negatives_count'] = false_negatives_count
    count_df['false_positives_count'] = false_positives_count

    summary_df = summary_df.merge(count_df, on=['Grand_Parent_Folder', 'File_name'], how='left')

    # Calculate summary Sensitivity/Recall and Accuracy
    summary_df['Sensitivity'] = summary_df['true_positives_count'] / (summary_df['true_positives_count'] + summary_df['false_negatives_count'])
    summary_df['Accuracy'] = summary_df['true_positives_count'] / (summary_df['true_positives_count'] + summary_df['false_positives_count'] + summary_df['false_negatives_count'])

    # Save summary statistics in csv file
    summary_df.to_csv(os.path.join(res_dir, directory.split(os.sep)[-1] +'_summary_stats.csv'))    

    # Save IoU per object statistics in csv file
    IoU_per_obj_df.to_csv(os.path.join(res_dir, directory.split(os.sep)[-1] +'_IoU_per_obj_stats.csv'))

    print('Done.')

    return summary_df, IoU_per_obj_df

def semantic_statistics(
        directory: str,
        res_dir: str
    ) -> Tuple:
    """
    Calculate the IoU, f1 score, and other statistics for each label in the semantic segmentation GT and Prediction images. Only for 2 labels + background.
    
    Args:
        directory (str): Directory with folders of sampling folders with GT and Prediction folder pairs inside.
        res_dir (str): Directory to save the results.
        obj_props_df (pd.DataFrame): A table containing the calculated properties for each region.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two dataframes. The first contains per object IoU and f1 score statistics. The second contains a summary of the statistics.
    """

    # Create a dictionary of parent folders and grand parent folders
    parent_folder_dict = {'Grand_Parent_Folder': ['upsampling_16', 'upsampling_8', 'upsampling_4', 'upsampling_2', 'OG', 'downsampling_2', 'downsampling_4', 'downsampling_8', 'downsampling_16'], 'Parent_Folder': ['GT', 'Prediction']}
    GP_dict = parent_folder_dict['Grand_Parent_Folder']

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
    for GP_folder in sorted(GP_dict):
        # Create the path variable to the GT and Prediction folders
        GT_path = os.path.join(directory, GP_folder, 'GT')
        pred_path = os.path.join(directory, GP_folder, 'Prediction')

        if not os.path.exists(GT_path) or not os.path.exists(pred_path):
            continue

        #Create results sub-folder if it doesn't exist
        res_pred_dir = os.path.join(res_dir, GP_folder)
        if not os.path.exists(res_pred_dir):
            os.mkdir(res_pred_dir)

        # Get the list of GT and Prediction .tif files
        GT_file_list = [file for file in os.listdir(GT_path) if file.endswith('.tif')]
        pred_file_list = [file for file in os.listdir(pred_path) if file.endswith('.tif')]

        # Get the list of the paired files (both GT and Prediction .tif files)
        paired_files = list(set(GT_file_list) & set(pred_file_list))

        # Loop through the paired files
        for file in paired_files:
            GT_img = ski.io.imread(os.path.join(GT_path, file))
            pred_img = ski.io.imread(os.path.join(pred_path, file))
            start_time = perf_counter()
            
            # Check if the shape of the GT is bigger than the Prediction are the same and pad Prediction if not
            if GT_img.shape > pred_img.shape:
                print(f'{file} from {GP_folder} has shape {GT_img.shape} in GT and {pred_img.shape} in Prediction. Padded Prediction to match GT shape.')
                pred_img = pad_br_with_zeroes(GT_img, pred_img)

            # Check if the shape of the GT and Prediction images are the same
            if GT_img.shape == pred_img.shape:
                # Calculate the number of objects in each image and remap the labels
                GT_count = np.max(GT_img)
                pred_count = np.max(pred_img)
                
                if GT_count !=2 or pred_count != 2 or GT_count != pred_count:
                    continue

                # Print the number of objects in each image
                print(f'Calculating Semantic Segmentation statistics for {file} from {GP_folder}.')

                for obj in range(1, 3):
                    # Extract the object from the GT
                    GT_obj = GT_img == obj
                    pred_obj = pred_img == obj

                    #Calculate IoU and f1 score
                    intersection = np.logical_and(GT_obj, pred_obj)
                    union = np.logical_or(GT_obj, pred_obj)

                    iou_score =  np.sum(intersection) / np.sum(union)
                    f1_score = skl.f1_score(GT_obj, pred_obj, average='micro')

                    #Add object information to lists
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
                        #Add object information to lists
                        GP_folder_list.append(GP_folder)
                        file_name_list.append(file)
                        GT_label_list.append('ALL')
                        pred_label_list.append('ALL')
                        IoU_list.append((iou_1 + iou_score) / 2)
                        f1_score_list.append((f1_1 + f1_score) / 2)

            else:
                print(f'Error: {file} has different shape in GT and Prediction folders.')

            if start_time is not None:
                    print(f'Elapsed time: {strftime("%H:%M:%S", gmtime(perf_counter() - start_time))}')
                
    #Store Object properties in a dataframe
    IoU_per_SS_df['Grand_Parent_Folder'] = GP_folder_list
    IoU_per_SS_df['File_name'] = file_name_list
    IoU_per_SS_df['GT_Label'] = GT_label_list
    IoU_per_SS_df['Prediction_Label'] = pred_label_list
    IoU_per_SS_df['IoU'] = IoU_list
    IoU_per_SS_df['f1_score'] = f1_score_list

    if len(GP_folder_list) != 0:
        # Save IoU per object statistics in csv file
        IoU_per_SS_df.to_csv(os.path.join(res_dir, directory.split(os.sep)[-1] +'_semantic_segmentation_stats.csv'))

        print('Done.')

        return IoU_per_SS_df

    else:
        print('No semantic segmentation images found.')
        return None
        


## Plot generating functions


def generate_basic_plot(
    res_dir: str, 
    dataframe: pd.DataFrame, 
    folder_sampling_dict: Dict[str, float], 
    column_to_plot: str, 
    kind_of_plot: Literal['split_violin', 'violin', 'box', 'box_no_outliers', 'strip', 'swarm'], 
    log_scale: bool, 
    hue: Optional[str] = 'Parent_Folder', 
    save: bool = True
) -> Optional[sns.FacetGrid]:
    """
    Generate a plot of the properties of the objects in the image.
    
    Args:
        res_dir (str): The directory to save the plot.
        dataframe (pd.DataFrame): A dataframe containing the properties of the objects in the image.
        folder_sampling_dict (Dict[str, float]): A dictionary of grandparent folders and their sampling multipliers. 
        column_to_plot (str): The column to plot on the y-axis.
        kind_of_plot (Literal['split_violin', 'violin', 'box', 'box_no_outliers', 'strip', 'swarm']): The kind of plot to generate. Use split_violin for a violin plot split by parent folder.
        log_scale (bool): Whether to use a logarithmic scale for the y-axis.
        hue (Optional[str]): The column to use for hue. Defaults to 'Parent_Folder'.
        save (bool): Whether to save the plot or return it. Defaults to True.
    
    Returns:
        Optional[sns.FacetGrid]: A plot of the properties in the column_to_plot of the objects in the image, or None if save is False.
    """
    # Re-order and remove unnecessary sampling folder names for x-axis
    order = order_axis_by_folder(folder_sampling_dict, dataframe)

    # Set seaborn plot theme and style - set the grid, ticks, and edge color
    sns.set_theme(context = 'talk', style = 'white', rc = {'axes.grid': True, 'xtick.bottom': True,'ytick.left': True, 'axes.edgecolor': 'black'}, palette = 'colorblind')

    # Generate plot, violin type plots have extra arguments
    if kind_of_plot == 'split_violin':
        kind_of_plot = 'violin'

        # In this type of violin plot each half of the violin represents a parent folder
        plot = sns.catplot(data = dataframe, x = 'Grand_Parent_Folder', y = column_to_plot, kind = kind_of_plot, hue = hue,  order = order, log_scale = log_scale, height = 7, aspect = 1.5, split = True, inner = 'quart')

    elif kind_of_plot == 'violin':
        # In this type of violin plot each parent folder has its own violin plot
        plot = sns.catplot(data = dataframe, x = 'Grand_Parent_Folder', y = column_to_plot, kind = kind_of_plot, hue = hue,  order = order, log_scale = log_scale, height = 7, aspect = 1.5, inner = 'quart')

    elif kind_of_plot == 'box_no_outliers':
        kind_of_plot = 'box'
        
        # In this type of box plot outliers are not shown
        plot = sns.catplot(data = dataframe, x = 'Grand_Parent_Folder', y = column_to_plot, kind = kind_of_plot, hue = hue,  order = order, log_scale = log_scale, height = 7, aspect = 1.5, showfliers=False, gap = 0.2)

    elif kind_of_plot in ['strip', 'swarm']:
        # increases size of dot in plot
        plot = sns.catplot(data = dataframe, x = 'Grand_Parent_Folder', y = column_to_plot, kind = kind_of_plot, hue = hue,  order = order, log_scale = log_scale, height = 7, aspect = 1.5, s = 200,edgecolor = 'black', linewidth = 2)
    
    else:
        # Generic plot creation for all other types
        plot = sns.catplot(data = dataframe, x = 'Grand_Parent_Folder', y = column_to_plot, kind = kind_of_plot, hue = hue,  order = order, log_scale = log_scale, height = 7, aspect = 1.5)

    # Customize the plot
    # Axis titles and Axis labels rotation 
    plt.xlabel('Sampling')
    plt.ylabel(column_to_plot.replace('_', ' '))
    plt.xticks(rotation = 30)

    # Move legend to top left and close plot
    if hue != None:
        sns.move_legend(plot, "lower center", bbox_to_anchor = (0.2, 0.83), ncol = 1,title = None, frameon = True)
    sns.despine(top = False, right = False) 

    if save == True:
        # Set and create folder to store graphs if it doesn't exist
        res_graph_dir = os.path.join(res_dir, 'Graphs')

        if not os.path.exists(res_graph_dir):
            os.mkdir(res_graph_dir)

        # Save plot as svg to allow for easy rescaling
        plot.savefig(res_graph_dir + os.sep + column_to_plot +'_' + kind_of_plot + '_plot.svg', dpi = 300, bbox_inches = 'tight')

    else:
        return plot

def box_strip_plot_from_csv(
    csv_1: str, 
    x_axis: str, 
    y_axis: str, 
    y_range: Optional[List[float]] = None, 
    og_px: Optional[int] = None, 
    save_dir: Optional[str] = None,  
    csv_2: Optional[str] = None
) -> None:
    """
    Generate a plot of the properties of the objects in the image from the saved csv.
    
    Args:
        csv_1 (str) : The input csv file path for graphing.
        x_axis (str): The column to use for the x-axis.
        y_axis (str): The column to use for the y-axis.
        y_range (Optional[List[float]]): The range of the y-axis.
        og_px (Optional[int]): The original pixel size of the image.
        save_dir (Optional[str]): If given graphs are saved in the specified directory.
        csv_2 (Optional[str]): The input csv file path for graphing - if needed, use this for the semantic segmentation data csv.
    """
    # Read the csv file
    primary_df = pd.read_csv(csv_1)
    csv_name = csv_1.split(os.sep)[-1].replace('.csv', '')
    hue: Optional[str] = None
    
    # Read the secondary csv file if it exists
    if csv_2 is None:
        main_df = primary_df

    else:
        secondary_df = pd.read_csv(csv_2)
        
        hue = 'Hue'

        # Add column to dataframe labeling the segmentation
        primary_df ['Hue'] = 'Binary Segmentation'

        # Select only the rows with the average of both labels for semantic segmentation
        secondary_df = secondary_df[(secondary_df['Label'] == 'All')]
        secondary_df ['Hue'] = 'Semantic Segementation'

        # Concatenate the two dataframes
        main_df = pd.concat([primary_df, secondary_df], ignore_index = True)

    box_strip_plot_from_df(
        dataframe=main_df, 
        x_axis=x_axis, 
        y_axis=y_axis, 
        y_range=y_range, 
        csv_name=csv_name, 
        hue=hue, 
        og_px=og_px, 
        save_dir=save_dir
    )

def box_strip_plot_from_df(
    dataframe: pd.DataFrame, 
    x_axis: str, 
    y_axis: str, 
    csv_name: str, 
    hue: Optional[str] = None, 
    y_range: Optional[List[float]] = None, 
    og_px: Optional[int] = None, 
    save_dir: Optional[str] = None
) -> Optional[sns.FacetGrid]:
    """
    Generate a plot of the properties of the objects in the image from the saved dataframe.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe containing the properties of the objects in the image.   
        x_axis (str): The column to use for the x-axis.
        y_axis (str): The column to use for the y-axis.
        csv_name (str): The name of the csv file.
        hue (Optional[str]): The column to use for the hue.
        og_px (Optional[int]): The original pixel size of the image.
        save_dir (Optional[str]): If given graphs are saved in the specified directory.
    
    Returns:
        plot (Optional[Figure]): A plot of the properties in the column_to_plot of the objects in the image.
        or
        None if the save_dir is given and the plot is saved in the specified directory
    """
    # Misc variables
    # Dictionary identifying sampling multipliers according to folder names
    folder_sampling_dict = {'upsampling_16': 16, 'upsampling_8': 8, 'upsampling_4': 4,'upsampling_2': 2, 'OG': 1, 'downsampling_2': 1/2, 'downsampling_4': 1/4, 'downsampling_8': 1/8, 'downsampling_16': 1/16}
    order = None

    # Color palettes
    pastel_colors = ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff', '#debb9b', '#fab0e4', '#cfcfcf', '#fffea3', '#b9f2f0']
    saturated_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Get order of sampling and calculate corresponding pixel size values
    if x_axis == 'Grand_Parent_Folder':
        order = order_axis_by_folder(folder_sampling_dict, dataframe)
    if og_px != None:
        order_px = [og_px/folder_sampling_dict[folder] for folder in order]

    sns.set_context('paper', rc={'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12, 'legend.title_fontsize': 12})

    # Arguments for plotting
    plot_args_box = {'data': dataframe, 
                'x': x_axis, 
                'y': y_axis,
                'kind':'box', 
                'height': 7, 
                'aspect' : 1, 
                'dodge':True , 
                'order': order, 
                'linecolor':'black', 
                'linewidth':1, 
                'fill':0.5,
                'whis':1.5 # 1.5 IQR
                }
    plot_args_strip = {'data': dataframe, 
                    'x': x_axis, 
                    'y': y_axis,
                    'dodge':True,
                    'linewidth':1,
                    'edgecolor':'black',
                    'legend':False,
                    'size': 7,
                    }

    if hue != None:
        plot_args_box['hue'] = hue
        plot_args_box['palette'] = pastel_colors
        plot_args_strip['hue'] = hue
        plot_args_strip['palette'] = saturated_colors
    else:
        plot_args_box['color'] = pastel_colors[0]
        plot_args_strip['color'] = saturated_colors[0]
    
    # Plot
    plot = sns.catplot(**plot_args_box)

    if x_axis == 'Grand_Parent_Folder':
        plt.axvline('OG', color='black', dashes=(2, 5))

    plt.grid(axis='y', which='major')
    plt.title(csv_name)
    #plt.tight_layout()

    if og_px != None:
        plt.xticks(order, order_px)
        plt.xlabel('Pixel size in microns')
    else:
        plt.xticks(rotation = 45, ha = 'right')
        plt.xlabel('Sampling')
    
    if y_range != None:
        plt.ylim(y_range)

    # Plot with boxplot
    if save_dir != None:
        # Save the plot
        plot.savefig(save_dir + csv_name + '_' + y_axis +'_plot.svg', dpi = 300, bbox_inches = 'tight',transparent=True)
   

    # Plot with stripplot
    sns.stripplot(**plot_args_strip)
    box_strip_plot = plot.figure

    if save_dir != None:
        # Save the plot
        plot.savefig(save_dir + csv_name + '_' + y_axis + '_plot_strip.svg', dpi = 300, bbox_inches = 'tight',transparent=True)
    
    return box_strip_plot


## Get data from PDFs

def get_metrics_from_pdfs(model_main_dir):

    # Define variables
    pdf_ends_with = '_training_report.pdf'

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
        if model in ['.DS_Store', '__pycache__']:
                continue
        else:
            model_dir = os.path.join(model_main_dir, model)

            if not os.path.isdir(model_dir):
                continue

            #print('Processing model: ', model)

            pdf_path = os.path.join(model_dir, model + pdf_ends_with)
            
            if os.path.exists(pdf_path):
                pdf = pypdf.PdfReader(pdf_path)
                page = pdf.pages[0].extract_text()

                # Extract epochs
                epochs = int(re.search(r'(\d+) epochs', page).group(1))
                n_epochs_list.append(epochs)

                # Extract paired image patches
                paired_image_patches = int(re.search(r'(\d+) paired image patches', page).group(1))
                n_paired_image_patches_list.append(paired_image_patches)

                # Extract dimensions
                dimensions_match = re.search(r"image\s*dimensions:\s*\((\d+),\s*(\d+)\)", page)
                if dimensions_match:
                    img_dimensions = (int(dimensions_match.group(1)), int(dimensions_match.group(2)))
                else:
                    img_dimensions = (0, 0)

                img_dimensions_list.append(img_dimensions)

                # Extract patch size
                patch_size_match = re.search(r'\((\d+),(\d+)\)\)', page)
                if patch_size_match:
                    patch_size = (int(patch_size_match.group(1)), int(patch_size_match.group(2)))
                else:
                    patch_size = (0, 0)

                patch_size_list.append(patch_size)

                # Extract training time
                for line in page.split('\n'):
                    if 'Training time: ' in line:
                        time_str = line.split(': ')[-1]
                        time = compact_time_string(time_str)
                        training_time_list.append(time)

                # Extract info from model name
                model_components = model.split('_')
                notebook.append(model_components[1])
                
                if model_components[3] == 'mix':
                    sampling.append(model_components[4])
                    sample.append(model_components[2] + '_' + model_components[3])
                else:
                    sampling.append(model_components[3])
                    sample.append(model_components[2])
                
            else:
                continue

    pdf_metrics_df['notebook'] = notebook
    pdf_metrics_df['sample'] = sample
    pdf_metrics_df['sampling'] = sampling
    pdf_metrics_df['n_epochs'] = n_epochs_list
    pdf_metrics_df['n_paired_image_patches'] = n_paired_image_patches_list
    pdf_metrics_df['img_dimensions'] = img_dimensions_list
    pdf_metrics_df['patch_size'] = patch_size_list
    pdf_metrics_df['training_time'] = training_time_list

    pdf_metrics_df.sort_values(by=['sample', 'img_dimensions'], ascending=True, ignore_index=True, inplace=True)

    pdf_metrics_df.to_csv(os.path.join(model_main_dir, 'pdf_metrics.csv'))

    print('DONE!')
    
    return pdf_metrics_df


## Miscellaneous functions


def parent_folder_dict(
        obj_props_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
    """
    Create a dictionary of parent folders and grand parent folders.
    
    Args:
        obj_props_df (pd.DataFrame): A table containing the calculated properties for each region.
        
    Returns:
        parent_folder_dict (Dict[str, List[str]]): A dictionary of parent folders and grand parent folders.
    """
    #Create blank dicitionary
    parent_folder_dict: Dict[str, List[str]] = {}

    # Get the unique folder names from the object properties dataframe
    for col in obj_props_df.columns.unique():
        if col.endswith('Folder'):
            parent_folder_dict[col] = obj_props_df[col].unique().tolist()

    return parent_folder_dict

def pixel_coverage_percent(
        img_array: np.ndarray
    ) -> float:
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
        bbox: List[int], 
        xmax: int, 
        ymax: int
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
    #Calculate the half the edge length of the box for padding
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

def object_diameter(image_array: np.array):

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
    median_diameter = np.median(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)]) * 2

    return min_diameter, max_diameter, mean_diameter, median_diameter

def pad_br_with_zeroes(gt_img: np.array, pred_img: np.array):
    """
    Calculate the padding size between the GT and Prediction images.
    
    Args:
        gt_img: A numpy array containing the GT image.
        pred_img: A numpy array containing the Prediction image.
        
    Returns:
        pad_with_zero: The padding size between the GT and Prediction images.
    """
    padded_pred = np.pad(pred_img, ((0, gt_img.shape[0]-pred_img.shape[0]), (0, gt_img.shape[1]-pred_img.shape[1])), 'constant', constant_values=0)

    return padded_pred

def order_axis_by_folder(folder_sampling_dict, dataframe):
    """
    Create ordered list of the multiplier folders for the x-axis.
    
    Args:
        folder_sampling_dict: A dictionary of grandparent folders and their sampling multipliers. 
        obj_props_df: A dataframe containing the properties of the objects in the image.
        
    """
    # Loop through all grandparent folders and add them to the order list if they are in the df
    order = [folder for folder in folder_sampling_dict.keys() if folder in dataframe['Grand_Parent_Folder'].unique()]

    return order

def get_csv_dict(main_directory: str) -> Dict[str, List[str]]:
    """
    Find all csv files in the latest Results folder of each folder in the input directory.

    Args:
        directory (str): The input directory containing the sub folders contating the image files.
                            directory |----> Dataset_Folder|----> Grandparent_Folder |----> Parent_Folder |----> Files
    """
    
    # Initialize a dictionary to store the csv files names
    csv_dict = {}

    directory_list = os.listdir(main_directory)

    for dir in directory_list:
        if dir in ['.DS_Store', '__pycache__', 'blank', 'main_folder']:
            continue
        else:
            curr_dir = os.path.join(main_directory, dir)

            # Create the Results folder path for the current directory
            res_dir = os.path.join(curr_dir, 'Results')
            base_res_dir = res_dir
            count = 1

            if not os.path.exists(res_dir):
                continue
            
            else:
                while os.path.exists(res_dir):
                    prev_res_dir = res_dir
                    res_dir = base_res_dir + '_' + f"{count:02d}"
                    count += 1
                
                res_dir = prev_res_dir

            # Find all csv files in the Results folder in the current directory
            csv_dict[dir] = [os.path.join(res_dir, f) for f in sorted(os.listdir(res_dir)) if f.endswith('.csv')]

    print('DONE!')

    return csv_dict

def compact_time_string(time_str: str) -> str:
    """
    Convert a string like 'x hour(s) y min(s) z sec(s)' to a HH:MM:SS string.

    Args:
        time_str: A string containing the time in the format 'x hour(s) y min(s) z sec(s)'

    Returns:
        A str representing the time in 'HH:MM:SS' format
    """

    # Split the time string into hours, minutes, and seconds, and remove the word 'hour(s)', 'min(s)', and 'sec(s)'
    parts = time_str.split()
    hours = float(parts[0].replace('hour(s)', ''))
    minutes = float(parts[1].replace('min(s)', ''))
    seconds = float(parts[2].replace('sec(s)', ''))
    
    return str(f'{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}')


## Region properties function and sub functions


def object_props(
        directory: str, 
        properties: Optional[List[str]] = ['label', 'area', 'eccentricity',  'perimeter', 'equivalent_diameter_area', 'axis_major_length', 'axis_minor_length', 'area_filled'],
        spacing: Optional[Tuple[float, float]] = None, 
        folder_sampling_dict: Optional[Dict[str, float]] = {'upsampling_16': 16, 'upsampling_8': 8, 'upsampling_4': 4,'upsampling_2': 2, 'OG': 1, 'downsampling_2': 1/2, 'downsampling_4': 1/4, 'downsampling_8': 1/8, 'downsampling_16': 1/16}
    ) -> pd.DataFrame:

    """
    Calculate the properties for each object in each image inthe input directory.
    
    Args:
        directory (str): The input directory containing the image files.
        properties (Optional[List[str]]): A list of the properties to calculated for each region.
        spacing (Optional[Tuple[float, float]]): The physical spacing of the image files.
        folder_sampling_dict (Optional[Dict[str, float]]): A dictionary of grandparent folders and their sampling multipliers.

    Returns:
        obj_props_df (pd.DataFrame): A table containing the calculated properties for each region, extraproperties, and including the file name and parent folder, normalized values by sampling for 'area', 'area_filled', 'equivalent_diameter_area', 'perimeter', 'axis_major_length', 'axis_minor_length'.
    """

    # Create a list to store the properties DataFrames
    props_list = []

    # Loop through all files in all subdirectories
    for root, dirs, files in os.walk(directory):
        if 'Results' not in root:

            # Loop through all files and add region properties/prediction statistics to a dataframe
            for file in files:
                if file.endswith(".tif"):
                    # Open image file
                    img_file = ski.io.imread(os.path.join(root, file))

                    start_time = perf_counter() # Start timer

                    # Calculate region properties
                    props = region_properties(img_file, properties = properties, spacing = spacing)
                    extra_properties(props)
                    add_file_name_to_dataframe(file, props)
                    add_parent_folder(props, given_dir = directory, root = root, folder_sampling_dict = folder_sampling_dict)
                    normalize_to_sampling(props, properties)  

                    # Append to the list of properties DataFrames
                    props_list.append(pd.DataFrame(props))

                    print(f"Properties calculated for {file} from {root.split(os.sep)[-1]} in {root.split(os.sep)[-2]} in {strftime('%H:%M:%S', gmtime(perf_counter() - start_time))} seconds")

    # Create the dataframe by concatenating the list of properties DataFrames
    obj_props_df = pd.concat(props_list, ignore_index=True)

    return obj_props_df

def region_properties(
        label_image: np.ndarray, 
        properties: List[str] = ['label', 'area', 'eccentricity',  'perimeter', 'equivalent_diameter_area', 'axis_major_length', 'axis_minor_length', 'area_filled'],
        spacing: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
    """
    Calculate properties of regions in the input file and return the results.
    
    Args:
        label_image: The input file path for region property calculation.
        properties: A list of properties to calculate for each region. Defaults to ['label', 'area', 'eccentricity',  'perimeter', 'equivalent_diameter_area', 'axis_major_length', 'axis_minor_length', 'area_filled']
        spacing: The physical spacing of the image files.
        
    Returns:
        IoU_per_obj_df: A table containing the calculated properties for each region.
    """
    
    IoU_per_obj_df = ski.measure.regionprops_table(
        label_image, 
        properties=properties, 
        spacing=spacing
    )
        
    return pd.DataFrame(IoU_per_obj_df)

def add_file_name_to_dataframe(
        file: str, 
        IoU_per_obj_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Add the file name to the dataframe.
    
    Args:
        file (str): The input file name used for region property calculation.
        IoU_per_obj_df (pd.DataFrame): A table containing the calculated properties for each region.
        
    Returns:
        IoU_per_obj_df (pd.DataFrame): Original table with added column for file name.
    """
    IoU_per_obj_df['File_name'] = file

    return IoU_per_obj_df

def extra_properties(
        IoU_per_obj_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculate roundness and expected roundness of regions in the input file and return the results, only if the 'area', 'equivalent_diameter_area' and 'perimeter' columns exist in the input table.
    
    Args:
        IoU_per_obj_df (pd.DataFrame): A table containing the calculated properties for each region.
        
    Returns:
        IoU_per_obj_df (pd.DataFrame:) Original table with added columns for circularity(4*pi*area/perimeter^2), roundness (minor axis/major axis), filledness (area/area_filled). If the required original columns are present.
    
    """
    if 'area' in IoU_per_obj_df.columns and 'perimeter' in IoU_per_obj_df.columns:
        IoU_per_obj_df['Circularity'] = 4*np.pi*IoU_per_obj_df['area'].astype(float)/IoU_per_obj_df['perimeter'].astype(float)**2

    if 'axis_major_length' in IoU_per_obj_df.columns and 'axis_minor_length' in IoU_per_obj_df.columns:
        IoU_per_obj_df['Roundness'] = IoU_per_obj_df['axis_minor_length'].astype(float)/IoU_per_obj_df['axis_major_length'].astype(float)

    if 'area' in IoU_per_obj_df.columns and 'area_filled' in IoU_per_obj_df.columns:
        IoU_per_obj_df['Filledness'] = IoU_per_obj_df['area'].astype(float)/IoU_per_obj_df['area_filled'].astype(float)

    if 'area' in IoU_per_obj_df.columns:
        IoU_per_obj_df['Pixel_Coverage_Percent'] = ( 1 / IoU_per_obj_df['area'].astype(float) ) * 100
    
    return IoU_per_obj_df

def add_parent_folder(
        IoU_per_obj_df: pd.DataFrame, 
        given_dir: str, 
        root: str, 
        folder_sampling_dict: Dict[str, float]
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
    folder_col_name: str = 'Parent_Folder'
    root_depth: int = len(root.split(os.sep))
    dir_depth: int = len(given_dir.split(os.sep))

    # loop through the parent folders and add them to the dataframe
    while (root_depth > (dir_depth + depth_count)):
        IoU_per_obj_df[folder_col_name] = root.split(os.sep)[-depth_count-1]

        # Add the sampling multiplier based on folder name
        for folder, sampling in folder_sampling_dict.items():
            if root.split(os.sep)[-depth_count-1] == folder:
                IoU_per_obj_df['sampling_multiplier'] = sampling

        depth_count += 1
        folder_col_name = 'Grand_' + folder_col_name

    return IoU_per_obj_df

def normalize_to_sampling(
        IoU_per_obj_df: pd.DataFrame, 
        properties: List[str]
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
        if property == 'area' or property == 'area_filled':
            IoU_per_obj_df['norm_' + property] = IoU_per_obj_df[property] / IoU_per_obj_df['sampling_multiplier'] **2
                
        if any(x in property for x in [ 'perimeter', 'axis_major_length', 'axis_minor_length', 'equivalent_diameter_area']):
            IoU_per_obj_df['norm_' + property] = IoU_per_obj_df[property] / IoU_per_obj_df['sampling_multiplier']

    return IoU_per_obj_df

## Deprecated Functions


def prediction_statistics(parent_folder_dict, directory):
    """
    Per image IoU and f1 score statistics.
    
    Args:
        parent_folder_dict: A dictionary with the listof parent folders and grand parent folders.
        
    Returns:
        pred_stats_df: A dataframe containing per object statistics.
    """
    if len(parent_folder_dict.keys()) > 2:
        return print("Error: More than two parent folders found.")
    
    # Create a dataframe to store the results
    pred_stats_df = pd.DataFrame([])
    GP_dict = parent_folder_dict['Grand_Parent_Folder']

    for GP_folder in sorted(GP_dict):
        # Create the path variable to the GT and Prediction folders
        GT_path = os.path.join(directory, GP_folder, 'GT')
        pred_path = os.path.join(directory, GP_folder, 'Prediction')

        # Get the list of GT and Prediction .tif files
        GT_file_list = [file for file in os.listdir(GT_path) if file.endswith('.tif')]
        pred_file_list = [file for file in os.listdir(pred_path) if file.endswith('.tif')]

        # Get the list of the paired files (both GT and Prediction .tif files)
        paired_files = list(set(GT_file_list) & set(pred_file_list))

        for file in paired_files:
            GT_img = ski.io.imread(os.path.join(GT_path, file))
            pred_img = ski.io.imread(os.path.join(pred_path, file))

            if GT_img.shape > pred_img.shape:
                print(f'{file} from {GP_folder} has shape {GT_img.shape} in GT and {pred_img.shape} in Prediction. Padded Prediction to GT shape.')
                pred_img = pad_br_with_zeroes(GT_img, pred_img)

            if GT_img.shape == pred_img.shape:
                
                if GT_img.max() > 2:
                    GT_img_bin = GT_img
                    GT_img_bin[GT_img_bin > 0] = 1
                    pred_img_bin = pred_img
                    pred_img_bin[pred_img_bin > 0] = 1
                
                    IoU = skl.jaccard_score(GT_img_bin, pred_img_bin, average='micro')

                    f1_score = skl.f1_score(GT_img_bin, pred_img_bin, average='micro')

                    # Add the results to the dataframe

                    pred_stats_df = pd.concat([pred_stats_df, pd.DataFrame([{'Grand_Parent_Folder': GP_folder, 'File_name': file, 'IoU': IoU, 'f1_score': f1_score}])], ignore_index=True)

                else:
                    for label in range(1, GT_img.max() + 1):
                        GT_img_bin = GT_img.copy()
                        GT_img_bin[GT_img_bin != label] = 0
                        GT_img_bin[GT_img_bin == label] = 1
                        pred_img_bin = pred_img.copy()
                        pred_img_bin[pred_img_bin != label] = 0
                        pred_img_bin[pred_img_bin == label] = 1

                        IoU = skl.jaccard_score(GT_img_bin, pred_img_bin, average='micro')

                        f1_score = skl.f1_score(GT_img_bin, pred_img_bin, average='micro')

                        if label == 1:
                            iou_1 = IoU
                            f1_1 = f1_score

                        # Add the results to the dataframe

                        pred_stats_df = pd.concat([pred_stats_df, pd.DataFrame([{'Grand_Parent_Folder': GP_folder, 'File_name': file, 'Label': label, 'IoU': IoU, 'f1_score': f1_score}])], ignore_index=True)

                        if label == 2:
                            IoU = (iou_1 + IoU) /2
                            f1_score = (f1_1 + f1_score) /2

                            pred_stats_df = pd.concat([pred_stats_df, pd.DataFrame([{'Grand_Parent_Folder': GP_folder, 'File_name': file, 'Label': 'All', 'IoU': IoU, 'f1_score': f1_score}])], ignore_index=True)

            else:
                print(f'Error: {file} has different shape in GT and Prediction folders.')

    return pred_stats_df