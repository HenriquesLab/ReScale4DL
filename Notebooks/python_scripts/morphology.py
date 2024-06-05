# Functions for morphology analysis of label images

#Import required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skimage as ski
from sklearn import metrics as skl
from time import perf_counter, strftime, gmtime
from scipy import ndimage

## Region properties functions start
def region_properties(label_image: np.ndarray, properties, spacing=None):
    """
    Calculate properties of regions in the input file and return the results.
    
    Args:
        label_image: The input file path for region property calculation.
        properties: A list of properties to calculate for each region. Defaults to ['label', 'area', 'eccentricity',  'perimeter', 'equivalent_diameter_area', 'axis_major_length', 'axis_minor_length', 'area_filled']
        
    Returns:
        props: A table containing the calculated properties for each region.
    """
    
    props_dataframe = ski.measure.regionprops_table(label_image, properties=properties, spacing=spacing)
        
    return pd.DataFrame(props_dataframe)

def add_file_name_to_dataframe(file: str, props_dataframe: pd.DataFrame):
    """
    Add the file name to the dataframe.
    
    Args:
        file: The input file path for region property calculation.
        props_dataframe: A table containing the calculated properties for each region.
        
    Returns:
        props_dataframe: Original table with added column for file name.
    """
    props_dataframe['File_name'] = os.path.basename(file)
    return props_dataframe

def extra_properties(props_dataframe: pd.DataFrame):
    """
    Calculate roundness and expected roundness of regions in the input file and return the results, only if the 'area', 'equivalent_diameter_area' and 'perimeter' columns exist in the input table.
    
    Args:
        props_dataframe: A table containing the calculated properties for each region.
        
    Returns:
        props_dataframe: Original table with added columns for circularity(4*pi*area/perimeter^2), roundness (minor axis/major axis), filledness (area/area_filled). If the required original columns are present.
    
    """
    if 'area' in props_dataframe.columns and 'perimeter' in props_dataframe.columns:
        props_dataframe['Circularity'] = 4*np.pi*props_dataframe['area']/props_dataframe['perimeter']**2

    if 'axis_major_length' in props_dataframe.columns and 'axis_minor_length' in props_dataframe.columns:
        props_dataframe['Roundness'] = props_dataframe['axis_minor_length']/props_dataframe['axis_major_length']

    if 'area' in props_dataframe.columns and 'area_filled' in props_dataframe.columns:
        props_dataframe['Filledness'] = props_dataframe['area']/props_dataframe['area_filled']

    if 'area' in props_dataframe.columns:
        props_dataframe['Pixel_Coverage_Percent'] = ( 1 / props_dataframe['area'] ) * 100
    
    return props_dataframe

def add_parent_folder(props_dataframe, given_dir, root, folder_sampling_dict):
    """
    Loop through the parent folders and add them to the dataframe.
    
    Args:
        props_dataframe: A table containing the calculated properties for each region.
        given_dir: Directory containing the image files.
        root: Current os.walk directory
        
    Returns:
        props_dataframe: Original table with added column(s) for parent folder(s).
    """
    # Default function variables
    depth_count = 0
    folder_col_name = 'Parent_Folder'
    root_depth = len(root.split(os.sep))
    dir_depth = len(given_dir.split(os.sep))

    # loop through the parent folders and add them to the dataframe
    while (root_depth > (dir_depth + depth_count)):
        props_dataframe[folder_col_name] = root.split('/')[-depth_count-1]

        # Add the sampling multiplier based on folder name
        for folder, sampling in folder_sampling_dict.items():
            if root.split('/')[-depth_count-1] == folder:
                props_dataframe['sampling_multiplier'] = sampling

        depth_count += 1
        folder_col_name = 'Grand_' + folder_col_name

    return props_dataframe

def normalize_to_sampling(props_dataframe, properties):
    """
    Normalize the dataframe to the given folder sampling.
    
    Args:
        props_dataframe: A table containing the calculated properties for each region.
        properties: A list of the properties to calculated for each region.
        
    Returns:
        props_dataframe: Original table with normalized columns for 'area', 'area_filled', 'equivalent_diameter_area', 'perimeter', 'axis_major_length', 'axis_minor_length'.
    """
    for property in properties:
        if property == 'area' or property == 'area_filled':
            props_dataframe['norm_' + property] = props_dataframe[property] / props_dataframe['sampling_multiplier'] **2
                
        if any(x in property for x in [ 'perimeter', 'axis_major_length', 'axis_minor_length', 'equivalent_diameter_area']):
            props_dataframe['norm_' + property] = props_dataframe[property] / props_dataframe['sampling_multiplier']

    return props_dataframe

def object_props(directory, properties, spacing, folder_sampling_dict):
    """
    Calculate the properties for each object in the input directory.
    
    Args:
        directory: The input directory containing the image files.
        properties: A list of the properties to calculated for each region.
        spacing: The physical spacing of the image files.
        folder_sampling_dict: A dictionary of grandparent folders and their sampling multipliers.   
    """
    # Create blank dataframe to store statistics of all images
    obj_props_df = pd.DataFrame([])

    # Loop through all files in all subdirectories
    for root, dirs, files in os.walk(directory):
        if 'Results' not in root:

            # Loop through all files and add region properties/prediction statistics to a dataframe
            for file in files:
                if file.endswith(".tif"):
                    # Open image file
                    img_file = ski.io.imread(os.path.join(root, file))

                    # Calculate region properties
                    props = region_properties(img_file, properties = properties, spacing = spacing)
                    extra_properties(props)
                    add_file_name_to_dataframe(os.path.join(root, file), props)
                    add_parent_folder(props, given_dir = directory, root = root, folder_sampling_dict = folder_sampling_dict)
                    normalize_to_sampling(props, properties)  

                    # Add region properties to dataframe
                    obj_props_df = pd.concat([obj_props_df, pd.DataFrame(props)], ignore_index=True)

    return obj_props_df



##Prediction statistics functions start

def parent_folder_dict(obj_props_df):
    """
    Create a list of parent folders and grand parent folders.
    
    Returns:
        parent_folder_list: A list of parent folders and grand parent folders.
    """
    parent_folder_dict = {}
    for col in obj_props_df.columns.unique():
        if col.endswith('Folder'):
            parent_folder_dict[col] = obj_props_df[col].unique()

    return parent_folder_dict

def prediction_statistics(parent_folder_dict, directory):
    """
    Calculate the number of files in each parent folder and grand parent folder.
    
    Args:
        parent_folder_dict: A dictionary of parent folders and grand parent folders.
        
    Returns:
        parent_folder_df: A dataframe containing the number of files in each parent folder and grand parent folder. 
    """
    if len(parent_folder_dict.keys()) > 2:
        return print("Error: More than two parent folders found.")
    
    # Create a dataframe to store the results
    pred_stats_df = pd.DataFrame([])
    GP_dict = parent_folder_dict.popitem()

    for GP_folder in sorted(GP_dict[1]):
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

            if GT_img.shape == pred_img.shape:

                GT_img_bin = GT_img
                GT_img_bin[GT_img_bin > 0] = 1
                pred_img_bin = pred_img
                pred_img_bin[pred_img_bin > 0] = 1
            
                IoU = skl.jaccard_score(GT_img_bin, pred_img_bin, average='micro')

                f1_score = skl.f1_score(GT_img_bin, pred_img_bin, average='micro')

                # Add the results to the dataframe

                pred_stats_df = pd.concat([pred_stats_df, pd.DataFrame([{'Grand_Parent_Folder': GP_folder, 'File_name': file, 'IoU': IoU, 'f1_score': f1_score}])], ignore_index=True)

            else:
                print(f'Error: {file} has different shape in GT and Prediction folders.')

    return pred_stats_df

def per_object_statistics(directory, res_dir, obj_props_df):
    """
    Calculate the IoU for each object in the image.
    
    Args:
        parent_folder_dict: A dictionary of parent folders and grand parent folders.
        obj_props_df: A dataframe containing the properties of the objects in the image.
    """

    # Create a dictionary of parent folders and grand parent folders
    parent_folder_dict_1 = parent_folder_dict(obj_props_df)

    if len(parent_folder_dict_1.keys()) > 2:
        return print("Error: More than two parent folders found.")
    
    # Create dataframes to store the results
    IoU_per_obj_df = pd.DataFrame([])
    summary_df = pd.DataFrame([])
    start_time = None

    GP_dict = parent_folder_dict_1.popitem()
    #GP_dict = [[''],['downsampling_16']]

    for GP_folder in sorted(GP_dict[1]):
            # Create the path variable to the GT and Prediction folders
            GT_path = os.path.join(directory, GP_folder, 'GT')
            pred_path = os.path.join(directory, GP_folder, 'Prediction')

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

                # Check if the shape of the GT and Prediction images are the same
                if GT_img.shape == pred_img.shape:
                    # Calculate the number of objects in each image and remap the labels
                    GT_remap, GT_count = ski.measure.label(GT_img, background=0, return_num=True)
                    pred_remap, pred_count = ski.measure.label(pred_img, background=0, return_num=True)

                    # If any of the label images are empty (only background), skip the file and move on to the next one
                    ##if pred_count == 0 or GT_count == 0:
                        ##print(f'{file} from {GP_folder} has {GT_count} objects in GT and {pred_count} objects in Prediction. Skipping...')
                        #continue

                    # Get Bounding Boxes coords for each GT object
                    bbox_list = pd.DataFrame(ski.measure.regionprops_table(GT_remap, properties=['label', 'bbox'], spacing=(1,1)))
                    #bbox_list.to_csv(os.path.join(res_pred_dir, file[:-4]+'_BBox.csv'))

                    # Initialize the true positives, false positives, and false negatives arrays
                    true_positives = np.zeros_like(GT_img)
                    false_positives = np.zeros_like(GT_img)
                    false_negatives = np.zeros_like(GT_img)

                    # Lists to store information about each object, reset at the start of each image
                    GP_folder_list = []
                    file_name_list = []
                    GT_label_list = []
                    pred_label_list = []
                    GT_px_cov_list = []
                    pred_px_cov_list = []
                    IoU_list = []
                    f1_score_list = []

                    GT_min_width = []
                    GT_max_width = []
                    GT_mean_width = []
                    GT_median_width = []

                    pred_min_width = []
                    pred_max_width = []
                    pred_mean_width = []
                    pred_median_width = []

                    # Reset temp dataframe
                    temp_df = pd.DataFrame([])

                    if start_time is not None:
                        print(f'Elapsed time: {strftime("%H:%M:%S", gmtime(perf_counter() - start_time))}')
                    print(f'{file} from {GP_folder} has {GT_count} objects in GT and {pred_count} objects in Prediction')
                    start_time = perf_counter()

                    # Loop through all objects in GT and Prediction
                    #For each object in GT
                    for obj in range(1, GT_count+1):
                        # Get the bounding box for the current object
                        bbox_index = bbox_list.loc[bbox_list['label'] == obj].index[0]
                        bbox = bbox_list.loc[bbox_index, ['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']]

                        # Get the coordinates of the bounding box    
                        x1, x2, y1, y2 = bbox_points_for_crop(bbox, bbox['bbox-2'].max(), bbox['bbox-3'].max())

                        #Copy the remaped GT image and remap the current object in GT to 1 and make all others 0
                        GT_obj = GT_remap[x1:x2, y1:y2] #GT_obj = GT_obj == obj #GT_obj = GT_obj[x1:x2+10, y1-10:y2+10]
                        GT_obj = GT_obj == obj

                        #Calculate the pixel coverage and object width values
                        GT_pixel_coverage = pixel_coverage_percent(GT_obj)
                        gt_min_w, gt_max_w, gt_mean_w, gt_median_w = object_width(GT_obj)

                        #Add object information to lists
                        GP_folder_list.append(GP_folder)
                        file_name_list.append(file)
                        GT_label_list.append(obj)
                        GT_px_cov_list.append(GT_pixel_coverage)
                        GT_min_width.append(gt_min_w)
                        GT_max_width.append(gt_max_w)
                        GT_mean_width.append(gt_mean_w)
                        GT_median_width.append(gt_median_w)

                        if pred_count == 0:
                            pred_label_list.append(0)
                            pred_px_cov_list.append(0)
                            IoU_list.append(0)
                            f1_score_list.append(0)

                            pred_min_width.append(0)
                            pred_max_width.append(0)
                            pred_mean_width.append(0)
                            pred_median_width.append(0)

                            continue

                        # Compare the current object in GT to all objects in Prediction
                        for p_obj in range(1, pred_count+1):
                            #Only if the object still exists in the remaped Prediction image
                            if p_obj in pred_remap:
                                #Copy the remaped Prediciton image and remap the current object in pred to 1 and make all others 0
                                pred_obj = pred_remap[x1:x2, y1:y2]
                                pred_obj = pred_obj == p_obj

                                if pred_obj.sum() != 0:
                                    #Calculate the IoU for the current objects
                                    #iou_score= skl.jaccard_score(GT_obj, pred_obj, average='micro')
                                    intersection = np.logical_and(GT_obj, pred_obj)
                                    union = np.logical_or(GT_obj, pred_obj)
                                    iou_score =  np.sum(intersection) / np.sum(union)
                                    
                                    #If the IoU is greater than 0.5 it is considered as true positive
                                    if iou_score > 0.5:
                                        f1_score = skl.f1_score(GT_obj, pred_obj, average='micro')

                                        #Calculate pixel coverage percentage for Prediction Label
                                        pred_pixel_coverage = pixel_coverage_percent(pred_obj)
                                        pred_min_w, pred_max_w, pred_mean_w, pred_median_w = object_width(pred_obj)

                                        #Add object to the true positives array and remove object from the remaped Prediction image
                                        true_positives[pred_remap == p_obj] = obj
                                        pred_remap[pred_remap == p_obj] = 0

                                        #Add object information to lists
                                        pred_label_list.append(p_obj)
                                        pred_px_cov_list.append(pred_pixel_coverage)
                                        IoU_list.append(iou_score)
                                        f1_score_list.append(f1_score)
                                        pred_min_width.append(pred_min_w)
                                        pred_max_width.append(pred_max_w)
                                        pred_mean_width.append(pred_mean_w)
                                        pred_median_width.append(pred_median_w)

                                        #Once a true positive is found, break out of the loop
                                        break

                            if p_obj == pred_count:                                    
                                false_negatives[GT_remap == obj] = obj

                                #Add object information to lists
                                pred_label_list.append(0)
                                pred_px_cov_list.append(0)
                                IoU_list.append(0)
                                f1_score_list.append(0)

                                pred_min_width.append(0)
                                pred_max_width.append(0)
                                pred_mean_width.append(0)
                                pred_median_width.append(0)

                                #IoU_per_obj_df = pd.concat([IoU_per_obj_df, pd.DataFrame([{'Grand_Parent_Folder': GP_folder, 'File_name': file,'GT_Label': obj, 'Prediction_Label': None, 'GT_Pixel_Coverage_Percent': GT_pixel_coverage, 'Prediction_Pixel_Coverage_Percent': 0, 'IoU': 0, 'f1_score': 0}])], ignore_index=True)
                    """ # Pad pred_label list if necessary
                    while len(pred_label_list) < len(GT_label_list):
                        pred_label_list.append(0)
                        pred_px_cov_list.append(0)
                        IoU_list.append(0)
                        f1_score_list.append(0) """

                    #Store false positives in the array image
                    false_positives[pred_remap != 0] = pred_remap[pred_remap != 0]

                    #Save the images
                    ski.io.imsave(os.path.join(res_pred_dir, file.split('.')[0] + '_true_positives.tif'), true_positives, check_contrast=False)
                    ski.io.imsave(os.path.join(res_pred_dir, file.split('.')[0] + '_false_negatives.tif'), false_negatives, check_contrast=False)
                    ski.io.imsave(os.path.join(res_pred_dir, file.split('.')[0] + '_false_positives.tif'), false_positives, check_contrast=False)

                    #Get summary statistics
                    true_positives, true_positives_count = ski.measure.label(true_positives, background=0, return_num=True)
                    false_negatives, false_negatives_count = ski.measure.label(false_negatives, background=0, return_num=True)
                    false_positives, false_positives_count = ski.measure.label(false_positives, background=0, return_num=True)

                    #Store Object properties in a dataframe
                    temp_df['Grand_Parent_Folder'] = GP_folder_list
                    temp_df['File_name'] = file_name_list
                    temp_df['GT_Label'] = GT_label_list
                    temp_df['Prediction_Label'] = pred_label_list
                    temp_df['GT_Pixel_Coverage_Percent'] = GT_px_cov_list
                    temp_df['Prediction_Pixel_Coverage_Percent'] = pred_px_cov_list
                    temp_df['IoU'] = IoU_list
                    temp_df['f1_score'] = f1_score_list

                    temp_df['GT_width_min'] = GT_min_width
                    temp_df['GT_width_max'] = GT_max_width
                    temp_df['GT_width_mean'] = GT_mean_width
                    temp_df['GT_width_median'] = GT_median_width

                    temp_df['pred_width_min'] = pred_min_width
                    temp_df['pred_width_max'] = pred_max_width
                    temp_df['pred_width_mean'] = pred_mean_width
                    temp_df['pred_width_median'] = pred_median_width

                    IoU_per_obj_df = pd.concat([IoU_per_obj_df, temp_df])
                    
                    #IoU_per_obj_df = pd.concat([IoU_per_obj_df, pd.DataFrame([{'Grand_Parent_Folder': GP_folder_list, 'File_name': file_name_list,'GT_Label': GT_label_list, 'Prediction_Label': pred_label_list, 'GT_Pixel_Coverage_Percent': GT_px_cov_list, 'Prediction_Pixel_Coverage_Percent': pred_px_cov_list, 'IoU': IoU_list, 'f1_score': f1_score_list}])])

                    filtered_df = IoU_per_obj_df[(IoU_per_obj_df['File_name'] == file) & (IoU_per_obj_df['Grand_Parent_Folder'] == GP_folder)]
                    mean_iou = filtered_df['IoU'].mean()
                    mean_f1 = filtered_df['f1_score'].mean()
                    
                    #Add summary statistics to the summary dataframe
                    summary_df = pd.concat([summary_df, pd.DataFrame([{'Grand_Parent_Folder': GP_folder, 'File_name': file, 'GT_count': GT_count, 'pred_count': pred_count, 'true_positives_count': true_positives_count, 'false_negatives_count': false_negatives_count, 'false_positives_count': false_positives_count, 'Mean_IoU': mean_iou , 'Mean_f1_score': mean_f1}])], ignore_index=True)

                    #break

                else:
                    print(f'Error: {file} has different shape in GT and Prediction folders.')
            #break

    # Calculate summary Sensitivity/Recall and Accuracy
    summary_df['Sensitivity'] = summary_df['true_positives_count'] / (summary_df['true_positives_count'] + summary_df['false_negatives_count'])
    summary_df['Accuracy'] = summary_df['true_positives_count'] / (summary_df['true_positives_count'] + summary_df['false_positives_count'] + summary_df['false_negatives_count'])

    return summary_df, IoU_per_obj_df


## Plotting functions start

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

def generate_basic_plot(res_dir: str, dataframe: pd.DataFrame, folder_sampling_dict: dict, column_to_plot: str , kind_of_plot: str, log_scale: bool, hue: str = 'Parent_Folder', save: bool = True):
    """
    Generate a plot of the properties of the objects in the image.
    
    Args:
        obj_props_df: A dataframe containing the properties of the objects in the image.
        folder_sampling_dict: A dictionary of grandparent folders and their sampling multipliers. 
        column_to_plot: The column to plot on the y-axis.
        kind_of_plot: The kind of plot to generate. Use split_violin for a violin plot split by parent folder.
        log_scale: Whether to use a logarithmic scale for the y-axis.
        save: Whether to save the plot or return it.
    
    Returns:
        plot: A plot of the properties in the column_to_plot of the objects in the image.

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
        plot.savefig(res_graph_dir + '/' + column_to_plot +'_' + kind_of_plot + '_plot.svg', dpi = 300, bbox_inches = 'tight')

    else:
        return plot
    

## Miscellaneous functions start

def pixel_coverage_percent(img_array: np.array):    
    """
    Calculate the pixel coverage percentage of the input image array.
    
    Args:
        image_array: A numpy/dataframe image array.
        
    Returns:
        pixel_coverage_percent: The percentage of the object that each pixel covers.

    """
    obj_area = np.sum(img_array)
    pixel_coverage = 1 / obj_area * 100

    return pixel_coverage

def bbox_points_for_crop(bbox, xmax ,ymax):
    """
    Calculate the top left and bottom right points of the bounding box.
    
    Args:
        bbox: A list containing the x and y coordinates of the top left and bottom right points of the bounding box.
        
    Returns:
        top_left: A list containing the x and y coordinates of the top left point of the bounding box.
        bottom_right: A list containing the x and y coordinates of the bottom right point of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    x_radius = (x2 - x1 + 2) // 2
    y_radius = (y2 - y1 + 2) // 2

    x1 = (x1 - x_radius) if (x1 - x_radius) > 0 else 0 # if x1 - x_radius > 0
    y1 = (y1 - y_radius) if (y1 - y_radius) > 0 else 0 # if y1 - y_radius > 0
    x2 = (x2 + x_radius) if (x2 + x_radius) < xmax else xmax # if x2 + x_radius < xmax
    y2 = (y2 + y_radius) if (y2 + y_radius) < ymax else ymax # if y2 + y_radius < ymax


    return x1, x2, y1, y2

def object_width(image_array: np.array):
    """
    Calculate the width of the object in the image array.
    
    Args:
        image_array: A numpy/dataframe image array with a single object
        
    Returns:
        min_width: The minimum width of the object in the image array.
        max_width: The maximum width of the object in the image array.
        mean_width: The mean width of the object in the image array.
        median_width: The median width of the object in the image array.
    """
    # Calculate the object skeleton and Euclidean distance transform
    obj_skeleton = ski.morphology.skeletonize(image_array)
    obj_edt = ndimage.distance_transform_edt(image_array)
    
    # Get the EDT values for the object skeleton
    obj_skeleton_edt = obj_skeleton * obj_edt

    # Calculate the min, max, mean, and median width excluding the zero values of the background
    min_width = np.min(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)])
    max_width = np.max(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)])
    mean_width = np.mean(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)])
    median_width = np.median(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)])

    return min_width, max_width, mean_width, median_width