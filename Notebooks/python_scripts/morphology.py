# Functions for morphology analysis of label images

#Import required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skimage as ski
from sklearn import metrics as skl

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

## Morphology functions end

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

                pred_stats_df = pd.concat([pred_stats_df, pd.DataFrame([{'Grand_Parent_Folder': GP_folder, 'File': file, 'IoU': IoU, 'f1_score': f1_score}])], ignore_index=True)

            else:
                print(f'Error: {file} has different shape in GT and Prediction folders.')

    return pred_stats_df
        
## Prediction statistics functions end

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