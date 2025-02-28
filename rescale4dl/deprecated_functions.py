#Import required libraries
import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import skimage as ski # type: ignore
from skimage.measure._regionprops_utils import perimeter # type: ignore
from sklearn import metrics as skl # type: ignore
from time import perf_counter, strftime, gmtime
from scipy import ndimage # type: ignore
from typing import List, Optional, Tuple, Dict, Literal, Union
from .morphology import pad_br_with_zeroes


## Deprecated Functions


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

    # set plot color palette if hue is given or not
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

def generate_basic_plot(
    result_dir: str, 
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
        result_dir (str): The directory to save the plot.
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
        res_graph_dir = os.path.join(result_dir, 'Graphs')

        if not os.path.exists(res_graph_dir):
            os.mkdir(res_graph_dir)

        # Save plot as svg to allow for easy rescaling
        plot.savefig(res_graph_dir + os.sep + column_to_plot +'_' + kind_of_plot + '_plot.svg', dpi = 300, bbox_inches = 'tight')

    else:
        return plot
    
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

def order_axis_by_folder(
        folder_sampling_dict: Dict[str, float],
        dataframe: pd.DataFrame
    ) -> List[str]:
    """
    Create ordered list of the multiplier folders for the x-axis.
    
    Args:
        folder_sampling_dict: A dictionary of grandparent folders and their sampling multipliers. 
        obj_props_df: A dataframe containing the properties of the objects in the image.
    
    Returns:
        order: A list of the grandparent folders in the order they should be displayed on the x-axis.
        
    """
    # Loop through all grandparent folders and add them to the order list if they are in the df
    order = [folder for folder in folder_sampling_dict.keys() if folder in dataframe['Grand_Parent_Folder'].unique()]

    return order
