# Imports




def main_function ():
    """
    INPUT:
    - main_dir - path to the folder containing the dataset folders
    - list of samplings - list of samplings to use in analysis (matching name of folders inside dataset folders)
    - sampling modifier dict - dictionary with sampling modifiers for each sampling
    - metrics - list of metrics to use in analysis

    Main function to run the code

    - os.listdir - iterate over found folders:

        -run per_dataset function

    RETURN:
    none
    """


def per_dataset ():
    """
    INPUT:
    - dataset_dir - path to the folder containing the dataset folders
    - sampling_dir_list - list of samplings to use in analysis (matching name of folders inside dataset folders)
    - sampling modifier dict - dictionary with sampling modifiers for each sampling
    - metrics - list of metrics to use in analysis

    
    Main function to run the code

    - check for results folder
        Create folder to store results if it doesn't exist, if it exists make new one

    - os.listdir - iterate over found sampling folders:
        - if sampling folder is not in sampling_dir_list, skip it
        - if sampling folder is in sampling_dir_list, run per_sampling function

    - concat all per_sampling outputs in one df

    - save table to csv file

    - generate summary table

    - save summary table to csv file
        
    RETURN:
    concated dataset results csv file
    dataset summary stats csv file
    
    """

def per_sampling ():
    """
    INPUT:
    - sampling_dir - path to the folder containing the GT/Prediction folders
    - sampling modifier dict - dictionary with sampling modifiers for each sampling
    - metrics - list of metrics to use in analysis


    Main function to run the code
    - check for GT and Prediction folders
        if not present, skip sampling folder
    - os.listdir - get list of GT and Prediction files
        - check if GT and Prediction files are paired
            - if paired, run per_image_pair function
        - if not, skip unpaired files

    - concat all per_image_pair outputs in one df

    -  calculate normalized values in reference to sampling modifier dict
        - calculate normalized values for each metric
        - add normalized values to dataframe

    RETURN:
    dataframe with results ffrom all images

    """

def per_image_pair ():
    """
    INPUT:
    - GT file - path to the GT file
    - Prediction file - path to the Prediction file
    - metrics - list of metrics to use in analysis

    Main function to run the code
    - load paired images
    - check if images are the same size
        - if not, run resize function (pad prediction)
        - if yes, continue
    - np.unique to get number of objects in GT and Prediction
    - iterate over each GT object
        - use bounding box per obj to check for IOU
            - compute_labels_matching_scores
            - find_matching_labels
            - store results in a list/df
        - get metrics from metrics list
            - run each metric function
            - add results to a list/df

    matric functions:
    - Pixel coverage
    - object diameter
    - object area
    - optional from region props table
    - user defined metrics from other sources?

    save results in a dataframe


    RETURN:
    dataframe with results for one image
    """