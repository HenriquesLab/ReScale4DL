import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def format_p_value(x):
    """Format p-values to four significant digits."""
    if x < 0.001:
        return "< 0.001"
    else:
        return f"{x:.4g}"  # .4g ensures four significant digits
    
def safe_log10_p_values(matrix):
    """Apply a safe logarithmic transformation to p-values, handling p=1 specifically."""
    # Replace non-positive values with a very small number just greater than 0
    small_value = np.nextafter(0, 1)
    adjusted_matrix = np.where(matrix > 0, matrix, small_value)

    logged_matrix = -np.log10(adjusted_matrix)
    logged_matrix[matrix == 1] = -np.log10(0.999)
    return logged_matrix
    
def create_pvalue_matrix(results_df):
    """
    Create symmetric p-value matrix from statistical analysis results.
    
    Args:
    results_df: DataFrame from perform_statistical_analysis()
    
    Returns:
    Symmetric pandas DataFrame with p-values (diagonal=1.0)
    """
    # Extract unique hierarchies and p-values
    all_hierarchies = set()
    pvalue_dict = {}
    
    for _, row in results_df.iterrows():
        h1, h2, pval = row['Hierarchy1'], row['Hierarchy2'], row['p-value']
        all_hierarchies.add(h1)
        all_hierarchies.add(h2)
        pvalue_dict[(h1, h2)] = pval
        pvalue_dict[(h2, h1)] = pval  # Make symmetric
    
    hierarchies = sorted(list(all_hierarchies))
    n = len(hierarchies)
    matrix = np.full((n, n), np.nan)
    
    # Fill matrix
    for i, h1 in enumerate(hierarchies):
        for j, h2 in enumerate(hierarchies):
            if i != j:
                key = (h1, h2)
                matrix[i, j] = pvalue_dict.get(key, np.nan)
    
    # Diagonal = 1.0 (self-comparison)
    np.fill_diagonal(matrix, 1.0)
    
    return pd.DataFrame(matrix, index=hierarchies, columns=hierarchies)


def choose_statistical_test(group1, group2):
    """
    Determine the appropriate statistical test based on data characteristics.
    
    Args:
    group1, group2: Arrays of data to compare
    
    Returns:
    String indicating the appropriate test
    """
    # Check for normality using the Shapiro-Wilk test
    _, p1 = stats.shapiro(group1)
    _, p2 = stats.shapiro(group2)
    
    # Check for equal variances using Levene's test
    _, p_var = stats.levene(group1, group2)
    
    if p1 > 0.05 and p2 > 0.05:  # Both groups are normally distributed
        if p_var > 0.05:  # Equal variances
            return "t-test"
        else:  # Unequal variances
            return "Welch's t-test"
    else:  # At least one group is not normally distributed
        return "Kolmogorov-Smirnov"

def run_statistical_test(group1, group2, test_type):
    """
    Perform the specified statistical test on two groups of data.
    
    Args:
    group1, group2: Arrays of data to compare
    test_type: String indicating which test to perform
    
    Returns:
    Tuple of (test statistic, p-value)
    """

    if test_type == "t-test":
        return stats.ttest_ind(group1, group2)
    elif test_type == "Welch's t-test":
        return stats.ttest_ind(group1, group2, equal_var=False)
    elif test_type == "Kolmogorov-Smirnov":
        return stats.ks_2samp(group1, group2)
    

def compute_statistical_analysis(ANALYSIS_DIR,
                                 DATASET,
                                 quantitative_column,
                                 categorical_columns,
                                 filename = "summary_stats.csv",
                                 test_type="Kolmogorov-Smirnov",
                                 choose_test="Manual"):
    """
    Perform pairwise statistical analysis on groups defined by multiple categorical variables.
    
    Args:
    ANALYSIS_DIR: Base directory where the data is stored
    DATASET: Name of the dataset to analyze
    quantitative_column: Name of the column containing the quantitative data to analyze
    categorical_columns: List of column names defining the hierarchy of groups
    test_type: "t-test", "Welch's t-test", or "Kolmogorov-Smirnov"
    choose_test: "Automatic" or "Manual"
    filename: footer of the CSV file containing the data (default: "summary_stats.csv")
    
    Returns:
    pandas DataFrame with results of statistical tests
    """

    PATH2FILE = os.path.join(ANALYSIS_DIR, f"{DATASET}/Results/{DATASET}_{filename}")
    df = pd.read_csv(PATH2FILE)
    if choose_test != "Automatic":
        print(f"The statistical test chosen is {test_type}")
    
    # Get all unique combinations
    combos_df = df[categorical_columns].drop_duplicates()
    combinations_list = list(combos_df.itertuples(index=False, name=None))
    
    results = []
    
    # Perform pairwise comparisons within each combination level
    for i, combo1 in enumerate(combinations_list):
        for combo2 in combinations_list[i+1:]:  # Avoid duplicate pairs
            
            # Create masks for both combinations
            mask1 = pd.Series([True] * len(df), index=df.index)
            mask2 = pd.Series([True] * len(df), index=df.index)
            
            for j, col in enumerate(categorical_columns):
                mask1 &= (df[col] == combo1[j])
                mask2 &= (df[col] == combo2[j])
            
            data1 = df.loc[mask1, quantitative_column].dropna()
            data2 = df.loc[mask2, quantitative_column].dropna()
            
            if len(data1) < 2 or len(data2) < 2:
                continue
                
            if choose_test == "Automatic":
                test_type = choose_statistical_test(data1, data2)
            
            statistic, p_value = run_statistical_test(data1, data2, test_type)
            
            # Format hierarchy paths for output
            path1 = ' | '.join([f"{categorical_columns[j]}: {combo1[j]}" for j in range(len(categorical_columns))])
            path2 = ' | '.join([f"{categorical_columns[j]}: {combo2[j]}" for j in range(len(categorical_columns))])
            
            if p_value < 0.005:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:    
                significance = '*'
            else:
                significance = ''

            results.append({
                'Hierarchy1': path1,
                'Hierarchy2': path2,
                'Test': test_type,
                'Statistic': statistic,
                'p-value': p_value,
                'Significance': significance
            })

    # Heatmap plot of the p-values (add to your notebook)    
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(ANALYSIS_DIR, f"{DATASET}/Results/{DATASET}_p-values_{filename}"), index=False)  # Save results to CSV for reference
    # Create a symmetric p-value matrix for heatmap visualization
    pval_matrix = create_pvalue_matrix(results)
    
    # Apply log transformation to p-values for better visualization of decimals, handling p=1 appropriately
    log_matrix = safe_log10_p_values(pval_matrix.fillna(1))
    # Define the normalization range 
    vmin = -np.log10(0.1)  # Set vmin to the log-transformed value of 0.1
    vmax = np.max(log_matrix[np.isfinite(log_matrix)])

    if vmin > vmax:
        vmin = vmax       
        
    formatted_annotations = pval_matrix.map(lambda x: format_p_value(x) if pd.notna(x) else "NaN")
    plt.figure(figsize=(5,5))
    plt.rcParams.update({'font.size': 5})
    sns.heatmap(log_matrix, cmap='Oranges', annot=formatted_annotations,
                xticklabels=pval_matrix.columns,
                yticklabels=pval_matrix.index, vmin=vmin, vmax=vmax, fmt='',
                cbar_kws={'label': '-log(p-value)'})
    plt.title('Pairwise p-value Matrix')
    plt.tight_layout()
    os.makedirs(os.path.join(ANALYSIS_DIR, f"{DATASET}/Plots"), exist_ok=True)
    plt.savefig(
        os.path.join(ANALYSIS_DIR, f"{DATASET}/Plots/{DATASET}_p-values_{filename}.png"),
        bbox_inches="tight",
        pad_inches=0.2,
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        os.path.join(ANALYSIS_DIR, f"{DATASET}/Plots/{DATASET}_p-values_{filename}.pdf"),
        bbox_inches="tight",
        pad_inches=0.2,
        dpi=300,
        transparent=True,
    )
    plt.show()
    
    return results

