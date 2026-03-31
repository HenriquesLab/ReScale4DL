import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
    


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

def perform_statistical_test(group1, group2, test_type):
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
    

def perform_statistical_analysis(df, quantitative_column, categorical_columns,
                               test_type="Kolmogorov-Smirnov", choose_test="Manual"):
    """
    Perform pairwise statistical analysis on groups defined by multiple categorical variables.
    
    Args:
    df: pandas DataFrame containing the data
    quantitative_column: Name of the column containing the quantitative data to analyze
    categorical_columns: List of column names defining the hierarchy of groups
    test_type: "t-test", "Welch's t-test", or "Kolmogorov-Smirnov"
    choose_test: "Automatic" or "Manual"
    
    Returns:
    pandas DataFrame with results of statistical tests
    """
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
            
            statistic, p_value = perform_statistical_test(data1, data2, test_type)
            
            # Format hierarchy paths for output
            path1 = ' | '.join([f"{categorical_columns[j]}: {combo1[j]}" for j in range(len(categorical_columns))])
            path2 = ' | '.join([f"{categorical_columns[j]}: {combo2[j]}" for j in range(len(categorical_columns))])
            
            results.append({
                'Hierarchy1': path1,
                'Hierarchy2': path2,
                'Test': test_type,
                'Statistic': statistic,
                'p-value': p_value
            })
    
    return pd.DataFrame(results)

def plot_data_distributions(df, quantitative_column, categorical_columns):
    """
    Plot histograms and Q-Q plots for subgroups defined by multiple categorical variables.
    
    Args:
    df: pandas DataFrame containing the data
    quantitative_column: Name of the column containing the quantitative data to plot
    categorical_columns: List of column names defining the hierarchy of subgroups
    """

    plt.rcParams.update({'font.size': 8})
    
    # Get all unique combinations across categorical columns
    combos_df = df[categorical_columns].drop_duplicates()
    combinations = list(combos_df.itertuples(index=False, name=None))
    
    n_groups = len(combinations)
    if n_groups == 0:
        print("No data found.")
        return
    
    n_cols = 2  # Histogram + Q-Q
    n_rows = n_groups
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, combo in enumerate(combinations):
        # Create mask for this combination
        mask = pd.Series([True] * len(df), index=df.index)
        for j, col in enumerate(categorical_columns):
            mask &= (df[col] == combo[j])
        
        data = df.loc[mask, quantitative_column].dropna()
        if len(data) == 0:
            axes[i, 0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i, 0].transAxes)
            axes[i, 1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i, 1].transAxes)
            continue
        
        # Histogram
        sns.histplot(data, kde=True, ax=axes[i, 0])
        title_parts = [f"{categorical_columns[j]}: {combo[j]}" for j in range(len(categorical_columns))]
        axes[i, 0].set_title(' | '.join(title_parts) + ' - Histogram')
        
        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(' | '.join(title_parts) + ' - Q-Q Plot')
    
    fig.suptitle(f'Data Distributions by {quantitative_column}', fontsize=12)
    plt.tight_layout()
    plt.show()
