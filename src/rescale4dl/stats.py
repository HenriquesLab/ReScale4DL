import pandas as pd
from scipy import stats

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
    

def compute_statistical_analysis(df, quantitative_column, categorical_columns,
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
            
            statistic, p_value = run_statistical_test(data1, data2, test_type)
            
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

