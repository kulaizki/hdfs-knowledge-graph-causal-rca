import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats
import seaborn as sns

def perform_causal_analysis(data, graph, target_variable):
    """
    Perform causal inference analysis to identify root causes
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset containing all variables
    graph : networkx.DiGraph
        Knowledge graph with causal relationships
    target_variable : str
        Name of the variable to analyze for root causes
        
    Returns:
    --------
    dict : Results of analysis including potential root causes and metrics
    """
    # Step 1: Identify potential causes from the graph
    ancestors = nx.ancestors(graph, target_variable)
    direct_causes = list(graph.predecessors(target_variable))
    
    print(f"All ancestors of {target_variable}: {ancestors}")
    print(f"Direct causes of {target_variable}: {direct_causes}")
    
    # Step 2: Calculate correlation between potential causes and target
    correlations = {}
    for node in ancestors:
        if node in data.columns and target_variable in data.columns:
            corr = data[node].corr(data[target_variable])
            correlations[node] = corr
    
    # Step 3: Perform regression analysis for direct causes
    if len(direct_causes) > 0 and all(cause in data.columns for cause in direct_causes):
        X = data[direct_causes]
        y = data[target_variable]
        
        # Check for multicollinearity
        X_with_const = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                            for i in range(X_with_const.shape[1])]
        
        # Fit OLS model
        model = sm.OLS(y, X_with_const).fit()
        
        # Get statistics
        regression_results = {
            'model_summary': model.summary(),
            'coefficients': model.params,
            'p_values': model.pvalues,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'vif': vif_data
        }
    else:
        regression_results = None
    
    # Step 4: Perform intervention analysis (simulated)
    intervention_effects = {}
    for cause in direct_causes:
        if cause in data.columns:
            # Simulate intervention by shifting the cause variable
            intervention_data = data.copy()
            std_dev = intervention_data[cause].std()
            intervention_data[cause] = intervention_data[cause] + std_dev
            
            # Predict effect on target using a simple linear model
            X_orig = sm.add_constant(data[cause])
            X_interv = sm.add_constant(intervention_data[cause])
            simple_model = sm.OLS(data[target_variable], X_orig).fit()
            
            # Estimate effect of intervention
            original_pred = simple_model.predict(X_orig).mean()
            intervention_pred = simple_model.predict(X_interv).mean()
            
            # Calculate percentage change
            effect = (intervention_pred - original_pred) / original_pred * 100
            intervention_effects[cause] = effect
    
    # Step 5: Rank potential root causes
    root_causes = {}
    for node in direct_causes:
        if node in data.columns:
            # Calculate composite score based on correlation, regression coefficient and intervention effect
            corr_score = abs(correlations.get(node, 0))
            reg_score = 0
            if regression_results is not None:
                p_value = regression_results['p_values'].get(node, 1)
                coef = abs(regression_results['coefficients'].get(node, 0))
                reg_score = coef * (1 - p_value)
            
            interv_score = abs(intervention_effects.get(node, 0)) / 100
            
            # Combine scores (simple average)
            composite_score = (corr_score + reg_score + interv_score) / 3
            root_causes[node] = composite_score
    
    # Sort root causes by score
    sorted_causes = {k: v for k, v in sorted(root_causes.items(), 
                                             key=lambda item: item[1], 
                                             reverse=True)}
    
    return {
        'target_variable': target_variable,
        'correlations': correlations,
        'regression_results': regression_results,
        'intervention_effects': intervention_effects,
        'root_causes': sorted_causes
    }

def visualize_results(results, graph=None):
    """
    Visualize the results of root cause analysis
    
    Parameters:
    -----------
    results : dict
        Results from perform_causal_analysis
    graph : networkx.DiGraph, optional
        Knowledge graph for visualization
    """
    # 1. Visualize correlations
    plt.figure(figsize=(10, 6))
    corr_df = pd.DataFrame({
        'Variable': list(results['correlations'].keys()),
        'Correlation': list(results['correlations'].values())
    })
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    sns.barplot(x='Correlation', y='Variable', data=corr_df)
    plt.title(f'Correlation with {results["target_variable"]}')
    plt.tight_layout()
    plt.show()
    
    # 2. Visualize root causes
    plt.figure(figsize=(10, 6))
    causes_df = pd.DataFrame({
        'Variable': list(results['root_causes'].keys()),
        'Impact Score': list(results['root_causes'].values())
    })
    
    sns.barplot(x='Impact Score', y='Variable', data=causes_df)
    plt.title(f'Root Causes Impact on {results["target_variable"]}')
    plt.tight_layout()
    plt.show()
    
    # 3. Visualize intervention effects
    plt.figure(figsize=(10, 6))
    intervention_df = pd.DataFrame({
        'Variable': list(results['intervention_effects'].keys()),
        'Effect (% change)': list(results['intervention_effects'].values())
    })
    
    sns.barplot(x='Effect (% change)', y='Variable', data=intervention_df)
    plt.title(f'Estimated Effect of Interventions on {results["target_variable"]}')
    plt.tight_layout()
    plt.show()
    
    # 4. Visualize the causal graph if provided
    if graph is not None:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        
        # Draw regular nodes
        nx.draw_networkx_nodes(graph, pos, 
                               node_size=500, 
                               node_color='lightblue')
        
        # Highlight the target variable
        target_node = results['target_variable']
        nx.draw_networkx_nodes(graph, pos, 
                               nodelist=[target_node], 
                               node_size=700, 
                               node_color='red')
        
        # Highlight direct causes
        direct_causes = list(results['root_causes'].keys())
        nx.draw_networkx_nodes(graph, pos, 
                               nodelist=direct_causes, 
                               node_size=600, 
                               node_color='orange')
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos)
        
        plt.title('Causal Graph with Target and Root Causes')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def calculate_metrics(results, actual_causes=None):
    """
    Calculate metrics to evaluate the root cause analysis
    
    Parameters:
    -----------
    results : dict
        Results from perform_causal_analysis
    actual_causes : list, optional
        List of known actual root causes for comparison
        
    Returns:
    --------
    dict : Metrics evaluating the analysis
    """
    metrics = {}
    
    # Statistical significance of identified causes
    if results['regression_results'] is not None:
        p_values = results['regression_results']['p_values']
        significant_causes = [cause for cause, p_val in p_values.items() 
                              if p_val < 0.05 and cause != 'const']
        metrics['significant_causes'] = significant_causes
        metrics['significant_causes_count'] = len(significant_causes)
    
    # Model fit metrics
    if results['regression_results'] is not None:
        metrics['r_squared'] = results['regression_results']['r_squared']
        metrics['adj_r_squared'] = results['regression_results']['adj_r_squared']
    
    # Intervention impact metrics
    metrics['avg_intervention_effect'] = np.mean(list(results['intervention_effects'].values()))
    metrics['max_intervention_effect'] = max(results['intervention_effects'].values()) if results['intervention_effects'] else 0
    
    # If actual causes are provided, calculate precision and recall
    if actual_causes is not None:
        predicted_causes = list(results['root_causes'].keys())
        true_positives = [cause for cause in predicted_causes if cause in actual_causes]
        
        precision = len(true_positives) / len(predicted_causes) if predicted_causes else 0
        recall = len(true_positives) / len(actual_causes) if actual_causes else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
    
    return metrics

def generate_synthetic_data(n_samples=1000, seed=42):
    """
    Generate synthetic data for testing the root cause analysis
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (DataFrame, DiGraph, list)
        Data, causal graph, and actual root causes
    """
    np.random.seed(seed)
    
    # Root causes
    machine_temp = np.random.normal(80, 15, n_samples)
    material_quality = np.random.normal(75, 10, n_samples)
    operator_experience = np.random.normal(5, 2, n_samples)
    
    # Intermediate variables
    machine_wear = 0.3 * machine_temp + 0.1 * np.random.normal(0, 1, n_samples)
    processing_time = 0.4 * operator_experience + 0.2 * machine_temp + 0.1 * np.random.normal(0, 1, n_samples)
    
    # Target variable
    defect_rate = (0.5 * machine_wear + 
                  0.3 * material_quality + 
                  -0.4 * operator_experience + 
                  0.2 * processing_time +
                  0.1 * np.random.normal(0, 1, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'machine_temp': machine_temp,
        'material_quality': material_quality,
        'operator_experience': operator_experience,
        'machine_wear': machine_wear,
        'processing_time': processing_time,
        'defect_rate': defect_rate
    })
    
    # Define causal relationships
    causal_relationships = [
        ('machine_temp', 'machine_wear'),
        ('machine_temp', 'processing_time'),
        ('operator_experience', 'processing_time'),
        ('machine_wear', 'defect_rate'),
        ('material_quality', 'defect_rate'),
        ('operator_experience', 'defect_rate'),
        ('processing_time', 'defect_rate')
    ]
    
    # Create the causal graph
    G = nx.DiGraph()
    G.add_nodes_from(data.columns)
    G.add_edges_from(causal_relationships)
    
    # Define actual root causes
    actual_causes = ['machine_temp', 'material_quality', 'operator_experience']
    
    return data, G, actual_causes

def run_example():
    """
    Run a complete example of root cause analysis
    """
    # Generate synthetic data and causal graph
    data, G, actual_causes = generate_synthetic_data()
    
    # Perform the analysis
    results = perform_causal_analysis(data, G, 'defect_rate')
    
    # Calculate metrics
    metrics = calculate_metrics(results, actual_causes)
    
    # Print results
    print("\nIdentified Root Causes (sorted by impact):")
    for cause, score in results['root_causes'].items():
        print(f"{cause}: {score:.4f}")
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Visualize results
    visualize_results(results, G)
    
    return results, metrics

if __name__ == "__main__":
    # This will run the example when the script is executed directly
    run_example()