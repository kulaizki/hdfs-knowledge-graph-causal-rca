import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
import re

# Import the function that builds and returns the knowledge graph
# (Adjust the import path if needed; e.g. if graph.py is in the same folder)
from graph import build_hdfs_knowledge_graph


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "data/raw",
        "data/processed",
        "data/processed/HDFS_v1",
        "data/processed/HDFS_v3",
        "reports"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def load_structured_logs(file_path):
    """
    Load and preprocess structured log data
    
    Parameters:
    -----------
    file_path : str
        Path to the structured log CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed structured log data
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp columns to datetime if they exist
    if 'Date' in df and 'Time' in df:
        df['Timestamp'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'], 
            format='%y%m%d %H%M%S',
            errors='coerce'
        )
    
    # Ensure EventId is a string
    if 'EventId' in df:
        df['EventId'] = df['EventId'].astype(str)
    
    return df


def parse_block_id(content):
    """
    Extract block ID from log content
    
    Parameters:
    -----------
    content : str
        Log content string
    
    Returns:
    --------
    str or None
        Extracted block ID or None if not found
    """
    block_patterns = [
        r'blk_(-?\d+)',        # Standard block ID format
        r'block_id=(\w+)',     # Alternative format
        r'Block=(\w+)'         # Another alternative format
    ]
    
    for pattern in block_patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    
    return None


def aggregate_logs_by_block(logs_df):
    """
    Group logs by block ID to create traces
    
    Parameters:
    -----------
    logs_df : pandas.DataFrame
        DataFrame containing structured logs
    
    Returns:
    --------
    dict
        Dictionary mapping block IDs to lists of log entries
    """
    traces = defaultdict(list)
    
    for _, row in logs_df.iterrows():
        # Try to find block ID in specific columns
        block_id = None
        for col in ['BlockId', 'BlkId', 'Block_Id']:
            if col in row and pd.notna(row[col]):
                block_id = row[col]
                break
        
        # If not found in columns, try to extract from content
        if block_id is None and 'Content' in row:
            block_id = parse_block_id(row['Content'])
        
        if block_id:
            traces[block_id].append(row.to_dict())
    
    return traces


def extract_event_sequences(traces):
    """
    Extract event sequences from traces
    
    Parameters:
    -----------
    traces : dict
        Dictionary mapping block IDs to lists of log entries
    
    Returns:
    --------
    dict
        Dictionary mapping block IDs to sequences of event IDs
    """
    sequences = {}
    
    for block_id, events in traces.items():
        # Sort events by timestamp if available
        if events and 'Timestamp' in events[0]:
            sorted_events = sorted(events, key=lambda x: x.get('Timestamp', ''))
        else:
            # Assume events are already in chronological order
            sorted_events = events
        
        # Extract event IDs
        event_sequence = [event.get('EventId', 'E?') for event in sorted_events]
        sequences[block_id] = event_sequence
    
    return sequences


def identify_anomalous_patterns(normal_sequences, anomaly_sequences, min_support=0.05, max_items=5):
    """
    Identify patterns that are more common in anomalous traces
    
    Parameters:
    -----------
    normal_sequences : dict
        Dictionary of normal event sequences
    anomaly_sequences : dict
        Dictionary of anomalous event sequences
    min_support : float
        Minimum support threshold for patterns
    max_items : int
        Maximum number of events in a pattern
    
    Returns:
    --------
    list
        List of (pattern, score) tuples, where score indicates how much more
        common the pattern is in anomalous traces
    """
    try:
        from mlxtend.frequent_patterns import apriori
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        print("Warning: mlxtend package not installed. Install with: pip install mlxtend")
        return []
    
    # Convert sequences to list of lists for apriori
    normal_transactions = list(normal_sequences.values())
    anomaly_transactions = list(anomaly_sequences.values())
    
    te = TransactionEncoder()
    te_ary = te.fit_transform(normal_transactions + anomaly_transactions)
    all_events = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(
        all_events, 
        min_support=min_support, 
        use_colnames=True, 
        max_len=max_items
    )
    
    if frequent_itemsets.empty:
        return []
    
    # Calculate support in normal and anomalous traces separately
    patterns = []
    for _, row in frequent_itemsets.iterrows():
        itemset = list(row['itemsets'])
        
        # Count occurrences in normal traces
        normal_count = sum(
            1 for seq in normal_transactions 
            if all(item in seq for item in itemset)
        )
        normal_support = (
            normal_count / len(normal_transactions) 
            if normal_transactions else 0
        )
        
        # Count occurrences in anomalous traces
        anomaly_count = sum(
            1 for seq in anomaly_transactions 
            if all(item in seq for item in itemset)
        )
        anomaly_support = (
            anomaly_count / len(anomaly_transactions) 
            if anomaly_transactions else 0
        )
        
        # Calculate relative risk
        if normal_support > 0:
            relative_risk = anomaly_support / normal_support
        else:
            relative_risk = float('inf') if anomaly_support > 0 else 0
            
        # Keep patterns that are more common in anomalies
        if relative_risk > 1:
            patterns.append((itemset, relative_risk))
    
    # Sort by relative risk (highest first)
    patterns.sort(key=lambda x: x[1], reverse=True)
    return patterns


def visualize_event_distribution(logs_df, output_path=None):
    """
    Visualize the distribution of event types
    
    Parameters:
    -----------
    logs_df : pandas.DataFrame
        DataFrame containing structured logs
    output_path : str
        Path to save the visualization (optional)
    """
    if 'EventId' not in logs_df.columns:
        print("Error: EventId column not found in logs")
        return
        
    plt.figure(figsize=(12, 6))
    
    # Count occurrences of each event type
    event_counts = logs_df['EventId'].value_counts()
    
    # Plot top 20 events
    top_events = event_counts.head(20)
    sns.barplot(x=top_events.index, y=top_events.values)
    plt.title('Top 20 Event Types by Frequency')
    plt.xlabel('Event ID')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Event distribution visualization saved to {output_path}")
    else:
        plt.show()


def visualize_component_distribution(logs_df, output_path=None):
    """
    Visualize the distribution of components
    
    Parameters:
    -----------
    logs_df : pandas.DataFrame
        DataFrame containing structured logs
    output_path : str
        Path to save the visualization (optional)
    """
    if 'Component' not in logs_df.columns:
        print("Error: Component column not found in logs")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Count occurrences of each component
    component_counts = logs_df['Component'].value_counts()
    
    # Plot components
    sns.barplot(x=component_counts.index, y=component_counts.values)
    plt.title('Component Distribution')
    plt.xlabel('Component')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Component distribution visualization saved to {output_path}")
    else:
        plt.show()


def compare_normal_anomaly_events(normal_df, anomaly_df, output_path=None):
    """
    Compare event distributions between normal and anomalous traces
    
    Parameters:
    -----------
    normal_df : pandas.DataFrame
        DataFrame containing normal logs
    anomaly_df : pandas.DataFrame
        DataFrame containing anomalous logs
    output_path : str
        Path to save the visualization (optional)
    """
    if 'EventId' not in normal_df.columns or 'EventId' not in anomaly_df.columns:
        print("Error: EventId column not found in logs")
        return
        
    plt.figure(figsize=(14, 7))
    
    # Count event occurrences in both datasets
    normal_counts = normal_df['EventId'].value_counts().to_dict()
    anomaly_counts = anomaly_df['EventId'].value_counts().to_dict()
    
    # Normalize counts to percentages
    normal_total = sum(normal_counts.values())
    anomaly_total = sum(anomaly_counts.values())
    
    for event_id in set(list(normal_counts.keys()) + list(anomaly_counts.keys())):
        normal_counts[event_id] = normal_counts.get(event_id, 0) / normal_total * 100
        anomaly_counts[event_id] = anomaly_counts.get(event_id, 0) / anomaly_total * 100
    
    # Prepare data for plotting
    events = sorted(set(list(normal_counts.keys()) + list(anomaly_counts.keys())))
    normal_pct = [normal_counts.get(e, 0) for e in events]
    anomaly_pct = [anomaly_counts.get(e, 0) for e in events]
    
    # Only include top events by difference
    diff = [abs(a - n) for n, a in zip(normal_pct, anomaly_pct)]
    indices = np.argsort(diff)[-20:]  # Top 20 by difference
    
    top_events = [events[i] for i in indices]
    top_normal = [normal_pct[i] for i in indices]
    top_anomaly = [anomaly_pct[i] for i in indices]
    
    # Plot comparison
    x = np.arange(len(top_events))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, top_normal, width, label='Normal')
    ax.bar(x + width/2, top_anomaly, width, label='Anomaly')
    
    ax.set_title('Event Distribution: Normal vs. Anomalous Traces (% of total)')
    ax.set_xlabel('Event ID')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(top_events, rotation=45)
    ax.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Normal vs. anomaly comparison saved to {output_path}")
    else:
        plt.show()


def get_sequence_length_stats(sequences):
    """
    Get statistics about sequence lengths
    
    Parameters:
    -----------
    sequences : dict
        Dictionary of event sequences
    
    Returns:
    --------
    dict
        Dictionary with sequence length statistics
    """
    lengths = [len(seq) for seq in sequences.values()]
    
    stats = {
        'min': min(lengths) if lengths else 0,
        'max': max(lengths) if lengths else 0,
        'mean': np.mean(lengths) if lengths else 0,
        'median': np.median(lengths) if lengths else 0,
        'std': np.std(lengths) if lengths else 0,
        'count': len(lengths)
    }
    
    return stats


def save_summary_report(results, output_path):
    """
    Save a summary report of analysis results
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results
    output_path : str
        Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("# HDFS Knowledge Graph Analysis Summary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Statistics\n\n")
        if 'dataset_stats' in results:
            stats = results['dataset_stats']
            f.write(f"- Total logs: {stats.get('total_logs', 'N/A')}\n")
            f.write(f"- Normal traces: {stats.get('normal_traces', 'N/A')}\n")
            f.write(f"- Anomalous traces: {stats.get('anomaly_traces', 'N/A')}\n")
            f.write(f"- Unique event types: {stats.get('unique_events', 'N/A')}\n")
            f.write(f"- Components: {stats.get('components', 'N/A')}\n\n")
        
        f.write("## Sequence Length Statistics\n\n")
        if 'sequence_stats' in results:
            normal = results['sequence_stats'].get('normal', {})
            anomaly = results['sequence_stats'].get('anomaly', {})
            
            f.write("### Normal Traces\n\n")
            f.write(f"- Count: {normal.get('count', 'N/A')}\n")
            f.write(f"- Min length: {normal.get('min', 'N/A')}\n")
            f.write(f"- Max length: {normal.get('max', 'N/A')}\n")
            f.write(f"- Mean length: {normal.get('mean', 'N/A'):.2f}\n")
            f.write(f"- Median length: {normal.get('median', 'N/A'):.2f}\n")
            f.write(f"- Standard deviation: {normal.get('std', 'N/A'):.2f}\n\n")
            
            f.write("### Anomalous Traces\n\n")
            f.write(f"- Count: {anomaly.get('count', 'N/A')}\n")
            f.write(f"- Min length: {anomaly.get('min', 'N/A')}\n")
            f.write(f"- Max length: {anomaly.get('max', 'N/A')}\n")
            f.write(f"- Mean length: {anomaly.get('mean', 'N/A'):.2f}\n")
            f.write(f"- Median length: {anomaly.get('median', 'N/A'):.2f}\n")
            f.write(f"- Standard deviation: {anomaly.get('std', 'N/A'):.2f}\n\n")
        
        f.write("## Graph Statistics\n\n")
        if 'graph_stats' in results:
            stats = results['graph_stats']
            f.write(f"- Total nodes: {stats.get('num_nodes', 'N/A')}\n")
            f.write(f"- Total edges: {stats.get('num_edges', 'N/A')}\n")
            node_types = stats.get('node_types', {})
            if node_types:
                f.write("- Node types:\n")
                for node_type, count in node_types.items():
                    f.write(f"  - {node_type}: {count}\n")
            edge_relations = stats.get('edge_relations', {})
            if edge_relations:
                f.write("- Edge relations:\n")
                for relation, count in edge_relations.items():
                    f.write(f"  - {relation}: {count}\n")
            f.write(f"- Connected components: {stats.get('num_components', 'N/A')}\n\n")
        
        f.write("## Anomalous Patterns\n\n")
        if 'anomalous_patterns' in results:
            patterns = results['anomalous_patterns']
            if patterns:
                f.write("| Pattern | Relative Risk |\n")
                f.write("|---------|---------------|\n")
                for pattern, risk in patterns:
                    pattern_str = ', '.join(pattern)
                    f.write(f"| {pattern_str} | {risk:.2f} |\n")
            else:
                f.write("No significant anomalous patterns identified.\n\n")
        
        f.write("\n## Visualizations\n\n")
        if 'visualizations' in results:
            for name, path in results['visualizations'].items():
                f.write(f"- [{name}]({path})\n")
    
    print(f"Summary report saved to {output_path}")


def load_json_file(file_path):
    """
    Load and parse a JSON file
    
    Parameters:
    -----------
    file_path : str
        Path to the JSON file
    
    Returns:
    --------
    dict
        Parsed JSON data
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return {}
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return {}


def split_logs_by_anomaly(logs_df, anomaly_labels):
    """
    Split logs into normal and anomalous based on anomaly labels
    
    Parameters:
    -----------
    logs_df : pandas.DataFrame
        DataFrame containing structured logs
    anomaly_labels : pandas.DataFrame
        DataFrame containing anomaly labels
    
    Returns:
    --------
    tuple
        (normal_logs, anomaly_logs) as pandas.DataFrames
    """
    if 'BlockId' not in logs_df.columns:
        # Try to extract block IDs if not present
        if 'Content' in logs_df.columns:
            logs_df['BlockId'] = logs_df['Content'].apply(parse_block_id)
    
    # Create a set of anomalous block IDs
    anomaly_blocks = set()
    if 'BlockId' in anomaly_labels.columns and 'Label' in anomaly_labels.columns:
        anomaly_blocks = set(
            anomaly_labels[anomaly_labels['Label'] == 'Anomaly']['BlockId'].astype(str)
        )
    
    # Split logs
    normal_logs = logs_df[~logs_df['BlockId'].astype(str).isin(anomaly_blocks)]
    anomaly_logs = logs_df[logs_df['BlockId'].astype(str).isin(anomaly_blocks)]
    
    return normal_logs, anomaly_logs


def analyze_hdfs_data(data_dir="data/processed", output_dir="reports"):
    """
    Analyze HDFS data and generate visualizations and reports.
    Now also builds the knowledge graph via graph.py and includes
    those stats in the summary.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing processed data
    output_dir : str
        Directory to save reports and visualizations
    
    Returns:
    --------
    dict
        Analysis results
    """
    # Ensure directories exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {
        'dataset_stats': {},
        'sequence_stats': {},
        'visualizations': {}
    }
    
    # Load data
    logs_path = os.path.join(data_dir, "HDFS_v3", "HDFS_v3.log_structured.csv")
    anomaly_path = os.path.join(data_dir, "HDFS_v1", "anomaly_label.csv")
    
    logs_df = load_structured_logs(logs_path)
    anomaly_labels = pd.read_csv(anomaly_path) if os.path.exists(anomaly_path) else None
    
    if logs_df is None:
        print("Error: Could not load structured logs")
        return results
    
    # Basic dataset statistics
    results['dataset_stats']['total_logs'] = len(logs_df)
    results['dataset_stats']['unique_events'] = (
        logs_df['EventId'].nunique() if 'EventId' in logs_df else 'N/A'
    )
    results['dataset_stats']['components'] = (
        logs_df['Component'].nunique() if 'Component' in logs_df else 'N/A'
    )
    
    # Split logs by anomaly
    if anomaly_labels is not None:
        normal_logs, anomaly_logs = split_logs_by_anomaly(logs_df, anomaly_labels)
    else:
        normal_logs, anomaly_logs = logs_df, pd.DataFrame()
        
    results['dataset_stats']['normal_traces'] = (
        normal_logs['BlockId'].nunique() if 'BlockId' in normal_logs else 'N/A'
    )
    results['dataset_stats']['anomaly_traces'] = (
        anomaly_logs['BlockId'].nunique() if 'BlockId' in anomaly_logs else 'N/A'
    )
    
    # Visualize event distribution
    event_viz_path = os.path.join(output_dir, "event_distribution.png")
    visualize_event_distribution(logs_df, event_viz_path)
    results['visualizations']['Event Type Distribution'] = event_viz_path
    
    # Visualize component distribution if available
    if 'Component' in logs_df:
        component_viz_path = os.path.join(output_dir, "component_distribution.png")
        visualize_component_distribution(logs_df, component_viz_path)
        results['visualizations']['Component Distribution'] = component_viz_path
    
    # Compare normal and anomalous traces
    if not anomaly_logs.empty:
        compare_viz_path = os.path.join(output_dir, "normal_vs_anomaly.png")
        compare_normal_anomaly_events(normal_logs, anomaly_logs, compare_viz_path)
        results['visualizations']['Normal vs. Anomalous Events'] = compare_viz_path
    
    # Analyze sequences
    normal_traces = aggregate_logs_by_block(normal_logs)
    anomaly_traces = aggregate_logs_by_block(anomaly_logs)
    
    normal_sequences = extract_event_sequences(normal_traces)
    anomaly_sequences = extract_event_sequences(anomaly_traces)
    
    # Sequence statistics
    results['sequence_stats']['normal'] = get_sequence_length_stats(normal_sequences)
    results['sequence_stats']['anomaly'] = get_sequence_length_stats(anomaly_sequences)
    
    # Identify anomalous patterns
    if normal_sequences and anomaly_sequences:
        anomalous_patterns = identify_anomalous_patterns(normal_sequences, anomaly_sequences)
        results['anomalous_patterns'] = anomalous_patterns
    
    # Build knowledge graph from graph.py and gather its stats
    kg = build_hdfs_knowledge_graph(data_dir=data_dir, output_dir=output_dir)
    if kg:
        graph_stats = kg.get_graph_statistics()
        results['graph_stats'] = graph_stats
    
    # Save summary report
    report_path = os.path.join(output_dir, "hdfs_analysis_summary.md")
    save_summary_report(results, report_path)
    
    return results