import os
import json
import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pickle
import re

class HDFSKnowledgeGraph:
    """
    A class to create, manipulate, and analyze a knowledge graph from HDFS_v2 trace data.
    This graph represents the relationships between HDFS components, operations, and anomalies.
    """
    
    def __init__(self, data_dir="data/processed"):
        """
        Initialize the knowledge graph.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing processed data files
        """
        self.data_dir = data_dir
        self.graph = nx.DiGraph()
        self.hdfs_v2_dir = os.path.join(data_dir, "HDFS_v2")
        self.templates = None
        self.structured_logs = None
        self.block_to_trace = {}  # Mapping from block IDs to trace IDs
        
    def load_data(self):
        """Load necessary data files from HDFS_v2"""
        print("Loading data files from HDFS_v2...")
        
        # Load template data (event definitions)
        templates_path = os.path.join(self.hdfs_v2_dir, "HDFS_2k.log_templates.csv")
        if os.path.exists(templates_path):
            self.templates = pd.read_csv(templates_path)
            print(f"Loaded {len(self.templates)} log templates from HDFS_2k.log_templates.csv")
        else:
            print(f"Warning: Template file not found at {templates_path}")
        
        # Load alternative templates for reference if needed
        alt_templates_path = os.path.join(self.hdfs_v2_dir, "HDFS_templates.csv")
        if os.path.exists(alt_templates_path):
            self.alt_templates = pd.read_csv(alt_templates_path)
            print(f"Loaded {len(self.alt_templates)} alternative templates from HDFS_templates.csv")
        
        # Load structured logs
        structured_logs_path = os.path.join(self.hdfs_v2_dir, "HDFS_2k.log_structured.csv")
        if os.path.exists(structured_logs_path):
            self.structured_logs = pd.read_csv(structured_logs_path)
            print(f"Loaded {len(self.structured_logs)} structured logs from HDFS_2k.log_structured.csv")
        else:
            print(f"Warning: Structured logs file not found at {structured_logs_path}")
    
    def extract_block_ids(self):
        """Extract block IDs from log content and create a mapping to traces"""
        if self.structured_logs is None:
            print("Error: No structured logs loaded")
            return
            
        print("Extracting block IDs from log content...")
        block_id_pattern = r'blk_[-\d]+'
        
        # Extract block IDs and associate them with log entries
        for idx, row in self.structured_logs.iterrows():
            content = row['Content']
            block_ids = re.findall(block_id_pattern, content)
            
            for block_id in block_ids:
                if block_id not in self.block_to_trace:
                    trace_id = f"Trace:{block_id}"
                    self.block_to_trace[block_id] = trace_id
        
        print(f"Extracted {len(self.block_to_trace)} unique block IDs")
        
    def build_graph(self):
        """Build the knowledge graph from loaded data"""
        if self.structured_logs is None or self.templates is None:
            print("Error: Missing required data (structured logs or templates)")
            return
            
        print("Building knowledge graph...")
        
        # Extract block IDs first
        self.extract_block_ids()
        
        # Add nodes for components and event types
        self._add_component_nodes()
        self._add_event_type_nodes()
        
        # Add trace nodes and connect them to events
        self._add_trace_nodes()
        
        # Connect events in sequences within traces
        self._connect_event_sequences()
        
        print(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def _add_component_nodes(self):
        """Add nodes for HDFS components (DataNode, NameNode, etc.)"""
        if self.structured_logs is None:
            return
            
        components = set()
        for _, row in self.structured_logs.iterrows():
            if 'Component' in row and pd.notna(row['Component']):
                components.add(row['Component'])
            
        for component in components:
            self.graph.add_node(f"Component:{component}", 
                               type="Component", 
                               name=component)
                
        print(f"Added {len(components)} component nodes")
        
    def _add_event_type_nodes(self):
        """Add nodes for event types from templates"""
        if self.templates is None:
            return
            
        for _, row in self.templates.iterrows():
            event_id = row['EventId']
            event_template = row['EventTemplate']
            
            self.graph.add_node(f"EventType:{event_id}", 
                               type="EventType",
                               event_id=event_id,
                               template=event_template)
                
        print(f"Added {len(self.templates)} event type nodes")
        
    def _add_trace_nodes(self):
        """Add nodes for traces and connect them to events"""
        if self.structured_logs is None:
            return
            
        # Add trace nodes for each unique block ID
        for block_id, trace_id in self.block_to_trace.items():
            if trace_id not in self.graph:
                self.graph.add_node(trace_id, 
                                   type="Trace",
                                   block_id=block_id)
        
        # Process each log entry
        for idx, row in self.structured_logs.iterrows():
            content = row['Content']
            block_id_pattern = r'blk_[-\d]+'
            block_ids = re.findall(block_id_pattern, content)
            
            if block_ids:
                # Create event instance node
                event_id = f"E{idx}"  # Use row index as event instance ID
                event_type_id = f"EventType:{row['EventId']}"
                
                self.graph.add_node(event_id,
                                   type="Event",
                                   timestamp=f"{row['Date']} {row['Time']}",
                                   content=content,
                                   event_type=row['EventId'],
                                   level=row['Level'],
                                   pid=row['Pid'])
                
                # Connect event to its type
                if event_type_id in self.graph:
                    self.graph.add_edge(event_id, event_type_id, 
                                       relation="instance_of")
                
                # Connect event to component if available
                if 'Component' in row and pd.notna(row['Component']):
                    component_id = f"Component:{row['Component']}"
                    if component_id in self.graph:
                        self.graph.add_edge(event_id, component_id,
                                           relation="executed_by")
                
                # Connect event to all related traces
                for block_id in block_ids:
                    trace_id = self.block_to_trace.get(block_id)
                    if trace_id and trace_id in self.graph:
                        self.graph.add_edge(event_id, trace_id, 
                                           relation="part_of")
        
        print(f"Added trace nodes and connected them to events")
        
    def _connect_event_sequences(self):
        """Connect events in temporal sequences within traces"""
        # Group events by trace
        trace_events = defaultdict(list)
        
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'Event':
                # Find which traces this event belongs to
                for _, trace, edge_data in self.graph.out_edges(node, data=True):
                    if edge_data.get('relation') == 'part_of' and self.graph.nodes[trace].get('type') == 'Trace':
                        trace_events[trace].append((node, data.get('timestamp', '')))
        
        # Sort events by timestamp within each trace and create sequence edges
        sequence_edges = 0
        for trace, events in trace_events.items():
            sorted_events = sorted(events, key=lambda x: x[1])
            for i in range(len(sorted_events) - 1):
                self.graph.add_edge(sorted_events[i][0], sorted_events[i+1][0],
                                   relation="followed_by",
                                   trace=trace)
                sequence_edges += 1
                
        print(f"Connected events in temporal sequences with {sequence_edges} edges")
    
    def identify_patterns(self):
        """Identify common event patterns in traces"""
        # Group events by trace
        trace_events = defaultdict(list)
        
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'Event':
                for _, trace, edge_data in self.graph.out_edges(node, data=True):
                    if edge_data.get('relation') == 'part_of' and self.graph.nodes[trace].get('type') == 'Trace':
                        event_type = data.get('event_type')
                        trace_events[trace].append(event_type)
        
        # Find event sequences
        sequences = []
        for trace, events in trace_events.items():
            if len(events) > 1:
                sequences.append(tuple(events))
        
        # Count sequence frequencies
        sequence_counts = Counter(sequences)
        common_sequences = sequence_counts.most_common(10)
        
        print("Common event sequences:")
        for seq, count in common_sequences:
            print(f"  Sequence {' -> '.join(seq)}: {count} occurrences")
        
        return common_sequences
        
    def visualize(self, max_nodes=100, output_path=None):
        """
        Visualize the knowledge graph (or a subset of it)
        
        Parameters:
        -----------
        max_nodes : int
            Maximum number of nodes to visualize
        output_path : str
            Path to save the visualization (if None, just display)
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Nothing to visualize.")
            return
            
        # Take a subgraph if the graph is too large
        if self.graph.number_of_nodes() > max_nodes:
            # Prioritize different node types for a more representative visualization
            component_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'Component']
            event_type_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'EventType']
            event_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'Event']
            trace_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'Trace']
            
            # Select a balanced subset
            selected_nodes = []
            selected_nodes.extend(component_nodes[:min(len(component_nodes), max_nodes // 10)])
            selected_nodes.extend(event_type_nodes[:min(len(event_type_nodes), max_nodes // 5)])
            remaining = max_nodes - len(selected_nodes)
            selected_nodes.extend(event_nodes[:min(len(event_nodes), remaining // 2)])
            selected_nodes.extend(trace_nodes[:min(len(trace_nodes), remaining // 2)])
            
            subgraph = self.graph.subgraph(selected_nodes)
        else:
            subgraph = self.graph
            
        plt.figure(figsize=(15, 10))
        
        # Create a position layout
        pos = nx.spring_layout(subgraph, seed=42)
        
        # Create node colors based on node type
        node_colors = []
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('type', '')
            if node_type == 'Component':
                node_colors.append('skyblue')
            elif node_type == 'EventType':
                node_colors.append('lightgreen')
            elif node_type == 'Event':
                node_colors.append('orange')
            elif node_type == 'Trace':
                node_colors.append('green')
            else:
                node_colors.append('gray')
                
        # Draw the graph
        nx.draw(subgraph, pos, with_labels=True, node_color=node_colors, 
                font_size=8, node_size=500, font_weight='bold', 
                edge_color='gray', width=0.5, alpha=0.8)
                
        plt.title(f"HDFS Knowledge Graph (showing {subgraph.number_of_nodes()} nodes)")
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
            
    def save_graph(self, output_path):
        """Save the graph to a file using pickle"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
        print(f"Graph saved to {output_path}")
        
    def load_graph(self, input_path):
        """Load a graph from a file using pickle"""
        with open(input_path, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"Graph loaded from {input_path} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def get_subgraph_for_trace(self, block_id):
        """Extract a subgraph for a specific trace"""
        trace_id = f"Trace:{block_id}"
        if trace_id not in self.graph:
            print(f"No trace found for block ID {block_id}")
            return None
            
        # Find all events in this trace
        trace_events = []
        for source, target, data in self.graph.edges(data=True):
            if target == trace_id and data.get('relation') == 'part_of':
                trace_events.append(source)
                
        # Get 1-hop neighborhood of all these events
        neighborhood = set(trace_events)
        for event in trace_events:
            neighborhood.update(self.graph.predecessors(event))
            neighborhood.update(self.graph.successors(event))
            
        # Create the subgraph
        return self.graph.subgraph(neighborhood)
    
    def get_graph_statistics(self):
        """Get basic statistics about the knowledge graph"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'node_types': Counter(data['type'] for _, data in self.graph.nodes(data=True) if 'type' in data),
            'edge_relations': Counter(data['relation'] for _, _, data in self.graph.edges(data=True) 
                                    if 'relation' in data),
            'num_components': nx.number_weakly_connected_components(self.graph),
            'num_traces': sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'Trace'),
            'num_event_types': sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'EventType'),
            'num_events': sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'Event')
        }
        
        return stats

# Example usage function
def build_hdfs_knowledge_graph(data_dir="data/processed", output_dir="reports"):
    """Build and save a knowledge graph from HDFS data"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the knowledge graph
    kg = HDFSKnowledgeGraph(data_dir)
    kg.load_data()
    kg.build_graph()
    
    # Identify common patterns
    kg.identify_patterns()
    
    # Print statistics
    stats = kg.get_graph_statistics()
    print("\nKnowledge Graph Statistics:")
    for key, value in stats.items():
        if isinstance(value, Counter):
            print(f"  {key}:")
            for k, v in value.most_common():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Visualize a subset of the graph
    visualization_path = os.path.join(output_dir, "hdfs_knowledge_graph.png")
    kg.visualize(max_nodes=500, output_path=visualization_path)
    
    # Save the graph
    graph_path = os.path.join(output_dir, "hdfs_knowledge_graph.pkl")
    kg.save_graph(graph_path)
    
    return kg

if __name__ == "__main__":
    build_hdfs_knowledge_graph()