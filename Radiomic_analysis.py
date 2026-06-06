#!/usr/bin/env python
# coding: utf-8

"""
BRAIN NETWORK EVOLUTION ACROSS ALL STAGES
Shows Normal, EMCI, MCI, LMCI, AD networks in a single figure
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('./brain_network_evolution', exist_ok=True)


class BrainNetworkEvolution:
    """Generate brain network evolution figures for all stages"""
    
    def __init__(self):
        self.stage_names = ['Normal', 'EMCI', 'MCI', 'LMCI', 'AD']
        self.stage_colors = ['#2E8B57', '#4169E1', '#FF8C00', '#DC143C', '#8B0000']
        self.stage_markers = ['o', 's', '^', 'D', 'v']
        
    def load_dataset(self, file_path):
        """Load radiomic dataset from Excel"""
        print(f"📂 Loading: {os.path.basename(file_path)}")
        
        df = pd.read_excel(file_path, sheet_name=0, header=None)
        
        # Skip header if present
        if df.iloc[0, 0] == 'A' or pd.isna(df.iloc[0, 0]):
            df = df.iloc[1:].reset_index(drop=True)
        
        # Extract features (57 columns)
        X = df.iloc[:, :57].values.astype(float)
        
        # Generate labels for 5 stages
        n_samples = len(df)
        rows_per_class = n_samples // 5
        labels = []
        
        for i in range(n_samples):
            class_idx = min(i // rows_per_class, 4)
            labels.append(self.stage_names[class_idx])
        
        y = np.array(labels)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"   ✅ {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   📊 Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return X, y_encoded
    
    def create_stage_networks(self, X, y, threshold=0.6):
        """Create networks for each stage"""
        networks = {}
        
        for stage_idx, stage_name in enumerate(self.stage_names):
            stage_data = X[y == stage_idx]
            
            if len(stage_data) < 2:
                print(f"   ⚠️  Not enough samples for {stage_name}")
                continue
            
            # Compute correlation matrix
            corr_matrix = np.corrcoef(stage_data.T)
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes (57 regions)
            for i in range(min(57, X.shape[1])):
                G.add_node(i)
            
            # Add edges based on correlation threshold
            edge_weights = []
            for i in range(min(57, X.shape[1])):
                for j in range(i+1, min(57, X.shape[1])):
                    if abs(corr_matrix[i, j]) > threshold:
                        G.add_edge(i, j, weight=abs(corr_matrix[i, j]))
                        edge_weights.append(abs(corr_matrix[i, j]))
            
            # Calculate metrics
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
            avg_corr = np.mean(edge_weights) if edge_weights else 0
            
            networks[stage_name] = {
                'graph': G,
                'n_edges': n_edges,
                'density': density,
                'avg_corr': avg_corr,
                'n_samples': len(stage_data),
                'stage_idx': stage_idx
            }
            
            print(f"   📍 {stage_name}: {n_edges} edges, density: {density:.3f}, avg_corr: {avg_corr:.3f}")
        
        return networks
    
    def plot_evolution_figure(self, networks, region_name, save_path):
        """
        Create the evolution figure with all 5 stages
        Format: 1 row × 5 columns
        """
        
        # Create figure with 1 row, 5 columns
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        # Use consistent layout across all stages (based on Normal stage)
        if 'Normal' in networks:
            ref_graph = networks['Normal']['graph']
            pos = nx.spring_layout(ref_graph, k=3, iterations=100, seed=42)
        else:
            # If no Normal, use first available
            first_stage = list(networks.keys())[0]
            ref_graph = networks[first_stage]['graph']
            pos = nx.spring_layout(ref_graph, k=3, iterations=100, seed=42)
        
        # Plot each stage
        for idx, stage_name in enumerate(self.stage_names):
            if stage_name not in networks:
                # Show empty plot with message
                axes[idx].text(0.5, 0.5, f'{stage_name}\nNo Data', 
                              ha='center', va='center', fontsize=14)
                axes[idx].axis('off')
                continue
            
            ax = axes[idx]
            G = networks[stage_name]['graph']
            
            # Draw edges with thickness based on correlation weight
            edges = G.edges(data=True)
            if edges:
                weights = [d['weight'] * 3 for (_, _, d) in edges]
                nx.draw_networkx_edges(G, pos, ax=ax, width=weights,
                                      alpha=0.5, edge_color='gray')
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=120,
                                  node_color=self.stage_colors[idx],
                                  edgecolors='black', linewidths=1,
                                  alpha=0.9)
            
            # Add labels for top 5 important nodes (by degree)
            if G.number_of_nodes() > 0:
                degrees = dict(G.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:5]
                labels = {node: f"R{node+1}" for node in top_nodes}
                nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10,
                                       font_weight='bold')
            
            # Set title with stage name and metrics
            ax.set_title(f'{stage_name}\n'
                        f'Edges: {networks[stage_name]["n_edges"]} | '
                        f'Density: {networks[stage_name]["density"]:.3f}\n'
                        f'Avg Corr: {networks[stage_name]["avg_corr"]:.3f}',
                        fontsize=12, fontweight='bold', 
                        color=self.stage_colors[idx], pad=15)
            ax.axis('off')
        
        # Main title
        plt.suptitle(f'{region_name}: Brain Network Evolution Across All Stages',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {os.path.basename(save_path)}")
    
    def plot_evolution_with_stats(self, networks, region_name, save_path):
        """
        Enhanced version with statistics panel below the networks
        """
        
        # Create figure with GridSpec: 2 rows, 5 columns
        fig = plt.figure(figsize=(25, 8))
        gs = GridSpec(2, 5, height_ratios=[4, 1], hspace=0.3, wspace=0.2)
        
        # Use consistent layout
        if 'Normal' in networks:
            ref_graph = networks['Normal']['graph']
            pos = nx.spring_layout(ref_graph, k=3, iterations=100, seed=42)
        else:
            first_stage = list(networks.keys())[0]
            ref_graph = networks[first_stage]['graph']
            pos = nx.spring_layout(ref_graph, k=3, iterations=100, seed=42)
        
        # Store metrics for statistics
        metrics_data = []
        
        # Plot networks (top row)
        for idx, stage_name in enumerate(self.stage_names):
            if stage_name not in networks:
                ax = fig.add_subplot(gs[0, idx])
                ax.text(0.5, 0.5, f'{stage_name}\nNo Data', 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                continue
            
            ax = fig.add_subplot(gs[0, idx])
            G = networks[stage_name]['graph']
            
            # Draw edges
            edges = G.edges(data=True)
            if edges:
                weights = [d['weight'] * 3 for (_, _, d) in edges]
                nx.draw_networkx_edges(G, pos, ax=ax, width=weights,
                                      alpha=0.4, edge_color='gray')
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=100,
                                  node_color=self.stage_colors[idx],
                                  edgecolors='black', linewidths=1,
                                  alpha=0.9)
            
            # Add labels for top nodes
            if G.number_of_nodes() > 0:
                degrees = dict(G.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:5]
                labels = {node: f"R{node+1}" for node in top_nodes}
                nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9,
                                       font_weight='bold')
            
            ax.set_title(f'{stage_name}', fontsize=14, fontweight='bold',
                        color=self.stage_colors[idx], pad=10)
            ax.axis('off')
            
            # Collect metrics
            metrics_data.append({
                'Stage': stage_name,
                'Edges': networks[stage_name]['n_edges'],
                'Density': networks[stage_name]['density'],
                'Avg Corr': networks[stage_name]['avg_corr']
            })
        
        # Statistics table (bottom row)
        ax_stats = fig.add_subplot(gs[1, :])
        ax_stats.axis('tight')
        ax_stats.axis('off')
        
        # Create table
        df_metrics = pd.DataFrame(metrics_data)
        
        # Calculate changes relative to Normal
        normal_edges = df_metrics[df_metrics['Stage'] == 'Normal']['Edges'].values[0] if 'Normal' in df_metrics['Stage'].values else None
        normal_density = df_metrics[df_metrics['Stage'] == 'Normal']['Density'].values[0] if 'Normal' in df_metrics['Stage'].values else None
        
        if normal_edges is not None:
            df_metrics['Δ Edges'] = df_metrics['Edges'] - normal_edges
            df_metrics['Δ Edges %'] = ((df_metrics['Edges'] - normal_edges) / normal_edges * 100).round(1)
            df_metrics['Δ Density %'] = ((df_metrics['Density'] - normal_density) / normal_density * 100).round(1)
        else:
            df_metrics['Δ Edges'] = '-'
            df_metrics['Δ Edges %'] = '-'
            df_metrics['Δ Density %'] = '-'
        
        # Format for display
        display_df = df_metrics[['Stage', 'Edges', 'Density', 'Avg Corr', 'Δ Edges', 'Δ Edges %', 'Δ Density %']].copy()
        display_df['Density'] = display_df['Density'].apply(lambda x: f'{x:.3f}')
        display_df['Avg Corr'] = display_df['Avg Corr'].apply(lambda x: f'{x:.3f}')
        
        # Create table
        table = ax_stats.table(cellText=display_df.values,
                              colLabels=display_df.columns,
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.1, 0.08, 0.08, 0.08, 0.1, 0.1, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        
        # Color code the stage rows
        for i, stage in enumerate(display_df['Stage']):
            cell = table[(i+1, 0)]
            if stage == 'Normal':
                cell.set_facecolor('#2E8B5733')
            elif stage == 'EMCI':
                cell.set_facecolor('#4169E133')
            elif stage == 'MCI':
                cell.set_facecolor('#FF8C0033')
            elif stage == 'LMCI':
                cell.set_facecolor('#DC143C33')
            elif stage == 'AD':
                cell.set_facecolor('#8B000033')
        
        # Main title
        plt.suptitle(f'{region_name}: Brain Network Evolution Across All Stages\n'
                    f'Nodes: 57 | Threshold: 0.6',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved (with stats): {os.path.basename(save_path)}")


# =================================================================
# PROCESS ALL REGIONS
# =================================================================

def analyze_all_regions():
    """Generate evolution figures for all brain regions"""
    
    print("="*80)
    print("🧠 BRAIN NETWORK EVOLUTION - ALL STAGES IN ONE FIGURE")
    print("📊 Normal → EMCI → MCI → LMCI → AD")
    print("="*80)
    
    # Define all brain regions
    brain_regions = [
        {'file': 'Radiomic feature Normal,EMCI,MCI,LMCI,AD WM.xlsx', 'name': 'White Matter (WM)'},
        {'file': 'RadiomicNor,EMCI,MCI,LMCI,AD BS.xlsx', 'name': 'Brainstem (BS)'},
        {'file': 'RadiomicNor,emci,mci,lmci,ad CC.xlsx', 'name': 'Corpus Callosum (CC)'},
        {'file': 'RadiomicNor,emci,mci,lmci,ad CSF.xlsx', 'name': 'Cerebrospinal Fluid (CSF)'},
        {'file': 'RadiomicNor,emci,mci,lmci,ad GM.xlsx', 'name': 'Gray Matter (GM)'},
        {'file': 'RadiomicNor,emci,mci,lmci,ad HIPPO.xlsx', 'name': 'Hippocampus (HIPPO)'},
        {'file': 'RadiomicNor,EMCI,MCI,LMCI,AD MB.xlsx', 'name': 'Midbrain (MB)'},
        {'file': 'RadiomicNor,emci,mci,lmci,ad VENT.xlsx', 'name': 'Ventricles (VENT)'}
    ]
    
    analyzer = BrainNetworkEvolution()
    
    # Process each region
    for region in brain_regions:
        if not os.path.exists(region['file']):
            print(f"\n⚠️  File not found: {region['file']}")
            continue
        
        print(f"\n{'='*60}")
        print(f"🔍 Processing: {region['name']}")
        print(f"{'='*60}")
        
        # Load data
        X, y = analyzer.load_dataset(region['file'])
        
        # Remove NaN rows
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.any():
            X = X[~nan_mask]
            y = y[~nan_mask]
            print(f"   🧹 Removed {nan_mask.sum()} samples with NaN")
        
        # Create networks
        networks = analyzer.create_stage_networks(X, y, threshold=0.6)
        
        if not networks:
            print(f"   ❌ No networks created")
            continue
        
        # Create region-specific directory
        region_short = region['name'].split()[0]
        region_dir = f'./brain_network_evolution/{region_short}'
        os.makedirs(region_dir, exist_ok=True)
        
        # Generate figures
        print(f"\n   🎨 Generating evolution figures...")
        
        # 1. Simple evolution figure (like your example)
        analyzer.plot_evolution_figure(
            networks,
            region['name'],
            f'{region_dir}/evolution_figure.png'
        )
        
        # 2. Enhanced version with statistics table
        analyzer.plot_evolution_with_stats(
            networks,
            region['name'],
            f'{region_dir}/evolution_with_stats.png'
        )
        
        # Save metrics to CSV
        metrics_data = []
        for stage_name, net_data in networks.items():
            metrics_data.append({
                'Region': region['name'],
                'Stage': stage_name,
                'Edges': net_data['n_edges'],
                'Density': net_data['density'],
                'Avg_Correlation': net_data['avg_corr'],
                'Samples': net_data['n_samples']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f'{region_dir}/network_metrics.csv', index=False)
        
        # Print summary
        print(f"\n   📊 Summary for {region['name']}:")
        print(metrics_df[['Stage', 'Edges', 'Density', 'Avg_Correlation']].to_string(index=False))
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📁 Output folders:")
    print("   • ./brain_network_evolution/[REGION]/evolution_figure.png")
    print("   • ./brain_network_evolution/[REGION]/evolution_with_stats.png")
    print("   • ./brain_network_evolution/[REGION]/network_metrics.csv")


# =================================================================
# SINGLE REGION ANALYSIS (for quick testing)
# =================================================================

def analyze_single_region(file_path, region_name):
    """Quick analysis for a single region"""
    
    print("="*80)
    print(f"🔍 Quick Analysis: {region_name}")
    print("="*80)
    
    analyzer = BrainNetworkEvolution()
    
    # Load data
    X, y = analyzer.load_dataset(file_path)
    
    # Remove NaN
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        X = X[~nan_mask]
        y = y[~nan_mask]
        print(f"   🧹 Removed {nan_mask.sum()} samples with NaN")
    
    # Create networks
    networks = analyzer.create_stage_networks(X, y, threshold=0.6)
    
    if networks:
        # Create output directory
        os.makedirs('./brain_network_evolution', exist_ok=True)
        
        # Generate figure
        analyzer.plot_evolution_figure(
            networks,
            region_name,
            f'./brain_network_evolution/{region_name}_evolution.png'
        )
        
        # Print metrics
        print(f"\n📊 Network Metrics:")
        for stage, data in networks.items():
            print(f"   {stage}: Edges={data['n_edges']}, Density={data['density']:.3f}, Avg Corr={data['avg_corr']:.3f}")
    
    return networks


# =================================================================
# MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    # Option 1: Analyze all regions
    analyze_all_regions()
    
    # Option 2: Analyze single region (uncomment to use)
    # analyze_single_region('RadiomicNor,EMCI,MCI,LMCI,AD BS.xlsx', 'BS')