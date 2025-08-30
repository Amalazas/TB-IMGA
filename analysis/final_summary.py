import os
import datetime
import re

import matplotlib.pyplot as plt
import pandas as pd

from .constants_and_params import (
    OUTPUT_DIR,
    PLOTS_DIR,
)


def plot_and_save_summary_box_and_whiskers_comparision_graph_with_final_results():
    
    # Custom label mapping for better readability
    label_mapping = {
        "Global_Basic": "Reputacja + Losowanie",
        "Global_TrustBasedAuction": "Reputacja + Aukcja",
        "Global_TrustBasedRoulette": "Reputacja + Ruletka", 
        "Local_Basic": "Zaufanie + Losowanie",
        "Local_TrustBasedAuction": "Zaufanie + Aukcja",
        "Local_TrustBasedRoulette": "Zaufanie + Ruletka",
        "NoTrust_Basic": "Bazowe IMGA"
    }
    
    # High contrast color mapping
    color_mapping = {
        "Global_Basic": "#0000FF",                  # Bright Blue
        "Global_TrustBasedAuction": "#FF0000",      # Bright Red
        "Global_TrustBasedRoulette": "#00FF00",     # Bright Green  
        "Local_Basic": "#FF8000",                   # Orange
        "Local_TrustBasedAuction": "#8000FF",       # Purple
        "Local_TrustBasedRoulette": "#FF0080",      # Magenta
        "NoTrust_Basic": "#000000"                  # Black
    }

    # Create subplots for 3 problems
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    for problem_idx, function_name in enumerate(["Griewank", "ExpandedSchaffer", "Ackley"]):
        ax = axes[problem_idx]
        
        exp_data = []  # List of lists for boxplot data
        exp_labels = []  # Labels for x-axis
        exp_colors = []  # Colors for each box
        
        for setup_name in [
            "Global_Basic_2025_8_20_17_39_53",
            "Global_TrustBasedAuction_2025_8_20_17_40_43",
            "Global_TrustBasedRoulette_2025_8_20_17_40_19",
            "Local_Basic_2025_8_20_17_38_5",
            "Local_TrustBasedAuction_2025_8_20_17_39_17",
            "Local_TrustBasedRoulette_2025_8_20_17_38_31",
            "NoTrust_Basic_2025_8_20_17_37_19"
        ]:
            
            setup_final_values = []
            
            # Check if directory exists for this setup and function
            setup_dir = f"{OUTPUT_DIR}/{setup_name}/{function_name}"
            if not os.path.exists(setup_dir):
                continue
                
            for filename in os.listdir(setup_dir):
                regex = r"exp_[0-9]+\.csv"
                
                if re.match(regex, filename):
                    df = pd.read_csv(f"{setup_dir}/{filename}")
                    # Get the final (minimum) score for this experiment
                    final_score = df["score"].min()
                    setup_final_values.append(final_score)
            
            if setup_final_values:  # Only add if we have data
                exp_data.append(setup_final_values)
                
                # Get clean label and color
                setup_key = '_'.join(setup_name.split('_')[:2])
                clean_label = label_mapping.get(setup_key, setup_key)
                plot_color = color_mapping.get(setup_key, 'black')
                
                exp_labels.append(clean_label)
                exp_colors.append(plot_color)
        
        # Create boxplot
        if exp_data:
            box_plot = ax.boxplot(exp_data, labels=exp_labels, patch_artist=True)
            
            # Apply colors to boxes
            for patch, color in zip(box_plot['boxes'], exp_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Style the boxplot
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(box_plot[element], color='black')
        
        ax.set_title(f"Porównanie najlepszych wartości dopasowania - {function_name}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Ostateczna najlepsza wartość dopasowania", fontsize=12)
        # ax.set_yscale('log')  # Use log scale for better visualization
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if they're too long
        ax.tick_params(axis='x', rotation=45)

    # Adjust layout and save
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(f"{PLOTS_DIR}/final_summary_boxplots.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    plot_and_save_summary_box_and_whiskers_comparision_graph_with_final_results()
