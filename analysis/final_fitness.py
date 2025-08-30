import datetime
import os
import re

from .constants_and_params import (
    MEAN_PLOTS_DIR,
    NUMBER_OF_ITERATIONS,
    OUTPUT_DIR,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




def plot_and_save_graphs_with_mean_best_results_for_each_iteration():
    
        
    # Custom label mapping for better readability
    label_mapping = {
        "Global_Basic": "Reputacja + Losowanie",
        "Global_TrustBasedAuction": "Reputacja + Aukcja",
        "Global_TrustBasedRoulette": "Reputacja + Ruletka", 
        "Local_Basic": "Zuafanie + Losowanie",
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
    
    for function_name in ["Griewank", "ExpandedSchaffer", "Ackley"]:
    
        fig, ax = plt.subplots(1, 1)
        
        final_positions = []
        
        for setup_name in [
            "Global_Basic_2025_8_20_17_39_53",
            "Global_TrustBasedAuction_2025_8_20_17_40_43",
            "Global_TrustBasedRoulette_2025_8_20_17_40_19",
            # "Local_Basic_2025_8_20_17_38_5",
            # "Local_TrustBasedAuction_2025_8_20_17_39_17",
            # "Local_TrustBasedRoulette_2025_8_20_17_38_31",
            # "NoTrust_Basic_2025_8_20_17_37_19", 
        ]:
            
            
            exp_iter = []
            exp_values = [] 
            
            for filename in os.listdir(f"{OUTPUT_DIR}/{setup_name}/{function_name}"):

                regex = r"exp_[0-9]+\.csv"
                
                if re.match(regex, filename):
                    current_df = pd.read_csv(f"{OUTPUT_DIR}/{setup_name}/{function_name}/{filename}")

                    current_df = current_df.loc[
                        current_df["generation"] <= NUMBER_OF_ITERATIONS
                    ]

                    current_df = current_df.loc[
                        current_df.groupby(["generation"])["score"].idxmin()
                    ]
                    current_df = current_df[["generation", "score"]]

                    exp_iter.extend(current_df["generation"].values.tolist())
                    exp_values.extend(current_df["score"].values.tolist())


            data = {"iter": exp_iter, "exp_value": exp_values}
            current_df = pd.DataFrame.from_dict(data)
            
            exp_data = []
            iter_labels = []
            std_data = []

            for iter_label in range(1, NUMBER_OF_ITERATIONS + 1):
                iter_labels.append(iter_label)

                iter_exp_values = current_df.loc[current_df["iter"] == iter_label]
                exp_data.append(iter_exp_values["exp_value"].values.mean().tolist())
                std_data.append(iter_exp_values["exp_value"].values.std(ddof=1).tolist())

            exp_data = np.array(exp_data)
            final_y = exp_data.min()
            std_data = np.array(std_data)
            
            # Get setup key (first two parts of the name) and map to clean label and color
            setup_key = '_'.join(setup_name.split('_')[:2])  # e.g., "Global_TrustBasedAuction"
            clean_label = label_mapping.get(setup_key, setup_key)  # Use mapping or fallback to key
            plot_color = color_mapping.get(setup_key, 'black')     # Get color or fallback to black
            
            ax.plot(iter_labels, exp_data, label=clean_label, color=plot_color, linewidth=2)
            ax.fill_between(
                iter_labels,
                exp_data - std_data,
                exp_data + std_data,
                alpha=0.2,
                color=plot_color
            )
            
            final_positions.append((iter_labels[-1], final_y))
        
        # Add annotations for each final position
        for j, (x, y) in reversed(list(enumerate(sorted(final_positions,reverse=True)))):    
            ax.annotate(
                f"{y:.2f}",  # Annotate with the final value (formatted to 2 decimals)
                (x, y),  # The point to annotate
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
                textcoords="offset points",  # Position the text relative to the point
                xytext=(25, -20*j),  # Offset the text by (x, y) pixels
                arrowprops=dict(arrowstyle="-", color="gray"),
                fontsize=10,
                color="black",
            )
            
        # Move legend outside the loop
        ax.legend()
        
        ax.set_title(f"Średnia wartość najlepszego dopasowania i odchylenie standardowe - Modele z Reputacją - {function_name}")
        ax.set_xlabel("Liczba ewaluacji")
        ax.set_ylabel("Średnia wartość najlepszego dopasowania")
        ax.set_yscale('log')

        # Plot saving.
        os.makedirs(MEAN_PLOTS_DIR, exist_ok=True)
        plt.subplots_adjust(
            left=0.2, bottom=0.1, right=0.8, top=0.95, wspace=0.4, hspace=0.4
        )
        fig.set_size_inches(10, 7)
        fig.savefig(f"{MEAN_PLOTS_DIR}/mean_global_{function_name}.png", dpi=100)
    
    print("Done")


if __name__ == "__main__":
    plot_and_save_graphs_with_mean_best_results_for_each_iteration()
