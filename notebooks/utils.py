import os
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
from collections import defaultdict

METRIC_FILES = {
    # testing
    # "accuracy": {
    #     "compiled": "accuracy_at_k_results.json",
    #     "details": "accuracy_at_k_query_details.json"
    # },
    # "recall": {
    #     "compiled": "recall_at_k_results.json",
    #     "details": "recall_at_k_query_details.json"
    # },
    "map": {
        "compiled": "map_at_k_results.json",
        "details": "map_at_k_query_details.json"
    },
    "precision": {
        "compiled": "precision_at_k_results.json",
        "details": "precision_at_k_query_details.json"
    }
}

def parse_experiment_folder(folder_name):
    """
    Parse an experiment folder name into (experiment_type, timestamp)
    """
    # Regular expression to extract experiment type and timestamp from folder names.
    # Expected format: <experiment_type>_<YYYY-MM-DD_HH-MM-SS>
    timestamp_pattern = re.compile(r"(.*)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")
    match = timestamp_pattern.match(folder_name)
    if match:
        experiment_type = match.group(1)
        timestamp_str = match.group(2)
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        return experiment_type, timestamp
    return None, None

def get_latest_experiments(base_dir, specific_experiments=None):
    """
    Scan the base directory for experiment folders, group by experiment type,
    and return a dictionary with the most recent experiment for each type.
    
    If specific_experiments is provided as a list:
      - if empty, treat it as not filtering any experiment
      - if non-empty, only use folders whose names are in that list.
    """
    experiments = {}
    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if os.path.isdir(full_path):
            # If filtering by a provided list and it is non-empty, skip if not in list.
            if specific_experiments is not None and len(specific_experiments) > 0 and folder not in specific_experiments:
                continue

            exp_type, timestamp = parse_experiment_folder(folder)
            if exp_type is None:
                continue
            # Keep the most recent experiment per experiment type.
            if exp_type not in experiments or timestamp > experiments[exp_type]["timestamp"]:
                experiments[exp_type] = {"folder": folder, "timestamp": timestamp, "path": full_path}
    return experiments

def load_json_file(file_path):
    """
    Utility function to load a JSON file.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def aggregate_metrics(exp_path):
    """
    Load all compiled metric JSON files from a given experiment folder.
    Returns a dictionary with keys: 'accuracy', 'recall', 'map', 'precision'.
    """
    aggregated = {}
    for metric, files in METRIC_FILES.items():
        file_path = os.path.join(exp_path, files["compiled"])
        data = load_json_file(file_path)
        if data is not None:
            aggregated[metric] = data
    return aggregated

def plot_metric(metric_data, metric_name, exp_type):
    """
    Create a bar chart and a line chart for the metric at different k values.
    Expects the keys in metric_data to follow the pattern: 
    e.g. for accuracy: "accuracyAt1", "accuracyAt3", etc.
    """
    ks = []
    values = []
    # Build a regex based on the metric name. For example, 'accuracyAt(\d+)'.
    pattern = re.compile(fr"{metric_name}At(\d+)", re.IGNORECASE)
    for key, value in metric_data.items():
        m = pattern.match(key)
        if m:
            k = int(m.group(1))
            ks.append(k)
            values.append(value)
    if not ks:
        print(f"No valid keys found for metric: {metric_name}")
        return

    # Sort by k values
    ks, values = zip(*sorted(zip(ks, values)))
    
    # Create a figure with two subplots: bar chart and line chart.
    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(12, 5))
    
    fig.suptitle(f"{exp_type.capitalize()} - {metric_name.capitalize()} at k")
    
    # Bar chart
    ax_bar.bar(ks, values, color='skyblue')
    ax_bar.set_xlabel("k")
    ax_bar.set_ylabel(metric_name.capitalize())
    ax_bar.set_title("Bar Chart")
    ax_bar.set_ylim(0, 1)  # Set y-axis range from 0 to 1
    
    # Line chart
    ax_line.plot(ks, values, marker='o', linestyle='-', color='coral')
    ax_line.set_xlabel("k")
    ax_line.set_ylabel(metric_name.capitalize())
    ax_line.set_title("Line Chart")
    ax_line.set_ylim(0, 1)  # Set y-axis range from 0 to 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
def plot_all_metrics(aggregated_metrics, exp_type):
    """
    Plot bar and line charts for all metrics available in aggregated_metrics.
    """
    for metric, data in aggregated_metrics.items():
        if data is not None:
            # print(f"Plotting {metric} metrics...")
            plot_metric(data, metric, exp_type)
        else:
            print(f"No data available for metric: {metric}")
            
            
def plot_metric_comparison(metric_name, experiments):
    """
    Plot a comparison of a selected metric (e.g. "map") across multiple experiments.
    Each experiment's compiled metric JSON file is loaded and a line chart is generated
    with the metric values at different k values. The y-axis is fixed to [0, 1].
    The legend is sorted based on the greatest values at the last index.
    
    Parameters:
        metric_name (str): The metric to plot (e.g. "map", "accuracy", "recall", "precision")
        experiments (dict): A dictionary of experiments as returned by get_latest_experiments.
                            The keys are experiment types and the values are dicts containing folder info.
    """
    import matplotlib.pyplot as plt
    import re
    # Create a new figure for comparison
    plt.figure(figsize=(10, 7))
    
    # Build a regex pattern to extract k from the keys (e.g., "mapAt1", "mapAt3", etc.)
    pattern = re.compile(fr"{metric_name}At(\d+)", re.IGNORECASE)
    
    # Store experiment data for sorting
    exp_data = []
    
    for exp_type, exp_info in experiments.items():
        # Construct the file path for the compiled metric results for the selected metric
        file_path = os.path.join(exp_info["path"], METRIC_FILES[metric_name.lower()]["compiled"])
        data = load_json_file(file_path)
        if not data:
            print(f"Warning: No data for {metric_name} in experiment {exp_info['folder']}")
            continue
        
        # Extract the k values and corresponding metric values
        ks = []
        values = []
        for key, value in data.items():
            m = pattern.match(key)
            if m:
                k = int(m.group(1))
                ks.append(k)
                values.append(value)
        if not ks:
            print(f"No valid keys found for metric {metric_name} in experiment {exp_info['folder']}")
            continue
        
        # Sort by k
        ks, values = zip(*sorted(zip(ks, values)))
        
        # Store experiment data for sorting
        exp_data.append({
            'exp_type': exp_type,
            'exp_info': exp_info,
            'ks': ks,
            'values': values,
            'last_value': values[-1] if values else 0
        })
    
    # Sort experiments by the last value (descending)
    exp_data.sort(key=lambda x: x['exp_info']["folder"], reverse=False)
    
    def get_label(exp):
        """
        Generate a label for the experiment based on its folder name.
        """
        models = {
            "dinov2": "DINOv2",
            "dino_": "DINO",
            "vit_": "ViT",
            "resnet": "ResNet",
            "clip": "CLIP",
            "uni": "UNI",
            "UNI2-h": "UNI2",
            "virchow2": "Virchow2",
            "phikon-v2": "Phikon-v2",
            "phikon_": "Phikon",
        }
        
        m_map = round(sum(exp['values']) / len(exp['values']) *100, 2)
        std = round(np.std(exp['values']) * 100, 2)
        if std > 0:
            m_map = f"{m_map}% ± {std}"
        else:
            m_map = str(m_map)
            
        folder = exp['exp_info']["folder"]
        for key, value in models.items():
            if key in folder:
                return f'{value} ({m_map})'
    
    # Define fixed colors and line styles for each model type
    model_styles = {
        'DINOv2': {'color': '#1f77b4', 'marker': 'o'},   # Tableau blue
        'DINO': {'color': '#2ca02c', 'marker': 's'},     # Tableau green
        'ViT': {'color': '#ff7f0e', 'marker': '^'},      # Tableau orange
        'ResNet': {'color': '#9467bd', 'marker': 'D'},   # Tableau purple
        'CLIP': {'color': '#d62728', 'marker': 'v'},     # Tableau red
        'UNI': {'color': '#8c564b', 'marker': '*'},      # Tableau brown
        'Virchow2': {'color': '#e377c2', 'marker': 'P'},  # Tableau pink
        'Phikon-v2': {'color': '#bcbd22', 'marker': 'h'},  # Tableau olive
        'Phikon': {'color': '#7f7f7f', 'marker': 'x'},   # Tableau gray
        'UNI2': {'color': '#17becf', 'marker': 'X'},    # Tableau cyan
        
    }

    # Plot the sorted experiments using consistent styles
    handles = []
    labels = []

    # Adiciona separadores
    foundation_models = []
    pretrained_backbones = []

    for exp in exp_data:
        label = get_label(exp)
        model_type = next((k for k in model_styles.keys() if k in label), 'other')
        style = model_styles.get(model_type, {'color': 'gray', 'marker': 'x'})
        
        line, = plt.plot(
            exp['ks'], exp['values'], 
            color=style['color'], 
            marker=style['marker'],
            label=label
        )

        # Decide a qual grupo pertence
        if model_type in ['UNI', 'UNIv2', 'Phikon', 'Phikon-v2', 'Virchow2']:
            foundation_models.append((line, label))
        else:
            pretrained_backbones.append((line, label))

    # Junta as legendas com pseudo-subgrupos
    handles += [plt.Line2D([0], [0], color='none', label='— Foundation Models —')]
    labels += ['• Foundation Models  ']
    for h, l in foundation_models:
        handles.append(h)
        labels.append(l)

    handles += [plt.Line2D([0], [0], color='none', label='— Pretrained Backbones —')]
    labels += ['• Pretrained Backbones ']
    for h, l in pretrained_backbones:
        handles.append(h)
        labels.append(l)

    # Adiciona a legenda formatada

    plt.xlabel("k", fontsize=12)
    plt.ylabel(metric_name)
    plt.ylim(0, 1)  # Fix y-axis from 0 to 1
    plt.xlim(0, 16)  # Fix y-axis from 0 to 1
    plt.yticks(np.arange(0, 1.05, 0.1))  # Set y-axis ticks from 0 to 1 in steps of 0.5 print("•")
    plt.legend(handles=handles, labels=labels, loc='lower right')
    plt.grid(True)
    plt.show()
    
    
def load_json_file(file_path):
    """
    Load a JSON file and return its contents as a dictionary.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_class_map(query_retrievals_file):
    data = load_json_file(query_retrievals_file)
    if data is None:
        return {}
    
    queries = data.get("query_retrievals", [])
    k = data.get("k", None)
    if not queries:
        print("No query retrievals found in the file.")
        return {}
    
    class_stats = defaultdict(list)
    for query in queries:
        query_class = query.get("query_class")
        average_precision = query.get("average_precision")
        class_stats[query_class].append(average_precision)
    # Calculate mean average precision per class
    class_map = {}
    for cls, aps in class_stats.items():
        if aps:
            class_map[cls] = sum(aps) / len(aps)
        else:
            class_map[cls] = 0.0
        
    return class_map, k
            
    
# def plot_class_map(class_stats, title="", k=None):
#     """
#     Plot a bar chart of error rates per class.
#     """
#     if not class_stats:
#         print("No class statistics to plot.")
#         return
    
#     classes = list(class_stats.keys())
#     error_rates = [class_stats[c] for c in classes]
    
#     plt.figure(figsize=(10, 6))
#     plt.bar(classes, error_rates, color='salmon')
#     plt.xlabel("Class")
#     plt.ylabel("Average Precision")
#     plt.title(f"{title} - Average Precision at {k} per Class")
#     plt.ylim(0, 1)  # Fix y-axis from 0 to 1
#     plt.xticks(rotation=45)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()
 
def plot_class_map(class_stats, title="", k=None):
    """
    Plot a bar chart of average precision per class, formatted for scientific publication.
    """
    if not class_stats:
        print("No class statistics to plot.")
        return

    classes = list(class_stats.keys())
    error_rates = [class_stats[c] for c in classes]
    
    # Paleta colorblind-friendly
    colors = sns.color_palette("colorblind", len(classes))

    fig, ax = plt.subplots(figsize=(10, 6))  # IEEE-friendly size

    bars = ax.bar(classes, error_rates, color=colors, edgecolor='black')

    # Eixos e layout
    ax.set_xlabel("Class", fontsize=9)
    ax.set_ylabel(f"ap@{k}", fontsize=9)
    ax.set_ylim(0, 1)
    # ax.set_title(f"{title} - AP@{k}" if k is not None else title, fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xticklabels(classes, rotation=45, ha='right')

    # Fonte dos ticks
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_fontname('Times New Roman')

    # Anota valores sobre as barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8,
                   )

    # Remover bordas desnecessárias
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    
def plot_classes_maps(latest_experiments, metric_suffix='map_at_k_query_details.json'):
    for model, files in latest_experiments.items():
        path = files["path"]
        file = path + "/" + metric_suffix        
        class_map, k = calculate_class_map(file)
        print(f"Plotting class map for {model} with k={k}...")
        plot_class_map(class_map, title=f"{model} - Average Precision per Class", k=k)
        

def calculate_class_performance(query_retrievals_file):
    """
    Calculate performance per class using the query retrievals file.
    
    Expected JSON format:
    {
      "mapk": <float>,
      "k": <int>,
      "query_retrievals": [
         {
           "average_precision": <float>,
           "query_label": <int>,
           "query_class": <str>,
           "query_path": <str>,
           "retrieved": [
             {
               "k": <int>,
               "retrieved_label": <int>,
               "retrieved_class": <str>,
               "retrieved_path": <str>,
               "is_relevant": <int>,   # 1 if relevant, 0 otherwise
               "similarity": <float>
             },
             ...
           ]
         },
         ...
      ]
    }
    
    For each query, a query is considered "correct" if at least one retrieved item has is_relevant==1.
    
    Returns a dictionary with statistics per query class.
    """
    data = load_json_file(query_retrievals_file)
    if data is None:
        return {}
    
    queries = data.get("query_retrievals", [])
    if not queries:
        print("No query retrievals found in the file.")
        return {}
    
    class_stats = {}
    for query in queries:
        query_class = query.get("query_class", "Unknown")
        retrieved_items = query.get("retrieved", [])
        # A query is correct if any retrieved item is relevant.
        # found_correct = 1 if any(item.get("is_relevant", 0) == 1 for item in retrieved_items) else 0
        revelance_list = [item.get("is_relevant") for item in retrieved_items]

        if query_class not in class_stats:
            class_stats[query_class] = {"total": 0, "correct": 0, "errors": 0}
        class_stats[query_class]["total"] += len(revelance_list)
        class_stats[query_class]["correct"] += sum(revelance_list)
        class_stats[query_class]["errors"] += len(revelance_list) - sum(revelance_list)
    
    # Calculate error and accuracy rates per class.
    for cls, stats in class_stats.items():
        total = stats["total"]
        stats["error_rate"] = stats["errors"] / total if total > 0 else 0
        stats["accuracy_rate"] = stats["correct"] / total if total > 0 else 0
    
    return class_stats

def print_class_performance(class_stats):
    """
    Print performance statistics per class.
    """
    if not class_stats:
        print("No performance statistics available.")
        return
    
    print("Performance per class:")
    for cls, stats in class_stats.items():
        print(f"Class: {cls}")
        print(f"  Total queries: {stats['total']}")
        print(f"  Correct queries: {stats['correct']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Error rate: {stats['error_rate']:.2f}")
        print(f"  Accuracy rate: {stats['accuracy_rate']:.2f}")
        print()

def plot_class_performance(class_stats, title=""):
    """
    Plot a bar chart of error rates per class.
    """
    if not class_stats:
        print("No class statistics to plot.")
        return
    
    classes = list(class_stats.keys())
    error_rates = [class_stats[c]["error_rate"] for c in classes]
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, error_rates, color='salmon')
    plt.xlabel("Class")
    plt.ylabel("Error Rate")
    plt.title(f"{title} - Error Rate per Class")
    plt.ylim(0, 1)  # Fix y-axis from 0 to 1
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    
def plot_class_performances(latest_experiments, metric_suffix='map_at_k_query_details.json'):
    for metric, files in latest_experiments.items():
        path = files["path"]
        file = path + "/" + metric_suffix        
        performance = calculate_class_performance(file)
        plot_class_performance(performance, metric)
          
def plot_class_performance_comparison(file_path1, file_path2, label1="Experiment 1", label2="Experiment 2", title="Class Performance Comparison"):
    """
    Plot a bar chart comparing error rates per class for two experiments.
    """
    # Calculate performance for both experiments
    performance1 = calculate_class_performance(file_path1)
    performance2 = calculate_class_performance(file_path2)
    
    if not performance1 or not performance2:
        print("Insufficient data to plot comparison.")
        return
    
    # Get the union of all classes from both experiments
    all_classes = sorted(set(performance1.keys()).union(set(performance2.keys())))
    
    # Extract error rates for each class, defaulting to 0 if the class is missing
    error_rates1 = [performance1.get(cls, {}).get("error_rate", 0) for cls in all_classes]
    error_rates2 = [performance2.get(cls, {}).get("error_rate", 0) for cls in all_classes]
    
    # Plot the comparison
    x = np.arange(len(all_classes))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, error_rates1, width, label=label1, color='skyblue')
    plt.bar(x + width/2, error_rates2, width, label=label2, color='salmon')
    
    plt.xlabel("Class")
    plt.ylabel("Error Rate")
    plt.title(title)
    plt.ylim(0, 1)  # Fix y-axis from 0 to 1
    plt.xticks(x, all_classes, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
        
def permutation_test(model1, model2, confidence=95, n_permutations=10000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    dict1 = {item["query_path"]: item["average_precision"] for item in model1["query_retrievals"]}
    dict2 = {item["query_path"]: item["average_precision"] for item in model2["query_retrievals"]}

    common_queries = sorted(set(dict1.keys()) & set(dict2.keys()))
    prec1 = np.array([dict1[q] for q in common_queries])
    prec2 = np.array([dict2[q] for q in common_queries])

    observed_diff = np.mean(prec1 - prec2)

    diffs = []
    for _ in range(n_permutations):
        perm_prec1 = []
        perm_prec2 = []
        for p1, p2 in zip(prec1, prec2):
            if random.random() < 0.5:
                perm_prec1.append(p1)
                perm_prec2.append(p2)
            else:
                perm_prec1.append(p2)
                perm_prec2.append(p1)
        diffs.append(np.mean(np.array(perm_prec1) - np.array(perm_prec2)))

    diffs = np.array(diffs)
    p_value = np.mean(np.abs(diffs) >= abs(observed_diff))

    # Intervalo de confiança
    lower, upper = np.percentile(diffs, [(100 - confidence) / 2, 100 - (100 - confidence) / 2])
    ci = (lower, upper)

    # Verificação da hipótese nula
    reject_null = not (lower <= 0 <= upper)

    return {
        "observed_difference": observed_diff,
        "p_value": p_value,
        "permutation_distribution": diffs,
        "confidence_interval": ci,
        "reject_null": reject_null
    }


def plot_permutation_test_distribution(result):
    plt.figure(figsize=(10, 5))
    
    # Histograma da distribuição de permutação
    plt.hist(result["permutation_distribution"], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Linha da diferença observada
    plt.axvline(result["observed_difference"], color='red', linestyle='--', linewidth=2,
                label=f'Diferença Observada = {result["observed_difference"]:.4f}')
    
    # Linhas do intervalo de confiança
    lower, upper = result["confidence_interval"]
    plt.axvline(lower, color='green', linestyle='--', linewidth=2, label=f'IC 95% = [{lower:.2f}, {upper:.2f}]')
    plt.axvline(upper, color='green', linestyle='--', linewidth=2)
    
    plt.title("Permutation Test - Precision@k")
    plt.xlabel("Diferença média (Modelo 1 - Modelo 2)")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def permutation_anova(models, n_permutations=10000, seed=42, confidence=95):
    import numpy as np
    import random

    np.random.seed(seed)
    random.seed(seed)

    model_dicts = [
        {item["query_path"]: item["average_precision"] for item in model["query_retrievals"]}
        for model in models
    ]
    common_queries = sorted(set.intersection(*[set(md.keys()) for md in model_dicts]))
    data = np.array([[md[q] for md in model_dicts] for q in common_queries])

    observed_means = data.mean(axis=0)
    grand_mean = data.mean()
    ss_between = len(common_queries) * np.sum((observed_means - grand_mean) ** 2)

    ss_between_dist = []
    for _ in range(n_permutations):
        shuffled = data.copy()
        for row in shuffled:
            np.random.shuffle(row)
        perm_means = shuffled.mean(axis=0)
        perm_grand_mean = shuffled.mean()
        ss = len(common_queries) * np.sum((perm_means - perm_grand_mean) ** 2)
        ss_between_dist.append(ss)

    ss_between_dist = np.array(ss_between_dist)
    p_value = np.mean(ss_between_dist >= ss_between)
    ci = np.percentile(ss_between_dist, [(100 - confidence) / 2, 100 - (100 - confidence) / 2])
    reject_null = not (ci[0] <= ss_between <= ci[1])

    return {
        "ss_between": ss_between,
        "p_value": p_value,
        "permutation_distribution": ss_between_dist,
        "confidence_interval": ci,
        "reject_null": reject_null
    }



def plot_permutation_anova_distribution(result):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.hist(result["permutation_distribution"], bins=50, alpha=0.7, color='orchid', edgecolor='black')
    plt.axvline(result["ss_between"], color='red', linestyle='--', linewidth=2,
                label=f'SS Entre Grupos Observado = {result["ss_between"]:.4f}')
    lower, upper = result["confidence_interval"]
    plt.axvline(lower, color='green', linestyle='--', linewidth=2, label=f'IC 95% = [{lower:.2f}, {upper:.2f}]')
    plt.axvline(upper, color='green', linestyle='--', linewidth=2)
    plt.title("Permutation ANOVA - Precision@k")
    plt.xlabel("Soma dos quadrados entre grupos (SS Between)")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()