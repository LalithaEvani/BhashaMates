# analysis.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import os
import json
import argparse

# --- Import your project's modules ---
from config import Config # Assuming your Config class is here or accessible
from data_handler import DataHandler
from models import get_model # Your function to instantiate the model
from utils import load_config, set_seed # Your utility functions

# Setup logging (optional for analysis, but can be helpful)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration for Analysis ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions (Mostly from previous version, adapted) ---

def load_model_for_analysis(model_checkpoint_path, config_obj, num_labels_for_model):
    """
    Loads a trained model from a checkpoint using your project's structure.
    Args:
        model_checkpoint_path: Path to the .pt file with the model's state_dict.
        config_obj: The full Config object used to train this model.
        num_labels_for_model: Number of labels the model was trained on.
    """
    logger.info(f"Loading model for analysis using config: {config_obj.model.model_name}")
    # Use your get_model function
    model = get_model(config_obj, num_labels_for_model)
    
    # Load the state dict
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info(f"Loaded model state_dict from {model_checkpoint_path}")
    return model

def get_predictions_and_features(model, dataloader):
    """Runs the model on the dataloader and collects features, logits, predictions, and labels."""
    all_features = []
    all_logits = []
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features/Preds"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE) # Assuming labels are present in test_loader for true_labels
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)


            # Call your model's forward pass
            # Ensure your model's forward pass can be called with labels=None for inference
            # and still return "features" and "logits"
            outputs_dict = model(input_ids=input_ids, 
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, 
                                 labels=None) # Pass None for labels during inference
            
            logits = outputs_dict["logits"]
            features = outputs_dict["features"] 

            all_features.append(features.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_features = np.concatenate(all_features, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_features, all_logits, all_preds, all_labels

# plot_tsne, calculate_geometric_metrics, plot_confusion_matrix (from previous version, should be fine)
# Ensure calculate_geometric_metrics handles num_classes correctly if not all classes are in the test subset.
def plot_tsne(features, labels, title="t-SNE visualization", perplexity=30, n_iter=1000, filename="tsne.png"):
    """Generates and saves a t-SNE plot."""
    logger.info(f"Running t-SNE for {title} (this may take a while for large datasets)...")
    # Subsample if features are too many to speed up t-SNE, e.g., 5000 samples
    if features.shape[0] > 5000:
        logger.info(f"Subsampling features from {features.shape[0]} to 5000 for t-SNE.")
        indices = np.random.choice(features.shape[0], 5000, replace=False)
        features = features[indices]
        labels = labels[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, features.shape[0]-1), 
                  n_iter=n_iter, init='pca', learning_rate='auto', n_jobs=-1) # Use n_jobs=-1 for parallelism
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(14, 12))
    unique_labels = np.unique(labels)
    # If you have many labels, plotting all might be too cluttered. Consider other cmap or no legend.
    cmap = plt.cm.get_cmap('viridis', len(unique_labels)) if len(unique_labels) <= 20 else plt.cm.get_cmap('viridis')

    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, s=10, alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    
    # Optional: Add a legend if you have class names and not too many classes
    if len(unique_labels) <= 20 : # Only add legend for fewer classes
        # Assuming you have a mapping from label_id to label_name
        # legend_labels = [intent_id_to_name_map[l] for l in sorted(unique_labels)]
        # handles, _ = scatter.legend_elements(prop="colors", alpha=0.6, num=sorted(unique_labels))
        # plt.legend(handles, legend_labels, title="Intents", bbox_to_anchor=(1.05, 1), loc='upper left')
        pass

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved t-SNE plot to {filename}")

def calculate_geometric_metrics(embeddings, labels, num_classes_total_dataset):
    """Calculates intra-class and inter-class centroid distances & Silhouette."""
    class_features = defaultdict(list)
    for i in range(len(labels)): # labels are 0-indexed class IDs
        class_features[labels[i]].append(embeddings[i])

    for k_label in class_features:
        class_features[k_label] = np.array(class_features[k_label])

    all_class_intra_dists_to_centroid = []
    class_centroids = {} 
    
    present_labels_in_subset = sorted(class_features.keys())

    for c_label in present_labels_in_subset:
        features_c = class_features[c_label]
        if features_c.shape[0] > 0: # Should always be true by construction
            centroid_c = np.mean(features_c, axis=0)
            class_centroids[c_label] = centroid_c
            if features_c.shape[0] > 1:
                dists = euclidean_distances(features_c, centroid_c.reshape(1, -1))
                all_class_intra_dists_to_centroid.append(np.mean(dists))
            else: # Single sample in this class for this subset
                all_class_intra_dists_to_centroid.append(0.0)

    avg_intra_class_dist = np.mean(all_class_intra_dists_to_centroid) if all_class_intra_dists_to_centroid else 0.0

    avg_inter_class_centroid_dist = 0.0
    # Use only centroids of classes present in the current subset for inter-dist
    unique_learned_centroids_list = [c for label, c in sorted(class_centroids.items())] 
    
    if len(unique_learned_centroids_list) > 1:
        centroids_array = np.array(unique_learned_centroids_list)
        pairwise_centroid_dists = euclidean_distances(centroids_array, centroids_array)
        upper_triangle_indices_centroids = np.triu_indices_from(pairwise_centroid_dists, k=1)
        if upper_triangle_indices_centroids[0].size > 0:
             avg_inter_class_centroid_dist = np.mean(pairwise_centroid_dists[upper_triangle_indices_centroids])
    
    sil_score = 0.0
    # silhouette_score requires at least 2 distinct labels in the provided `labels` array
    # and more than 1 sample overall.
    if len(np.unique(labels)) > 1 and len(labels) > 1:
        try:
            # Subsample for silhouette score if dataset is very large to avoid memory issues/long compute
            if embeddings.shape[0] > 10000:
                logger.info(f"Subsampling embeddings from {embeddings.shape[0]} to 10000 for Silhouette score.")
                indices = np.random.choice(embeddings.shape[0], 10000, replace=False)
                embeddings_sample = embeddings[indices]
                labels_sample = labels[indices]
                if len(np.unique(labels_sample)) > 1: # Ensure subsample still has >1 label
                    sil_score = silhouette_score(embeddings_sample, labels_sample, metric='euclidean')
                else:
                    logger.warning("Subsample for Silhouette has only 1 label, score is 0.")
            else:
                sil_score = silhouette_score(embeddings, labels, metric='euclidean')
        except ValueError as e:
            logger.warning(f"Could not calculate silhouette score: {e}")

    return avg_intra_class_dist, avg_inter_class_centroid_dist, sil_score


def plot_confusion_matrix(true_labels, pred_labels, class_names_list, title="Confusion Matrix", filename="confusion_matrix.png"):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names_list)))) # Ensure all classes are represented
    
    # Determine figure size based on number of classes
    figsize = (10, 8)
    if len(class_names_list) > 20:
        figsize = (max(20, len(class_names_list) * 0.3), max(18, len(class_names_list) * 0.3))
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True if len(class_names_list) <=30 else False, # Only annotate if not too cluttered
                fmt="d", cmap="Blues", 
                xticklabels=class_names_list, yticklabels=class_names_list)
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved confusion matrix to {filename}")


# --- Main Analysis Script ---
def run_analysis(analysis_args):
    """
    Main analysis function.
    Args:
        analysis_args: Parsed arguments for analysis (e.g., list of model output dirs).
    """
    
    analysis_results = {}
    # Base output directory for all analysis plots for this run
    analysis_output_dir = analysis_args.analysis_output_dir
    os.makedirs(analysis_output_dir, exist_ok=True)

    for model_output_dir in analysis_args.model_output_dirs:
        logger.info(f"\n--- Analyzing Model from Output Directory: {model_output_dir} ---")
        
        # 1. Load Configuration used for training this model
        config_path = os.path.join(model_output_dir, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at {config_path}. Skipping this model.")
            continue
        
        # Load config from JSON and reconstruct your Config object
        # This part needs to match how your Config object can be created from a dict
        with open(config_path, 'r') as f:
            config_dict_loaded = json.load(f)
        
        # Recreate your Config object. This is a common pattern:
        # You might need a helper in config.py: Config.from_dict(config_dict_loaded)
        # Or manually reconstruct:
        config = Config() # Assuming default Config()
        config.model.__dict__.update(config_dict_loaded.get("model", {}))
        config.data.__dict__.update(config_dict_loaded.get("data", {}))
        config.regularizers.__dict__.update(config_dict_loaded.get("regularizers", {}))
        config.training.__dict__.update(config_dict_loaded.get("training", {}))
        config.seed = config_dict_loaded.get("seed", 42)
        config.output_dir = config_dict_loaded.get("output_dir", model_output_dir) # Original output dir
        config.device = config_dict_loaded.get("device", "cuda")
        config.fp16 = config_dict_loaded.get("fp16", False)
        # Crucially, ensure config.model.model_name (or model_name_or_path) is set for get_model
        
        set_seed(config.seed) # Set seed for t-SNE, etc. consistency if desired

        # 2. Load Test Data (using the config of the trained model)
        logger.info("Loading test data...")
        data_handler = DataHandler(config) # Uses data_dir from loaded config
        data_handler.load_data() 
        _,_,_ = data_handler.prepare_data() # Tokenizes
        _, _, test_dataloader = data_handler.get_dataloaders()
        
        num_labels_from_data = data_handler.get_num_labels()
        intent_names_list = data_handler.get_intent_names() # List of intent string names

        # 3. Load Trained Model
        # Assuming your trainer saves the best model as "best_model.pt" in its output_dir
        model_checkpoint = os.path.join(model_output_dir, "best_model.pt") 
        if not os.path.exists(model_checkpoint):
            logger.error(f"Model checkpoint not found at {model_checkpoint}. Skipping this model.")
            continue
        
        # Use the number of labels the model was actually trained with
        model = load_model_for_analysis(model_checkpoint, config, num_labels_from_data)

        # 4. Extract Features and Predictions
        logger.info("Extracting features and predictions from test set...")
        features, logits, preds, true_labels = get_predictions_and_features(model, test_dataloader)

        # Define a model_key for storing results and naming files, e.g., from dir name or a config param
        model_key = os.path.basename(model_output_dir) # Or a more descriptive name
        analysis_results[model_key] = {}
        
        # Create a subdirectory for this model's analysis plots
        current_model_plot_dir = os.path.join(analysis_output_dir, model_key)
        os.makedirs(current_model_plot_dir, exist_ok=True)


        # 5. Basic Classification Metrics
        logger.info("Calculating classification metrics...")
        # Ensure labels for classification_report are 0 to N-1 if intent_names_list is used for display
        report_dict = classification_report(true_labels, preds, 
                                        labels=list(range(len(intent_names_list))), # Use all possible labels
                                        target_names=intent_names_list, 
                                        output_dict=True, zero_division=0)
        accuracy = report_dict["accuracy"]
        f1_macro = report_dict["macro avg"]["f1-score"]
        analysis_results[model_key]["accuracy"] = accuracy
        analysis_results[model_key]["f1_macro"] = f1_macro
        logger.info(f"Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
        # Save full report
        with open(os.path.join(current_model_plot_dir, "classification_report.txt"), "w") as f:
            f.write(classification_report(true_labels, preds, 
                                           labels=list(range(len(intent_names_list))),
                                           target_names=intent_names_list, zero_division=0))


        # 6. Confusion Matrix
        logger.info("Generating confusion matrix...")
        plot_confusion_matrix(true_labels, preds, intent_names_list, 
                              title=f"CM - {model_key}", 
                              filename=os.path.join(current_model_plot_dir, f"cm_{model_key}.png"))

        # 7. t-SNE Plot
        logger.info("Generating t-SNE plot...")
        plot_tsne(features, true_labels, 
                  title=f"t-SNE - {model_key}", 
                  filename=os.path.join(current_model_plot_dir, f"tsne_{model_key}.png"))

        # 8. Geometric Metrics
        logger.info("Calculating geometric metrics...")
        intra_dist, inter_dist, sil_score = calculate_geometric_metrics(features, true_labels, num_labels_from_data)
        analysis_results[model_key]["avg_intra_class_dist"] = intra_dist
        analysis_results[model_key]["avg_inter_class_centroid_dist"] = inter_dist
        analysis_results[model_key]["silhouette_score"] = sil_score
        logger.info(f"Avg Intra-Class Dist: {intra_dist:.4f}")
        logger.info(f"Avg Inter-Class Centroid Dist: {inter_dist:.4f}")
        logger.info(f"Silhouette Score: {sil_score:.4f}")
        
        # Optional: Analyze anchor behavior (more complex, needs access to model.afr_regularizer.class_anchors etc.)
        if hasattr(model, 'afr_regularizer') and config.regularizers.use_afr:
             logger.info("AFR Regularizer Anchors:")
             if model.afr_regularizer.anchor_type == 'class' and hasattr(model.afr_regularizer, 'class_anchors'):
                 class_anchors_np = model.afr_regularizer.class_anchors.detach().cpu().numpy()
                 logger.info(f"  Class Anchors shape: {class_anchors_np.shape}")
                 # Could save or plot norms, distances between anchors, etc.
             if hasattr(model.afr_regularizer, 'global_anchor'):
                 global_anchor_np = model.afr_regularizer.global_anchor.detach().cpu().numpy()
                 logger.info(f"  Global Anchor shape: {global_anchor_np.shape}, Norm: {np.linalg.norm(global_anchor_np):.4f}")


    # 9. Comparative Summary
    logger.info("\n\n--- Comparative Analysis Summary ---")
    summary_path = os.path.join(analysis_output_dir, "comparative_summary.txt")
    with open(summary_path, "w") as f_summary:
        header = f"{'Model Key':<50} | {'Accuracy':<10} | {'F1 Macro':<10} | {'Intra-Dist':<12} | {'Inter-Dist':<12} | {'Silhouette':<10}\n"
        separator = "-" * (50 + 13 + 13 + 15 + 15 + 13) + "\n"
        f_summary.write(header)
        f_summary.write(separator)
        print(header.strip())
        print(separator.strip())

        for model_name_key, metrics in analysis_results.items():
            line = (f"{model_name_key:<50} | "
                    f"{metrics.get('accuracy', 0.0):<10.4f} | "
                    f"{metrics.get('f1_macro', 0.0):<10.4f} | "
                    f"{metrics.get('avg_intra_class_dist', 0.0):<12.4f} | "
                    f"{metrics.get('avg_inter_class_centroid_dist', 0.0):<12.4f} | "
                    f"{metrics.get('silhouette_score', 0.0):<10.4f}\n")
            f_summary.write(line)
            print(line.strip())
    logger.info(f"Comparative summary saved to {summary_path}")


def parse_analysis_args():
    parser = argparse.ArgumentParser(description="Run analysis on trained intent classification models")
    parser.add_argument("--model_output_dirs", nargs='+', required=True,
                        help="List of paths to model output directories (each containing config.json and best_model.pt)")
    parser.add_argument("--analysis_output_dir", type=str, default="./analysis_results",
                        help="Directory to save all analysis plots and summaries")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_analysis_args()
    run_analysis(args)