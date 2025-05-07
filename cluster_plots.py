import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset
from tqdm.auto import tqdm
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def parse_args():
    p = argparse.ArgumentParser(description="Compare cluster projections across multiple models")
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--max_length",   type=int,   default=128)
    p.add_argument("--use_gpu",      action="store_true")
    p.add_argument("--cache_dir",    type=str,   default=None)
    p.add_argument("--umap_neighbors", type=int, default=30)
    p.add_argument("--umap_min_dist",  type=float, default=0.3)
    p.add_argument("--umap_metric",    type=str,   default="cosine")
    p.add_argument("--output_dir",     type=str,   default="./plots")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Banking77 dataset
    print("Loading Banking77 train split from Hugging Face...")
    ds = load_dataset("banking77", split="train", cache_dir=args.cache_dir)
    num_labels = ds.features["label"].num_classes
    print("Detected num_labels =", num_labels)


    models = [
        ("BERT Baseline",      r"/ssd_scratch/cvit/lalitha/INLP/inlp_project_checkpoints/trial_12_baseline_20250507_114931", "bert-large-uncased"),
        ("BERT Regularizer",   r"/ssd_scratch/cvit/lalitha/INLP/inlp_project_checkpoints/trial_13_IB_7e-03_AFR-global_2e-02_20250507_120037", "bert-large-uncased"),
        ("RoBERTa Baseline",    r"/ssd_scratch/cvit/lalitha/INLP/inlp_project_checkpoints/trial_13_baseline_20250507_135024", "roberta-large"),
        ("RoBERTa Regularizer", r"/ssd_scratch/cvit/lalitha/INLP/inlp_project_checkpoints/trial_12_IB_1e-02_AFR-global_7e-05_20250507_112803", "roberta-large"),
    ]


    fig_umap, axs_umap = plt.subplots(2, 2, figsize=(16,12))
    fig_pca,  axs_pca  = plt.subplots(2, 2, figsize=(16,12))

    for idx, (name, ckpt_dir, base_model) in enumerate(models):
        print(f"\n=== {name} ===")

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        config = AutoConfig.from_pretrained(base_model, num_labels=num_labels)

        cfg_file = os.path.join(ckpt_dir, "config_trial.json")
        if os.path.isfile(cfg_file):
            print(f"  → Applying overrides from {cfg_file}")
            with open(cfg_file) as f:
                meta = json.load(f)
            dp = meta.get("model", {}).get("dropout", None)
            if dp is not None:
                config.hidden_dropout_prob     = dp
                if hasattr(config, "classifier_dropout"):
                    config.classifier_dropout   = dp

        print("  → Loading model and head of size", config.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir,
            config=config,
        )
        model.to(device).eval()

        enc = lambda x: tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        ds_tok = ds.map(enc, batched=True)
        ds_tok.set_format(type="torch", columns=["input_ids","attention_mask","label"])
        loader = DataLoader(ds_tok, batch_size=args.batch_size, shuffle=False)

        embs, labs = [], []
        for batch in tqdm(loader, desc=name):
            inputs = {k:v.to(device) for k,v in batch.items() if k in ("input_ids","attention_mask")}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            cls_emb = out.hidden_states[-1][:,0,:]  
            embs.append(cls_emb.cpu())
            labs.append(batch["label"])
        embs = torch.cat(embs).numpy()
        labs = torch.cat(labs).numpy()

        umap_proj = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=42,
        ).fit_transform(embs)
        axu = axs_umap.flat[idx]
        axu.scatter(umap_proj[:,0], umap_proj[:,1], c=labs, s=5, cmap="tab20", alpha=0.8)
        axu.set_title(f"UMAP: {name}")
        axu.set_xticks([]); axu.set_yticks([])

        pca_proj = PCA(n_components=2).fit_transform(embs)
        axp = axs_pca.flat[idx]
        axp.scatter(pca_proj[:,0], pca_proj[:,1], c=labs, s=5, cmap="tab20", alpha=0.8)
        axp.set_title(f"PCA: {name}")
        axp.set_xticks([]); axp.set_yticks([])

    # Plots
    fig_umap.suptitle("UMAP Projections for All Models", fontsize=16)
    fig_umap.tight_layout(rect=[0,0.03,1,0.95])
    upath = os.path.join(args.output_dir, "umap_comparison.png")
    fig_umap.savefig(upath, dpi=300)
    print(f"\nSaved UMAP → {upath}")

    fig_pca.suptitle("PCA Projections for All Models", fontsize=16)
    fig_pca.tight_layout(rect=[0,0.03,1,0.95])
    ppath = os.path.join(args.output_dir, "pca_comparison.png")
    fig_pca.savefig(ppath, dpi=300)
    print(f"Saved PCA → {ppath}")

    plt.show()

if __name__ == "__main__":
    main()
