#!/usr/bin/env python
# coding: utf-8

import os
import torch
from tqdm import tqdm
import plotly.express as px
import webbrowser
import http.server
import socketserver
import threading

# Disable gradient computation since we're only doing inference
torch.set_grad_enabled(False)

# Import required components from transformer_lens
from transformer_lens import HookedTransformer
from sae_lens import SAE, SAEConfig, load_sae_model

def main():
    print("Loading model and SAE...")
    
    # Load the model
    model = HookedTransformer.from_pretrained("gpt2-small")
    
    # Load the SAE
    sae = load_sae_model("open-source-sae/gpt2-small-sae-l2-act-0")
    
    # Create some sample text
    text = "Once upon a time"
    tokens = model.to_tokens(text)
    
    # Get model activations
    print(f"\nAnalyzing text: {text}")
    
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        
        # Get the activations from the specified layer
        activations = cache[sae.cfg.hook_name]
        
        # Run the SAE on these activations
        sae_out = sae.run_on_activations(activations)
        
        # Print some basic statistics
        print(f"\nShape of activations: {activations.shape}")
        print(f"Number of active features: {(sae_out > 0).sum().item()}")
        print(f"Mean activation value: {sae_out.mean().item():.4f}")
        
        # Get the top activated features
        top_features = torch.topk(sae_out[0, 0], k=5)
        print("\nTop 5 activated features:")
        for idx, value in zip(top_features.indices.tolist(), top_features.values.tolist()):
            print(f"Feature {idx}: {value:.4f}")

if __name__ == "__main__":
    main()
