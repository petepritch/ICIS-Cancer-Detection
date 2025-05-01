import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiMutualAttention(nn.Module):
    def __init__(self, feature_dims, num_heads, head_dim): # Removed dropout for now
        """
        Initializes the MultiMutualAttention module (Bare Bones Version).

        Args:
            feature_dims (list or tuple): List containing the feature dimension
                                          of each input stream (e.g., [d_m1, d_m2, d_m3, d_m4, d_mt]).
            num_heads (int): The number of attention heads.
            head_dim (int): The dimension of each attention head's Q, K, V.
            # dropout (float): (Optional) Dropout probability - removed for now.
        """
        super().__init__()

        # --- Store configuration ---
        if not isinstance(feature_dims, (list, tuple)):
            raise TypeError("feature_dims must be a list or tuple")
        self.num_streams = len(feature_dims)
        if self.num_streams < 2:
            raise ValueError("MultiMutualAttention requires at least 2 feature streams.")

        self.feature_dims = feature_dims
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        print(f"Initializing MultiMutualAttention (Bare Bones):")
        print(f"  Number of streams: {self.num_streams}")
        print(f"  Feature dimensions: {self.feature_dims}")
        print(f"  Number of heads: {self.num_heads}")
        print(f"  Dimension per head: {self.head_dim}")
        print(f"  Inner projection dim: {self.inner_dim}")

        # --- Initialize ModuleLists for Q, K, V projections ---
        self.q_projs = nn.ModuleList()
        self.k_projs = nn.ModuleList()
        self.v_projs = nn.ModuleList()

        print(f"  Creating {self.num_streams * (self.num_streams - 1)} Q/K/V projection pathways...")

        # Loop through each stream 'i' (Query/target) and 'j' (Key/Value/source)
        for i in range(self.num_streams):
            for j in range(self.num_streams):
                if i == j: continue # Skip self-attention pathway

                dim_q_input = self.feature_dims[i]
                dim_kv_input = self.feature_dims[j]

                # Define layers for pathway (i queries j)
                q_proj = nn.Linear(dim_q_input, self.inner_dim, bias=False)
                k_proj = nn.Linear(dim_kv_input, self.inner_dim, bias=False)
                v_proj = nn.Linear(dim_kv_input, self.inner_dim, bias=False)

                # Append layers to the flat lists
                self.q_projs.append(q_proj)
                self.k_projs.append(k_proj)
                self.v_projs.append(v_proj)

        print(f"    Created {len(self.q_projs)} Q, {len(self.k_projs)} K, {len(self.v_projs)} V projection layers.")

        # --- Output projection layers ---
        # Project the summed attention output (size inner_dim) back to original dim
        self.out_projs = nn.ModuleList([
            nn.Linear(self.inner_dim, self.feature_dims[i]) for i in range(self.num_streams)
        ])
        print(f"  Created {len(self.out_projs)} output projection layers.")

        # --- Helper modules ---
        self.softmax = nn.Softmax(dim=-1) # Still need softmax for attention weights

        # Optional additions (commented out for now):
        # self.attn_dropout = nn.Dropout(dropout_rate)
        # self.resid_dropout = nn.Dropout(dropout_rate)
        # self.norm_layers = nn.ModuleList([nn.LayerNorm(self.feature_dims[i]) for i in range(self.num_streams)])

        print("-" * 30)

    def _get_pathway_idx(self, target_stream_idx, source_stream_idx):
        """
        Helper to calculate the flat index for the q/k/v_projs lists
        corresponding to target_stream_idx (i) querying source_stream_idx (j).
        Assumes the loops in __init__ filled the lists in row-major order (skipping i==j).
        """
        if target_stream_idx == source_stream_idx:
            raise ValueError("Cannot get pathway index for self-attention (i==j)")
        # Calculate index based on the nested loop structure in __init__
        k_minus_1 = self.num_streams - 1
        # Index = (full rows completed * paths_per_row) + position_in_current_row
        flat_index = target_stream_idx * k_minus_1 + \
                     (source_stream_idx if source_stream_idx < target_stream_idx else source_stream_idx - 1)
        return flat_index

    def forward(self, features):
        """
        Performs per-sample, cross-stream multi-mutual attention.

        Args:
            features (list or tuple): A list/tuple of K feature tensors,
                                      each of shape (batch_size, feature_dims[i]).

        Returns:
            torch.Tensor: The concatenated output tensor after mutual attention and fusion.
                          Shape (batch_size, sum_of_feature_dims).
        """
        # --- Input Validation ---
        if not isinstance(features, (list, tuple)) or len(features) != self.num_streams:
            raise TypeError(f"Input must be a list/tuple of {self.num_streams} tensors.")
        batch_size = features[0].shape[0]
        device = features[0].device

        # --- 1. Calculate Projections and Reshape ---
        # Pre-calculate all Q, K, V for all streams and pathways
        # Store them indexed for clarity, e.g., Qs[target_i][source_j]
        # Note: Q only depends on target i, K/V depend on source j
        Qs_per_target = [[] for _ in range(self.num_streams)]
        Ks_per_source = [[] for _ in range(self.num_streams)]
        Vs_per_source = [[] for _ in range(self.num_streams)]

        for i in range(self.num_streams): # Target stream index (Query: Si)
            for j in range(self.num_streams): # Source stream index (Key/Value: Sj)
                if i == j: continue

                pathway_idx = self._get_pathway_idx(i, j)

                # Project Q, K, V
                q = self.q_projs[pathway_idx](features[i]) # Shape: (batch, inner_dim)
                k = self.k_projs[pathway_idx](features[j]) # Shape: (batch, inner_dim)
                v = self.v_projs[pathway_idx](features[j]) # Shape: (batch, inner_dim)

                # Reshape for Multi-Head Attention: (batch, num_heads, head_dim)
                q = q.view(batch_size, self.num_heads, self.head_dim)
                k = k.view(batch_size, self.num_heads, self.head_dim)
                v = v.view(batch_size, self.num_heads, self.head_dim)

                # Store reshaped projections indexed by target_i and source_j
                # Qs_per_target[i] will contain Q vectors generated by stream i to query others
                # Ks_per_source[j] will contain K vectors generated by stream j to be queried by others
                # We store K/V based on the source stream 'j'
                # Note: We calculate multiple Qs for target i, and multiple K/Vs for source j
                # Let's store based on pathway (i, j) for simplicity in the next step
                Qs_per_target[i].append({"source_j": j, "q": q}) # Store Q with its intended source
                Ks_per_source[j].append({"target_i": i, "k": k}) # Store K with its intended target
                Vs_per_source[j].append({"target_i": i, "v": v}) # Store V with its intended target

        # --- 2. Calculate Per-Sample Attention and Combine ---
        refined_features = []
        for i in range(self.num_streams): # Target stream i
            # For target stream i, gather its queries Q(i->j) and the corresponding Keys K(j) and Values V(j)
            scores_i = [] # List to hold scores relative to sources [score(i->0), score(i->1), ...]
            values_j = [] # List to hold values from sources [V(0), V(1), ...] corresponding to scores

            for j in range(self.num_streams): # Potential source stream j
                if i == j: continue

                # Find the Q where target is i and source is j
                q_ij = next(item["q"] for item in Qs_per_target[i] if item["source_j"] == j)
                # Find the K where source is j and target is i
                k_ji = next(item["k"] for item in Ks_per_source[j] if item["target_i"] == i)
                # Find the V where source is j and target is i
                v_ji = next(item["v"] for item in Vs_per_source[j] if item["target_i"] == i)

                # Calculate PER-SAMPLE scores (dot product within each sample)
                # score_ij[b, n] = score between q_ij[b,n] and k_ji[b,n]
                score_ij = torch.einsum('bnh,bnh->bn', q_ij, k_ji) * self.scale
                scores_i.append(score_ij) # Shape (batch, num_heads)
                values_j.append(v_ji)     # Shape (batch, num_heads, head_dim)

            # Combine scores and values for target stream i
            # Stack scores along a new 'source stream' dimension
            all_scores_i = torch.stack(scores_i, dim=-1) # Shape: (batch, num_heads, K-1)
            # Stack values along a new 'source stream' dimension
            all_values_j = torch.stack(values_j, dim=2) # Shape: (batch, num_heads, K-1, head_dim)

            # Apply softmax across the source streams (dim=-1) to get attention weights
            attn_weights_i = F.softmax(all_scores_i, dim=-1) # Shape: (batch, num_heads, K-1)
            # Optional: attn_weights_i = self.attn_dropout(attn_weights_i)

            # Calculate weighted sum of values
            # weights need shape (b, n, k-1, 1) to broadcast with values (b, n, k-1, d)
            weighted_sum = (all_values_j * attn_weights_i.unsqueeze(-1)).sum(dim=2)
            # Sum across the source stream dim (dim=2). Result shape: (batch, num_heads, head_dim)
            summed_attn_outputs = weighted_sum

            # Reshape/flatten heads
            # Shape: (batch_size, num_heads * head_dim) = (batch_size, inner_dim)
            summed_attn_flat = summed_attn_outputs.reshape(batch_size, self.inner_dim)

            # Project back to original feature dimension
            projected_attn = self.out_projs[i](summed_attn_flat) # Shape: (batch_size, feature_dims[i])
            # Optional: projected_attn = self.resid_dropout(projected_attn)

            # Apply skip connection
            # Optional: Apply LayerNorm: features_normed = self.norm_layers[i](features[i])
            # refined_features.append(features_normed + projected_attn)
            refined_features.append(features[i] + projected_attn) # Bare bones

        # --- 3. Final Concatenation ---
        output = torch.cat(refined_features, dim=-1)
        # Expected shape: (batch_size, sum(feature_dims))

        return output