import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiMutualAttention(nn.Module):
    def __init__(self, feature_dims, num_heads, head_dim):
        super().__init__()

        if not isinstance(feature_dims, (list, tuple)):
            raise TypeError("feature_dims must be a list or tuple")
        self.num_streams = len(feature_dims)
        if self.num_streams < 2:
            raise ValueError(
                "MultiMutualAttention requires at least 2 feature streams."
            )

        self.feature_dims = feature_dims
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim**-0.5

        self.q_projs = nn.ModuleList()
        self.k_projs = nn.ModuleList()
        self.v_projs = nn.ModuleList()

        for i in range(self.num_streams):
            for j in range(self.num_streams):
                if i == j:
                    continue

                dim_q_input = self.feature_dims[i]
                dim_kv_input = self.feature_dims[j]

                q_proj = nn.Linear(dim_q_input, self.inner_dim, bias=False)
                k_proj = nn.Linear(dim_kv_input, self.inner_dim, bias=False)
                v_proj = nn.Linear(dim_kv_input, self.inner_dim, bias=False)

                self.q_projs.append(q_proj)
                self.k_projs.append(k_proj)
                self.v_projs.append(v_proj)

        self.out_projs = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, self.feature_dims[i])
                for i in range(self.num_streams)
            ]
        )

        self.softmax = nn.Softmax(dim=-1)

    def _get_pathway_idx(self, target_stream_idx, source_stream_idx):
        if target_stream_idx == source_stream_idx:
            raise ValueError("Cannot get pathway index for self-attention (i==j)")

        k_minus_1 = self.num_streams - 1
        flat_index = target_stream_idx * k_minus_1 + (
            source_stream_idx
            if source_stream_idx < target_stream_idx
            else source_stream_idx - 1
        )
        return flat_index

    def forward(self, features):
        if not isinstance(features, (list, tuple)) or len(features) != self.num_streams:
            raise TypeError(
                f"Input must be a list/tuple of {self.num_streams} tensors."
            )
        batch_size = features[0].shape[0]
        device = features[0].device

        Qs_per_target = [[] for _ in range(self.num_streams)]
        Ks_per_source = [[] for _ in range(self.num_streams)]
        Vs_per_source = [[] for _ in range(self.num_streams)]

        for i in range(self.num_streams):
            for j in range(self.num_streams):
                if i == j:
                    continue

                pathway_idx = self._get_pathway_idx(i, j)

                q = self.q_projs[pathway_idx](features[i])
                k = self.k_projs[pathway_idx](features[j])
                v = self.v_projs[pathway_idx](features[j])

                q = q.view(batch_size, self.num_heads, self.head_dim)
                k = k.view(batch_size, self.num_heads, self.head_dim)
                v = v.view(batch_size, self.num_heads, self.head_dim)

                Qs_per_target[i].append({"source_j": j, "q": q})
                Ks_per_source[j].append({"target_i": i, "k": k})
                Vs_per_source[j].append({"target_i": i, "v": v})

        refined_features = []
        for i in range(self.num_streams):
            scores_i = []
            values_j = []

            for j in range(self.num_streams):
                if i == j:
                    continue

                q_ij = next(
                    item["q"] for item in Qs_per_target[i] if item["source_j"] == j
                )
                k_ji = next(
                    item["k"] for item in Ks_per_source[j] if item["target_i"] == i
                )
                v_ji = next(
                    item["v"] for item in Vs_per_source[j] if item["target_i"] == i
                )

                score_ij = torch.einsum("bnh,bnh->bn", q_ij, k_ji) * self.scale
                scores_i.append(score_ij)
                values_j.append(v_ji)

            all_scores_i = torch.stack(scores_i, dim=-1)
            all_values_j = torch.stack(values_j, dim=2)

            attn_weights_i = F.softmax(all_scores_i, dim=-1)
            weighted_sum = (all_values_j * attn_weights_i.unsqueeze(-1)).sum(dim=2)
            summed_attn_outputs = weighted_sum

            summed_attn_flat = summed_attn_outputs.reshape(batch_size, self.inner_dim)

            projected_attn = self.out_projs[i](summed_attn_flat)
            refined_features.append(features[i] + projected_attn)

        output = torch.cat(refined_features, dim=-1)

        return output
