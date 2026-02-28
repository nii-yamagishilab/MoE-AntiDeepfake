import torch
import torch.nn as nn
import torch.nn.functional as F


# class FlatteningLoss(nn.Module):
#     def forward(self, x):
#         return -F.log_softmax(x, dim=1).mean()

class FlatteningLoss(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.target_val = 1.0 / num_experts

    def forward(self, x):
        # x is already softmaxed
        target = torch.full_like(x, self.target_val)
        return F.mse_loss(x, target)

class MOE_ICASSP(nn.Module):
    def __init__(self, experts, freezing=True, return_weights=False):
        super(MOE_ICASSP, self).__init__()

        self.return_weights = return_weights
        self.experts = nn.ModuleList(experts)
        num_experts = len(experts)

        self.proj_dim = 32
        self.bn = nn.BatchNorm1d(self.proj_dim)

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert.latent_dim, self.proj_dim),
                self.bn,
                nn.LeakyReLU()
            )
            for expert in experts
        ])

        self.pooling = nn.Parameter(torch.ones(self.proj_dim))

        gating_input_dim = self.proj_dim * (num_experts + 1)
        self.gating_network = nn.Sequential(
            nn.Linear(gating_input_dim, 64),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, num_experts),
        )

        if freezing:
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = False

    def forward(self, x):

        if isinstance(x, list):
            results = [expert(feat) for expert, feat in zip(self.experts, x)]
        else:
            results = [expert(x) for expert in self.experts]

        outputs = [res[0] for res in results]
        embeddings = [res[1] for res in results]

        projected_embs = [proj(emb) for proj, emb in zip(self.proj, embeddings)]

        stacked = torch.stack(projected_embs, dim=0)
        combined = torch.prod(stacked, dim=0)
        weighted_combined = combined * self.pooling.unsqueeze(0)

        concatenated_embeddings = torch.cat(projected_embs + [weighted_combined], dim=1)

        gating_weights = self.gating_network(concatenated_embeddings)
        gating_weights = F.softmax(gating_weights, dim=-1).unsqueeze(dim=-1)

        stacked_logits = torch.stack(outputs, dim=1)
        expert_probs = torch.softmax(stacked_logits, dim=-1)

        combined_output = torch.sum(gating_weights * expert_probs, dim=1)
        weights_x_probs = gating_weights * expert_probs

        if self.return_weights:
            return combined_output, gating_weights
        else:  # return all for eval analysis
            return combined_output, gating_weights, weights_x_probs, expert_probs


class MOE_WIFS(nn.Module):
    def __init__(self, experts, freezing=True, return_weights=False):
        super(MOE_WIFS, self).__init__()

        self.return_weights = return_weights
        self.temperature = 1
        self.hidden_dim = 128
        self.proj_dim = 32

        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)

        self.proc_emb = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert.latent_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.GLU(),
                nn.Linear(self.hidden_dim // 2, self.proj_dim)
            )
            for expert in experts
        ])

        self.TransfEnc = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=32, nhead=4, dropout=0.1, dim_feedforward=512, batch_first=True, norm_first=True),
            nn.TransformerEncoderLayer(d_model=32, nhead=4, dropout=0.1, dim_feedforward=512, batch_first=True, norm_first=True),
        )
        self.linear_out = nn.Linear(32, 1)

        self.softmax = nn.Softmax(dim=1)

        if freezing:
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = False

    # def forward(self, x):
    #     if isinstance(x, list):
    #         results = [expert(feat) for expert, feat in zip(self.experts, x)]
    #     else:
    #         results = [expert(x) for expert in self.experts]
    #
    #     outputs = [res[0] for res in results]
    #     embeddings = [res[1] for res in results]
    #
    #     processed_embs = torch.stack([proc_emb(emb) for proc_emb, emb in zip(self.proc_emb, embeddings)], dim=1)
    #
    #     transf_out = self.TransfEnc(processed_embs)
    #
    #     gating_weights = self.linear_out(transf_out)
    #     gating_weights = self.softmax(gating_weights / self.temperature)
    #
    #     stacked_logits = torch.stack(outputs, dim=1)
    #     expert_probs = torch.sigmoid(stacked_logits)
    #
    #     combined_output = torch.sum(gating_weights * expert_probs.unsqueeze(-1), dim=1)
    #     weights_x_probs = gating_weights * expert_probs.unsqueeze(-1)
    #
    #     if self.return_weights:
    #         return combined_output, weights_x_probs
    #     else:  # return all for eval analysis
    #         return combined_output, gating_weights, weights_x_probs, expert_probs

    def forward(self, x):
        if isinstance(x, list):
            results = [expert(feat) for expert, feat in zip(self.experts, x)]
        else:
            results = [expert(x) for expert in self.experts]

        outputs = [res[0] for res in results]
        embeddings = [res[1] for res in results]

        processed_embs = torch.stack([proc_emb(emb) for proc_emb, emb in zip(self.proc_emb, embeddings)], dim=1)

        transf_out = self.TransfEnc(processed_embs)

        gating_weights = self.linear_out(transf_out)
        gating_weights = self.softmax(gating_weights / self.temperature)

        stacked_logits = torch.stack(outputs, dim=1)

        weighted_logits = gating_weights * stacked_logits.unsqueeze(-1)
        combined_output = torch.sum(weighted_logits, dim=1)

        if self.return_weights:
            return combined_output, weighted_logits
        else:
            return combined_output, gating_weights, weighted_logits, stacked_logits