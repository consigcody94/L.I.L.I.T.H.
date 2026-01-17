"""
Graph Attention Network (GATv2) Encoder for LILITH.

Learns spatial relationships between weather stations using
attention-based message passing on a geographic graph.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """
    Graph Attention Network v2 layer.

    Implements the improved attention mechanism from:
    "How Attentive are Graph Attention Networks?" (Brody et al., 2021)

    Key improvement: applies attention after the linear transformation,
    allowing the attention function to be a universal approximator.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        residual: bool = True,
        share_weights: bool = False,
    ):
        """
        Initialize GATv2 layer.

        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension (per head)
            num_heads: Number of attention heads
            dropout: Dropout probability
            edge_dim: Edge feature dimension (optional)
            residual: Whether to use residual connection
            share_weights: Share weights between source and target transformations
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.residual = residual

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        # Linear transformations for source and target nodes
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        if share_weights:
            self.W_dst = self.W_src
        else:
            self.W_dst = nn.Linear(in_dim, out_dim, bias=False)

        # Attention parameters (one per head)
        self.attn = nn.Parameter(torch.empty(num_heads, self.head_dim))

        # Edge feature projection (optional)
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False)
        else:
            self.edge_proj = None

        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        if residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.W_src.weight)
        if self.W_dst is not self.W_src:
            nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.attn)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features of shape (num_nodes, in_dim)
            edge_index: Graph connectivity of shape (2, num_edges)
            edge_attr: Edge features of shape (num_edges, edge_dim)

        Returns:
            Updated node features of shape (num_nodes, out_dim)
        """
        num_nodes = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # Linear transformations
        h_src = self.W_src(x)  # (num_nodes, out_dim)
        h_dst = self.W_dst(x)  # (num_nodes, out_dim)

        # Reshape for multi-head attention
        h_src = h_src.view(num_nodes, self.num_heads, self.head_dim)
        h_dst = h_dst.view(num_nodes, self.num_heads, self.head_dim)

        # Get source and destination features for each edge
        h_src_edge = h_src[src_idx]  # (num_edges, num_heads, head_dim)
        h_dst_edge = h_dst[dst_idx]  # (num_edges, num_heads, head_dim)

        # GATv2 attention: apply attention after transformation
        # a(Wh_i || Wh_j) -> LeakyReLU(a * (Wh_i + Wh_j))
        attn_input = h_src_edge + h_dst_edge  # (num_edges, num_heads, head_dim)

        # Add edge features if available
        if edge_attr is not None and self.edge_proj is not None:
            edge_h = self.edge_proj(edge_attr)  # (num_edges, out_dim)
            edge_h = edge_h.view(-1, self.num_heads, self.head_dim)
            attn_input = attn_input + edge_h

        # Compute attention scores
        attn_input = F.leaky_relu(attn_input, negative_slope=0.2)
        attn_scores = (attn_input * self.attn).sum(dim=-1)  # (num_edges, num_heads)

        # Normalize attention scores using softmax over neighbors
        attn_scores = self._sparse_softmax(attn_scores, dst_idx, num_nodes)
        attn_scores = self.attn_dropout(attn_scores)

        # Aggregate messages
        # Weighted sum of source features
        messages = h_src_edge * attn_scores.unsqueeze(-1)  # (num_edges, num_heads, head_dim)

        # Scatter-add messages to destination nodes
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst_idx.view(-1, 1, 1).expand_as(messages), messages)

        # Reshape and project
        out = out.view(num_nodes, self.out_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection
        if self.residual:
            if self.residual_proj is not None:
                x = self.residual_proj(x)
            out = out + x

        # Layer norm
        out = self.norm(out)

        return out

    def _sparse_softmax(
        self,
        scores: torch.Tensor,
        indices: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Compute softmax over sparse attention scores.

        Args:
            scores: Attention scores (num_edges, num_heads)
            indices: Destination node indices (num_edges,)
            num_nodes: Total number of nodes

        Returns:
            Normalized attention weights (num_edges, num_heads)
        """
        # Compute max for numerical stability
        max_scores = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        max_scores.scatter_reduce_(
            0,
            indices.view(-1, 1).expand_as(scores),
            scores,
            reduce="amax",
            include_self=False,
        )
        scores = scores - max_scores[indices]

        # Exp and sum
        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        sum_exp.scatter_add_(0, indices.view(-1, 1).expand_as(exp_scores), exp_scores)

        # Normalize
        return exp_scores / (sum_exp[indices] + 1e-8)


class GATEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network encoder.

    Processes station observations through multiple GAT layers to capture
    spatial dependencies between weather stations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        """
        Initialize GAT encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            edge_dim: Edge feature dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GAT layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            self.layers.append(
                GATv2Layer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim if i == 0 else None,  # Only use edge features in first layer
                    residual=True,
                )
            )

        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode station features through GAT layers.

        Args:
            x: Node features of shape (num_nodes, input_dim)
            edge_index: Graph connectivity of shape (2, num_edges)
            edge_attr: Edge features of shape (num_edges, edge_dim)
            batch: Batch assignment of shape (num_nodes,)

        Returns:
            Encoded features of shape (num_nodes, output_dim)
        """
        # Input projection
        h = self.input_proj(x)

        # Apply GAT layers
        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                edge_index,
                edge_attr if i == 0 else None,
            )

        # Output projection
        h = self.output_proj(h)
        h = self.output_norm(h)

        return h

    def forward_batched(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with batched graphs.

        This handles multiple graphs in a single batch by using
        the batch tensor to track which nodes belong to which graph.

        Args:
            x: Batched node features
            edge_index: Batched edge indices
            edge_attr: Batched edge attributes
            batch: Batch assignment tensor
            return_attention: Whether to return attention weights

        Returns:
            Encoded features and optionally attention weights
        """
        h = self.forward(x, edge_index, edge_attr, batch)

        if return_attention:
            # Attention weights from last layer would go here
            # For now, return None
            return h, None

        return h, None
