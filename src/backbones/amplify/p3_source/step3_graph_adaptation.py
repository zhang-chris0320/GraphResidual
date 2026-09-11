#!/usr/bin/env python3
"""P3 Graph-Residual self-supervised adaptation and graph controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path("/root/autodl-tmp/bio/disease_mutation_ptm_gcl")
OUT = ROOT / "work/amplify_generalization"
LEGACY_P3 = OUT / "05_graph_self_supervised"
P3 = OUT / "05_graph_self_supervised_v2_fusion_coupled"
CONTROLS = OUT / "11_graph_controls_v2_fusion_coupled"
CACHE = LEGACY_P3 / "cache"
GRAPH_EMB = ROOT / "work/graph_embedding_assets_v1/embeddings"
PROTOCOLS = OUT / "00_protocols"
FORMAL_SEEDS = (42, 3407, 2026)
HIDDEN = 640
GRAPH_DIM = 64
RELATIONS = {
    "mutation_on_protein": ("mutation", "protein"),
    "protein_has_ptm": ("protein", "ptm"),
    "protein_in_pathway": ("protein", "pathway"),
}
STEPS_PER_EPOCH = 817
MAX_EPOCHS = 100
MINIMUM_EPOCHS = 20
PATIENCE = 12
TRAIN_TOKENS_PER_EPOCH = 7_203_191
FAIR_FORMAL_EPOCHS = 20
FUSION_VERSION = "v2_fusion_coupled"
MIN_GATE_MEAN = 0.01
MIN_RESIDUAL_NORM = 1e-3


def json_default(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


class GraphResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_projection = nn.ModuleDict({node_type: nn.Linear(GRAPH_DIM, HIDDEN) for node_type in ("protein", "mutation", "ptm", "pathway")})
        self.alignment_head = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.gate = nn.Linear(2 * HIDDEN, 1)
        # A bounded positive scale avoids sign flips and unbounded residuals.
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.relation_diagonal = nn.Parameter(torch.ones(len(RELATIONS), HIDDEN))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def protein_graph(self, graph_feature: torch.Tensor) -> torch.Tensor:
        return self.alignment_head(self.type_projection["protein"](graph_feature))

    def fuse_components(self, sequence: torch.Tensor, protein_graph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Gate inputs must be on comparable scales: legacy P0 norms are ~842,
        # whereas graph projections are ~3, which previously saturated the gate.
        sequence_unit = F.normalize(sequence, dim=-1)
        graph_unit = F.normalize(protein_graph, dim=-1)
        gate = torch.sigmoid(self.gate(torch.cat([sequence_unit, graph_unit], dim=-1)))
        residual = gate * self.alpha * graph_unit
        fused_unit = F.normalize(sequence_unit + residual, dim=-1)
        # Preserve the original sequence norm so raw L2 comparisons are meaningful.
        fused = fused_unit * torch.linalg.vector_norm(sequence, dim=-1, keepdim=True).clamp_min(1e-8)
        return fused, gate, residual

    def fuse(self, sequence: torch.Tensor, protein_graph: torch.Tensor) -> torch.Tensor:
        return self.fuse_components(sequence, protein_graph)[0]

    def relation_score_projected(self, relation_index: int, src_h: torch.Tensor, dst_h: torch.Tensor) -> torch.Tensor:
        src_h = F.normalize(src_h, dim=-1)
        dst_h = F.normalize(dst_h, dim=-1)
        return (src_h * self.relation_diagonal[relation_index] * dst_h).sum(dim=-1) / 0.2


class ParameterMatchedMLP(nn.Module):
    def __init__(self, target_parameters: int):
        super().__init__()
        best_hidden = min(range(64, 1025), key=lambda width: abs((HIDDEN * width + width + width * HIDDEN + HIDDEN) - target_parameters))
        self.mlp = nn.Sequential(nn.Linear(HIDDEN, best_hidden), nn.GELU(), nn.Linear(best_hidden, HIDDEN))
        self.gate = nn.Linear(2 * HIDDEN, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.hidden = best_hidden

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(sequence)
        sequence_unit = F.normalize(sequence, dim=-1)
        delta_unit = F.normalize(delta, dim=-1)
        gate = torch.sigmoid(self.gate(torch.cat([sequence_unit, delta_unit], dim=-1)))
        fused_unit = F.normalize(sequence_unit + gate * self.alpha * delta_unit, dim=-1)
        return fused_unit * torch.linalg.vector_norm(sequence, dim=-1, keepdim=True).clamp_min(1e-8)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_protocol() -> tuple[list[dict], list[dict], dict[str, int]]:
    mapping = list(csv.DictReader((CACHE / "p0_protein_mapping.tsv").open(encoding="utf-8"), delimiter="\t"))
    train = [row for row in mapping if row["split"] == "train"]
    validation = [row for row in mapping if row["split"] == "validation"]
    graph_index = {}
    with (ROOT / "work/graph_core/node_mappings/protein.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            graph_index[row["uniprot_id"]] = int(row["node_index"])
    return train, validation, graph_index


class GraphData:
    def __init__(self):
        self.sequence = np.load(CACHE / "p0_protein_embeddings.npy", mmap_mode="r")
        self.graph = {
            node_type: np.load(GRAPH_EMB / f"{node_type}_embeddings.npy", mmap_mode="r")
            for node_type in ("protein", "mutation", "ptm", "pathway")
        }
        self.train_edges = dict(np.load(CACHE / "train_visible_edges.npz"))
        self.validation_edges = dict(np.load(CACHE / "validation_visible_edges.npz"))
        self.train_rows, self.validation_rows, self.graph_index = load_protocol()
        self.train_sequence_indices = np.asarray([int(row["row_index"]) for row in self.train_rows], dtype=np.int64)
        self.validation_sequence_indices = np.asarray([int(row["row_index"]) for row in self.validation_rows], dtype=np.int64)
        self.train_graph_indices = np.asarray([self.graph_index[row["protein_id"]] for row in self.train_rows], dtype=np.int64)
        self.validation_graph_indices = np.asarray([self.graph_index[row["protein_id"]] for row in self.validation_rows], dtype=np.int64)
        self.sequence_by_graph_index = np.full(len(self.graph["protein"]), -1, dtype=np.int64)
        for row in self.train_rows + self.validation_rows:
            graph_index = self.graph_index.get(row["protein_id"])
            if graph_index is not None:
                self.sequence_by_graph_index[graph_index] = int(row["row_index"])
        self.protein_permutation = self._degree_bin_permutation()

    def _degree_bin_permutation(self) -> np.ndarray:
        degree = np.zeros(len(self.graph["protein"]), dtype=np.int64)
        for name, array in self.train_edges.items():
            protein_column = 1 if name == "mutation_on_protein" else 0
            np.add.at(degree, array[:, protein_column], 1)
        positive = degree[degree > 0]
        boundaries = np.quantile(positive, [0.2, 0.4, 0.6, 0.8]) if len(positive) else np.zeros(4)
        bins = np.digitize(degree, boundaries)
        permutation = np.arange(len(degree), dtype=np.int64)
        rng = np.random.default_rng(42)
        for bin_id in range(5):
            indexes = np.where(bins == bin_id)[0]
            shuffled = indexes.copy()
            rng.shuffle(shuffled)
            permutation[indexes] = shuffled
        return permutation

    def features(self, node_type: str, indexes: np.ndarray, mode: str, rng: np.random.Generator) -> torch.Tensor:
        effective = indexes
        if mode == "shuffled" and node_type == "protein":
            effective = self.protein_permutation[indexes]
        elif mode == "random":
            effective = rng.integers(0, len(self.graph[node_type]), size=len(indexes), endpoint=False)
        values = np.asarray(self.graph[node_type][effective], dtype=np.float32)
        return torch.from_numpy(values).cuda(non_blocking=True)

    def sequence_features(self, indexes: np.ndarray) -> torch.Tensor:
        values = np.asarray(self.sequence[indexes], dtype=np.float32)
        return torch.from_numpy(values).cuda(non_blocking=True)

    def protein_endpoint(self, model: GraphResidualAdapter, graph_indexes: np.ndarray, mode: str, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_indexes = self.sequence_by_graph_index[graph_indexes]
        if np.any(sequence_indexes < 0):
            raise RuntimeError("protein relation endpoint is missing the locked sequence mapping")
        sequence = self.sequence_features(sequence_indexes)
        graph_raw = self.features("protein", graph_indexes, mode, rng)
        protein_graph = model.protein_graph(graph_raw)
        return model.fuse_components(sequence, protein_graph)

    def relation_endpoint(self, model: GraphResidualAdapter, node_type: str, indexes: np.ndarray, mode: str, rng: np.random.Generator) -> torch.Tensor:
        if node_type == "protein":
            return self.protein_endpoint(model, indexes, mode, rng)[0]
        return model.type_projection[node_type](self.features(node_type, indexes, mode, rng))


def graph_losses(model: GraphResidualAdapter, data: GraphData, split: str, mode: str, rng: np.random.Generator, config: dict, protein_batch: int, edge_batch: int) -> tuple[torch.Tensor, dict[str, float]]:
    if split == "train":
        sequence_pool, graph_pool, edges = data.train_sequence_indices, data.train_graph_indices, data.train_edges
    else:
        sequence_pool, graph_pool, edges = data.validation_sequence_indices, data.validation_graph_indices, data.validation_edges
    choice = rng.integers(0, len(sequence_pool), size=protein_batch)
    sequence = data.sequence_features(sequence_pool[choice])
    protein_graph_raw = data.features("protein", graph_pool[choice], mode, rng)
    protein_graph = model.protein_graph(protein_graph_raw)
    normalized_graph = F.normalize(protein_graph, dim=-1)
    fused, gate, residual = model.fuse_components(sequence, protein_graph)
    normalized_fused = F.normalize(fused, dim=-1)
    # Alignment now flows through the exact representation exported downstream.
    logits = normalized_fused @ normalized_graph.T / 0.07
    labels = torch.arange(len(choice), device="cuda")
    alignment_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    preservation_loss = (1.0 - F.cosine_similarity(fused, sequence, dim=-1)).mean()

    relation_parts = []
    for relation_index, (relation, (src_type, dst_type)) in enumerate(RELATIONS.items()):
        array = edges[relation]
        picked = rng.integers(0, len(array), size=edge_batch)
        pairs = array[picked]
        src_index, dst_index = pairs[:, 0], pairs[:, 1]
        if mode == "random":
            negative_dst = rng.integers(0, len(data.graph[dst_type]), size=edge_batch)
        else:
            negative_dst = rng.permutation(dst_index)
        # Protein endpoints use the fused sequence+graph representation.  This
        # prevents relation reconstruction from bypassing gate/alpha entirely.
        src_hidden = data.relation_endpoint(model, src_type, src_index, mode, rng)
        dst_hidden = data.relation_endpoint(model, dst_type, dst_index, mode, rng)
        neg_hidden = data.relation_endpoint(model, dst_type, negative_dst, mode, rng)
        positive_score = model.relation_score_projected(relation_index, src_hidden, dst_hidden)
        negative_score = model.relation_score_projected(relation_index, src_hidden, neg_hidden)
        relation_parts.append(0.5 * (F.binary_cross_entropy_with_logits(positive_score, torch.ones_like(positive_score)) + F.binary_cross_entropy_with_logits(negative_score, torch.zeros_like(negative_score))))
    relation_loss = torch.stack(relation_parts).mean()
    total = (
        float(config["lambda_relation"]) * relation_loss
        + float(config["lambda_alignment"]) * alignment_loss
        + float(config["lambda_preserve"]) * preservation_loss
    )
    # Hyperparameter configurations must be compared on a fixed, dimensionless
    # validation score; comparing their differently weighted totals would always
    # favour smaller lambdas.  Random baselines are ln(2) for relation BCE and
    # ln(batch) for the symmetric alignment classification task.
    selection_score = (
        relation_loss / math.log(2.0)
        + alignment_loss / math.log(float(len(choice)))
        + preservation_loss / 0.01
    )
    details = {
        "total_loss": float(total.detach()),
        "relation_loss": float(relation_loss.detach()),
        "alignment_loss": float(alignment_loss.detach()),
        "preservation_loss": float(preservation_loss.detach()),
        "alpha": float(model.alpha.detach()),
        "gate_mean": float(gate.detach().mean()),
        "gate_std": float(gate.detach().std()),
        "residual_norm_mean": float(torch.linalg.vector_norm(residual.detach(), dim=-1).mean()),
        "fused_cosine_to_sequence": float(F.cosine_similarity(fused.detach(), sequence, dim=-1).mean()),
        "selection_score": float(selection_score.detach()),
    }
    return total, details


def validate_graph(model: GraphResidualAdapter, data: GraphData, mode: str, config: dict, batches: int = 8) -> dict:
    model.eval()
    accum: dict[str, list[float]] = {}
    with torch.inference_mode():
        for batch in range(batches):
            rng = np.random.default_rng(900_000 + batch)
            _, details = graph_losses(model, data, "validation", mode, rng, config, 128, 256)
            for key, value in details.items():
                accum.setdefault(key, []).append(value)
    model.train()
    return {key: float(np.mean(values)) for key, values in accum.items()}


def fusion_counterfactual_diagnostics(model: GraphResidualAdapter, data: GraphData, sample_size: int = 512) -> dict:
    """Compare one trained adapter under real, shuffled, and random graph inputs."""
    model.eval()
    count = min(sample_size, len(data.validation_graph_indices))
    graph_indexes = data.validation_graph_indices[:count]
    sequence_indexes = data.validation_sequence_indices[:count]
    sequence = data.sequence_features(sequence_indexes)
    outputs = {}
    with torch.inference_mode():
        for offset, mode in enumerate(("real", "shuffled", "random")):
            rng = np.random.default_rng(771_000 + offset)
            graph_raw = data.features("protein", graph_indexes, mode, rng)
            protein_graph = model.protein_graph(graph_raw)
            fused, gate, residual = model.fuse_components(sequence, protein_graph)
            outputs[mode] = {
                "fused": fused,
                "gate": gate,
                "residual": residual,
                "graph_projection": protein_graph,
            }
        base_norm = torch.linalg.vector_norm(sequence, dim=-1).clamp_min(1e-8)
        real = outputs["real"]
        report = {
            "sample_size": count,
            "alpha": float(model.alpha.detach()),
            "gate_mean": float(real["gate"].mean()),
            "gate_std": float(real["gate"].std()),
            "gate_min": float(real["gate"].min()),
            "gate_max": float(real["gate"].max()),
            "residual_norm_mean": float(torch.linalg.vector_norm(real["residual"], dim=-1).mean()),
            "residual_to_base_norm_ratio": float((torch.linalg.vector_norm(real["residual"], dim=-1) / base_norm).mean()),
            "graph_projection_variance": float(real["graph_projection"].var(dim=0, unbiased=False).mean()),
            "fused_base_cosine_mean": float(F.cosine_similarity(real["fused"], sequence, dim=-1).mean()),
        }
        for mode in ("shuffled", "random"):
            report[f"real_vs_{mode}_fused_l2_mean"] = float(torch.linalg.vector_norm(real["fused"] - outputs[mode]["fused"], dim=-1).mean())
            report[f"real_vs_{mode}_fused_cosine_mean"] = float(F.cosine_similarity(real["fused"], outputs[mode]["fused"], dim=-1).mean())
            report[f"real_vs_{mode}_residual_l2_mean"] = float(torch.linalg.vector_norm(real["residual"] - outputs[mode]["residual"], dim=-1).mean())
    model.train()
    return report


def train_graph_run(config: dict, seed: int, stage: str, mode: str, data: GraphData, run_dir: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GraphResidualAdapter().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 3e-4)), weight_decay=0.01)
    if stage == "dry":
        epochs, steps_per_epoch, minimum_epochs, patience = 1, 20, 1, 1
    elif stage in {"search", "control"}:
        epochs, steps_per_epoch, minimum_epochs, patience = 1, 200, 1, 1
    else:
        epochs, steps_per_epoch, minimum_epochs, patience = FAIR_FORMAL_EPOCHS, STEPS_PER_EPOCH, MINIMUM_EPOCHS, PATIENCE
    total_steps = epochs * steps_per_epoch
    warmup = max(1, int(total_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1, step) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last_resume.pt"
    history = []
    best_selection_score, best_validation_loss = float("inf"), float("inf")
    best_epoch, stale, global_step, start_epoch = 0, 0, 0, 1
    if stage == "formal" and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_selection_score = state["best_selection_score"]
        best_validation_loss = state["best_validation_loss"]
        best_epoch, stale, global_step = state["best_epoch"], state["stale"], state["global_step"]
        start_epoch = state["epoch"] + 1
    started = time.time()
    stop_reason = "MAX_EPOCHS"
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        step_details: dict[str, list[float]] = {}
        gradient_norms = []
        epoch_started = time.time()
        for step in range(steps_per_epoch):
            rng = np.random.default_rng(seed * 10_000_000 + epoch * steps_per_epoch + step)
            optimizer.zero_grad(set_to_none=True)
            total, details = graph_losses(model, data, "train", mode, rng, config, 128, 256)
            total.backward()
            gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
            optimizer.step()
            scheduler.step()
            global_step += 1
            for key, value in details.items():
                step_details.setdefault(key, []).append(value)
        validation = validate_graph(model, data, mode, config)
        improved = validation["selection_score"] < best_selection_score - 1e-7
        if improved:
            best_selection_score = validation["selection_score"]
            best_validation_loss, best_epoch, stale = validation["total_loss"], epoch, 0
            atomic_torch_save({"model": model.state_dict(), "config": config, "mode": mode, "seed": seed, "epoch": epoch}, run_dir / "best.pt")
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(step_details["total_loss"])),
            "validation_loss": validation["total_loss"],
            "relation_loss": validation["relation_loss"],
            "alignment_loss": validation["alignment_loss"],
            "preservation_loss": validation["preservation_loss"],
            "alpha": validation["alpha"],
            "gate_mean": validation["gate_mean"],
            "gate_std": validation["gate_std"],
            "residual_norm_mean": validation["residual_norm_mean"],
            "fused_cosine_to_sequence": validation["fused_cosine_to_sequence"],
            "selection_score": validation["selection_score"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": float(np.mean(gradient_norms)),
            "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
            "epoch_seconds": time.time() - epoch_started,
            "trainable_parameters": parameter_count(model),
            "global_step": global_step,
            "equivalent_locked_sequence_tokens_seen": epoch * TRAIN_TOKENS_PER_EPOCH,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_validation_selection_score": best_selection_score,
            "patience_counter": stale,
        }
        history.append(row)
        write_tsv(run_dir / "training_history.tsv", history, list(history[0]))
        if stage == "formal":
            atomic_torch_save({
                "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "history": history,
                "best_selection_score": best_selection_score,
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch, "stale": stale, "global_step": global_step,
                "config": config, "mode": mode, "seed": seed,
            }, last_path)
        print(json.dumps({"stage": stage, "mode": mode, "seed": seed, **row}), flush=True)
        if stage == "formal" and epoch >= minimum_epochs and stale >= patience:
            stop_reason = "EARLY_STOPPING_VALIDATION_SELF_SUPERVISED_LOSS"
            break
        if shutil.disk_usage("/root/autodl-tmp").free / (1 << 30) < 10:
            raise RuntimeError("disk gate below 10 GiB")
    if stage == "formal" and history and history[-1]["epoch"] >= FAIR_FORMAL_EPOCHS and stop_reason == "MAX_EPOCHS":
        stop_reason = "FAIR_FIXED_TOKEN_BUDGET"
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    reloaded = GraphResidualAdapter().cuda()
    reloaded.load_state_dict(best["model"])
    reload_validation = validate_graph(reloaded, data, mode, config)
    if abs(reload_validation["selection_score"] - best_selection_score) > 1e-6:
        raise RuntimeError("P3 checkpoint reload mismatch")
    if stage == "formal":
        final_path = run_dir / "final_export.pt"
        if final_path.exists():
            final_path.unlink()
        os.link(run_dir / "best.pt", final_path)
    fusion_diagnostics = fusion_counterfactual_diagnostics(reloaded, data) if mode == "real" else None
    summary = {
        "status": "PASS",
        "method": "P3 Graph-Residual Self-supervised Adaptation",
        "fusion_version": FUSION_VERSION,
        "mode": mode,
        "stage": stage,
        "seed": seed,
        "config": config,
        "trainable_parameters": parameter_count(model),
        "backbone": "P0 Official AMPLIFY-120M frozen cached representations",
        "backbone_parameters_updated": False,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_selection_score": best_selection_score,
        "reload_validation": reload_validation,
        "fusion_utilization_pass": bool(
            reload_validation["gate_mean"] >= MIN_GATE_MEAN
            and reload_validation["residual_norm_mean"] >= MIN_RESIDUAL_NORM
        ),
        "epochs_completed": history[-1]["epoch"],
        "global_steps": global_step,
        "equivalent_locked_sequence_tokens_seen": history[-1]["equivalent_locked_sequence_tokens_seen"],
        "stop_reason": stop_reason,
        "max_epochs_configured": MAX_EPOCHS,
        "minimum_epochs_configured": MINIMUM_EPOCHS,
        "early_stopping_patience_configured": PATIENCE,
        "fair_fixed_epoch_budget": FAIR_FORMAL_EPOCHS if stage == "formal" else None,
        "selection_used_test": False,
        "elapsed_seconds": time.time() - started,
    }
    if fusion_diagnostics is not None:
        summary["fusion_diagnostics"] = fusion_diagnostics
    atomic_json(run_dir / "run_summary.json", summary)
    torch.cuda.empty_cache()
    return summary


def train_mlp_control(data: GraphData, target_parameters: int, seed: int, run_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ParameterMatchedMLP(target_parameters).cuda()
    count = parameter_count(model)
    ratio = abs(count - target_parameters) / target_parameters
    if ratio > 0.10:
        raise RuntimeError(f"parameter matching failed: {ratio}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(200):
        choice = rng.integers(0, len(data.train_sequence_indices), size=128)
        sequence = data.sequence_features(data.train_sequence_indices[choice])
        optimizer.zero_grad(set_to_none=True)
        fused = model(sequence)
        # Capacity-only control: denoising/preservation without graph input.
        loss = (1.0 - F.cosine_similarity(fused, sequence, dim=-1)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.inference_mode():
        choice = np.arange(min(512, len(data.validation_sequence_indices)))
        sequence = data.sequence_features(data.validation_sequence_indices[choice])
        validation = float((1.0 - F.cosine_similarity(model(sequence), sequence, dim=-1)).mean())
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"model": model.state_dict(), "target_parameters": target_parameters, "actual_parameters": count}, run_dir / "best.pt")
    summary = {
        "control": "C1 Parameter-matched MLP", "status": "PASS", "seed": seed,
        "target_parameters": target_parameters, "actual_parameters": count,
        "parameter_match_error_fraction": ratio, "train_preservation_loss": float(np.mean(losses)),
        "validation_preservation_loss": validation, "graph_used": False, "selection_used_test": False,
    }
    atomic_json(run_dir / "run_summary.json", summary)
    return summary


def search_configs() -> list[dict]:
    return [
        {"lambda_relation": relation, "lambda_alignment": alignment, "lambda_preserve": preserve, "learning_rate": 3e-4}
        for relation, alignment, preserve in itertools.product((0.05, 0.10), (0.05, 0.10), (0.01, 0.05, 0.10))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dry", "search", "controls", "formal"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not (LEGACY_P3 / "P3_CACHE_PREPARATION_PASS.flag").exists():
        raise RuntimeError("P3_CACHE_PREPARATION_PASS.flag is required")
    if shutil.disk_usage("/root/autodl-tmp").free / (1 << 30) < 10:
        raise RuntimeError("disk gate failed")
    data = GraphData()
    atomic_json(P3 / "P3_V2_FUSION_PROTOCOL.json", {
        "status": "LOCKED", "fusion_version": FUSION_VERSION,
        "legacy_cache": str(CACHE), "legacy_outputs_overwritten": False,
        "protein_relation_endpoint": "fused_sequence_graph_embedding",
        "alignment_endpoint": "fused_sequence_graph_embedding",
        "output_norm": "preserve_input_sequence_norm",
        "minimum_gate_mean": MIN_GATE_MEAN,
        "minimum_residual_norm": MIN_RESIDUAL_NORM,
        "selection_used_test": False,
    })
    if args.stage == "dry":
        config = search_configs()[0]
        summary = train_graph_run(config, 42, "dry", "real", data, P3 / "dry_run")
        atomic_json(P3 / "P3_DRY_RUN_PASS.flag", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
        return 0
    if args.stage == "search":
        results = []
        for index, config in enumerate(search_configs()):
            run_dir = P3 / "search" / f"trial_{index:02d}"
            summary_path = run_dir / "run_summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else train_graph_run(config, 42, "search", "real", data, run_dir)
            results.append({
                "trial": index, **config,
                "validation_loss": summary["best_validation_loss"],
                "validation_selection_score": summary["reload_validation"]["selection_score"],
            })
        results.sort(key=lambda row: (row["validation_selection_score"], row["trial"]))
        write_tsv(P3 / "validation_hyperparameter_search.tsv", results, list(results[0]))
        selected = {key: results[0][key] for key in search_configs()[0]}
        atomic_json(P3 / "selected_hyperparameters.json", {
            "status": "PASS",
            "selection_metric": "fixed_dimensionless_validation_selection_score",
            "selection_formula": "relation/ln(2)+alignment/ln(batch)+preservation/0.01",
            "selection_used_test": False, "seed": 42,
            "trial": results[0]["trial"], "selected": selected,
            "selected_validation_score": results[0]["validation_selection_score"],
        })
        print(json.dumps({"status": "PASS", "trials": len(results), "selected": selected}, ensure_ascii=False, indent=2))
        return 0
    selected_path = P3 / "selected_hyperparameters.json"
    if not selected_path.exists():
        raise RuntimeError("P3 validation-only selected hyperparameters required")
    config = json.loads(selected_path.read_text())["selected"]
    if args.stage == "controls":
        real = train_graph_run(config, 42, "control", "real", data, CONTROLS / "C0_real_graph")
        shuffled = train_graph_run(config, 42, "control", "shuffled", data, CONTROLS / "C2_shuffled_graph")
        random_graph = train_graph_run(config, 42, "control", "random", data, CONTROLS / "C3_random_graph")
        mlp = train_mlp_control(data, real["trainable_parameters"], 42, CONTROLS / "C1_parameter_matched_mlp")
        rows = [
            {"control": "C0 Real Graph", "validation_loss": real["best_validation_loss"], "validation_selection_score": real["best_validation_selection_score"], "trainable_parameters": real["trainable_parameters"], "graph_used": True},
            {"control": "C1 Parameter-matched MLP", "validation_loss": mlp["validation_preservation_loss"], "trainable_parameters": mlp["actual_parameters"], "graph_used": False},
            {"control": "C2 Shuffled Graph", "validation_loss": shuffled["best_validation_loss"], "validation_selection_score": shuffled["best_validation_selection_score"], "trainable_parameters": shuffled["trainable_parameters"], "graph_used": True},
            {"control": "C3 Random Graph", "validation_loss": random_graph["best_validation_loss"], "validation_selection_score": random_graph["best_validation_selection_score"], "trainable_parameters": random_graph["trainable_parameters"], "graph_used": True},
        ]
        write_tsv(CONTROLS / "controls_summary.tsv", rows, list(rows[0]))
        diagnostics = real.get("fusion_diagnostics", {})
        loss_gate = (
            real["best_validation_selection_score"] < shuffled["best_validation_selection_score"]
            and real["best_validation_selection_score"] < random_graph["best_validation_selection_score"]
        )
        utilization_gate = (
            real.get("fusion_utilization_pass", False)
            and diagnostics.get("gate_mean", 0.0) >= MIN_GATE_MEAN
            and diagnostics.get("residual_norm_mean", 0.0) >= MIN_RESIDUAL_NORM
            and diagnostics.get("real_vs_shuffled_fused_l2_mean", 0.0) > 1e-4
            and diagnostics.get("real_vs_random_fused_l2_mean", 0.0) > 1e-4
        )
        supported = bool(loss_gate and utilization_gate)
        flag = CONTROLS / ("P3_GRAPH_GAIN_SUPPORTED.flag" if supported else "P3_GRAPH_GAIN_NOT_SUPPORTED.flag")
        other = CONTROLS / ("P3_GRAPH_GAIN_NOT_SUPPORTED.flag" if supported else "P3_GRAPH_GAIN_SUPPORTED.flag")
        other.unlink(missing_ok=True)
        atomic_json(flag, {
            "status": "PASS" if supported else "NOT_SUPPORTED",
            "real": real["best_validation_loss"],
            "shuffled": shuffled["best_validation_loss"],
            "random": random_graph["best_validation_loss"],
            "real_selection_score": real["best_validation_selection_score"],
            "shuffled_selection_score": shuffled["best_validation_selection_score"],
            "random_selection_score": random_graph["best_validation_selection_score"],
            "real_better_than_both_controls": loss_gate,
            "fusion_utilization_pass": utilization_gate,
            "fusion_diagnostics": diagnostics,
            "selection_used_test": False,
        })
        print(flag.read_text())
        return 0
    if args.seed not in FORMAL_SEEDS:
        raise RuntimeError(f"formal seed must be one of {FORMAL_SEEDS}")
    if not (CONTROLS / "P3_GRAPH_GAIN_SUPPORTED.flag").exists():
        raise RuntimeError("P3 real graph did not pass control gate; formal P3 is not authorized")
    summary = train_graph_run(config, args.seed, "formal", "real", data, P3 / "formal" / f"seed_{args.seed}")
    if not summary.get("fusion_utilization_pass", False):
        raise RuntimeError("formal P3 fusion utilization collapsed")
    atomic_json(P3 / "formal" / f"seed_{args.seed}" / "FORMAL_PASS.flag", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
