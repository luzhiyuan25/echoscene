"""Utility script to render before/after scene graph edits.

This demo constructs a tiny scene graph inline, renders the
original version and a manipulated version (with a new object and a
relationship change), and writes three PNGs:

* original_scene.png
* manipulated_scene.png
* side_by_side.png (matplotlib figure with the two graphs next to each other)

Dependencies: ``graphviz`` (with the Graphviz system package) and ``matplotlib``.
Run it directly with `python scripts/graph_manipulation_demo.py`.
"""
from __future__ import annotations

import os
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
from graphviz import Digraph

GraphEdges = Sequence[Tuple[str, str, str]]
GraphNodes = Sequence[Tuple[str, str, str]]


def render_graph(
    nodes: GraphNodes,
    edges: GraphEdges,
    title: str,
    output_dir: str,
    filename: str,
) -> str:
    """Render a single scene graph to ``PNG`` using Graphviz."""
    os.makedirs(output_dir, exist_ok=True)

    graph = Digraph(comment=title, format="png")
    for node_id, label, color in nodes:
        graph.node(node_id, label=label, color=color, fontname="helvetica", style="filled")

    for source, target, label in edges:
        graph.edge(source, target, label=label, color="grey")

    render_path = graph.render(os.path.join(output_dir, filename))
    return f"{render_path}.png"


def show_before_after(
    original_nodes: GraphNodes,
    original_edges: GraphEdges,
    manipulated_nodes: GraphNodes,
    manipulated_edges: GraphEdges,
    output_dir: str = "./vis_graphs/demo",
) -> None:
    """Render and save original/manipulated graphs and a side-by-side figure."""
    original_path = render_graph(original_nodes, original_edges, "Original scene", output_dir, "original_scene")
    manipulated_path = render_graph(
        manipulated_nodes,
        manipulated_edges,
        "Manipulated scene",
        output_dir,
        "manipulated_scene",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(plt.imread(original_path))
    axes[0].set_title("Original scene")
    axes[0].axis("off")

    axes[1].imshow(plt.imread(manipulated_path))
    axes[1].set_title("After graph edit")
    axes[1].axis("off")

    plt.tight_layout()
    side_by_side_path = os.path.join(output_dir, "side_by_side.png")
    plt.savefig(side_by_side_path, dpi=200)
    print(f"Saved original graph to: {original_path}")
    print(f"Saved manipulated graph to: {manipulated_path}")
    print(f"Saved side-by-side view to: {side_by_side_path}")


if __name__ == "__main__":
    # Define a minimal toy scene graph
    original_nodes: GraphNodes = [
        ("0", "sofa", "lightblue"),
        ("1", "coffee_table", "lightgreen"),
        ("2", "tv", "orange"),
    ]
    original_edges: GraphEdges = [
        ("0", "1", "in front of"),
        ("1", "2", "facing"),
    ]

    # Apply a graph manipulation: move TV to the left and add a plant
    manipulated_nodes: GraphNodes = [
        *original_nodes,
        ("3", "plant", "lightpink"),
    ]
    manipulated_edges: GraphEdges = [
        ("0", "1", "in front of"),
        ("2", "1", "left of"),
        ("3", "1", "next to"),
    ]

    show_before_after(original_nodes, original_edges, manipulated_nodes, manipulated_edges)
