import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 12


def create_mvt_full_model():
    """Create visualization of the full MVT model architecture"""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Define component colors
    colors = {
        "input": "#2C3E50",
        "spatial": "#3498DB",
        "spectral": "#2ECC71",
        "attention": "#E74C3C",
        "fusion": "#9B59B6",
        "output": "#F39C12",
    }

    # Define component positions
    components = [
        {
            "name": "Raw EEG Input",
            "x": 1,
            "y": 7,
            "width": 2,
            "height": 1,
            "type": "input",
        },
        {
            "name": "Spectral Embeddings\nScales 3,4,5",
            "x": 1,
            "y": 3,
            "width": 2,
            "height": 1,
            "type": "input",
        },
        {
            "name": "Spatial Feature\nExtractor",
            "x": 4,
            "y": 7,
            "width": 2,
            "height": 2,
            "type": "spatial",
        },
        {
            "name": "Multi-Scale Graph\nModule",
            "x": 4,
            "y": 3,
            "width": 2,
            "height": 2,
            "type": "spectral",
        },
        {
            "name": "Spatial Features",
            "x": 7,
            "y": 7,
            "width": 1.5,
            "height": 1,
            "type": "spatial",
        },
        {
            "name": "Graph Features",
            "x": 7,
            "y": 3,
            "width": 1.5,
            "height": 1,
            "type": "spectral",
        },
        {
            "name": "Cross-View\nAttention",
            "x": 9.5,
            "y": 5,
            "width": 1.5,
            "height": 2,
            "type": "attention",
        },
        {
            "name": "Fusion",
            "x": 12,
            "y": 5,
            "width": 1.5,
            "height": 2,
            "type": "fusion",
        },
        {
            "name": "Classifier",
            "x": 14.5,
            "y": 5,
            "width": 1.5,
            "height": 1,
            "type": "output",
        },
    ]

    # Draw components
    for comp in components:
        rect = patches.FancyBboxPatch(
            (comp["x"], comp["y"]),
            comp["width"],
            comp["height"],
            boxstyle=patches.BoxStyle("Round", pad=0.6, rounding_size=0.2),
            facecolor=colors[comp["type"]],
            edgecolor="black",
            linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(rect)

        # Add component name
        ax.text(
            comp["x"] + comp["width"] / 2,
            comp["y"] + comp["height"] / 2,
            comp["name"],
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    # Define arrows
    arrows = [
        {"start": (3, 7.5), "end": (4, 7.5)},  # EEG → Spatial Extractor
        {"start": (3, 3.5), "end": (4, 3.5)},  # Spectral → Graph Module
        {"start": (6, 7.5), "end": (7, 7.5)},  # Spatial Extractor → Spatial Features
        {"start": (6, 3.5), "end": (7, 3.5)},  # Graph Module → Graph Features
        {"start": (8.5, 7.5), "end": (9.5, 6)},  # Spatial Features → Attention
        {"start": (8.5, 3.5), "end": (9.5, 5)},  # Graph Features → Attention
        {"start": (11, 5.5), "end": (12, 5.5)},  # Attention → Fusion
        {"start": (13.5, 5.5), "end": (14.5, 5.5)},  # Fusion → Classifier
    ]

    # Draw arrows
    for arrow in arrows:
        ax.annotate(
            "",
            xy=arrow["end"],
            xytext=arrow["start"],
            arrowprops=dict(arrowstyle="->", lw=2, color="gray"),
        )

    # Add a legend
    legend_elements = [
        patches.Patch(facecolor=colors["input"], edgecolor="black", label="Input"),
        patches.Patch(
            facecolor=colors["spatial"], edgecolor="black", label="Spatial Processing"
        ),
        patches.Patch(
            facecolor=colors["spectral"], edgecolor="black", label="Spectral Processing"
        ),
        patches.Patch(
            facecolor=colors["attention"], edgecolor="black", label="Cross-Attention"
        ),
        patches.Patch(
            facecolor=colors["fusion"], edgecolor="black", label="Feature Fusion"
        ),
        patches.Patch(
            facecolor=colors["output"], edgecolor="black", label="Classification"
        ),
    ]
    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=3
    )

    # Add a title and labels
    ax.set_title(
        "Multi-View Transformer (MVT) Architecture for Alzheimer's Detection",
        fontsize=16,
        pad=20,
    )
    ax.set_xlim(0, 17)
    ax.set_ylim(1, 10)
    ax.axis("off")

    plt.tight_layout()
    return fig


def create_spatial_extractor_visualization():
    """Create visualization of the Spatial Feature Extractor"""
    G = nx.DiGraph()

    # Add nodes with positions
    nodes = {
        "input": {"pos": (0, 0), "label": "Raw EEG Input\n(B×19×T)"},
        "spatial_conv": {"pos": (2, 1), "label": "Spatial Conv\n(Dilated)"},
        "channel_conv": {"pos": (2, -1), "label": "Channel Conv\n(1D)"},
        "spatial_features": {"pos": (4, 1), "label": "Spatial Features"},
        "channel_features": {"pos": (4, -1), "label": "Channel Features"},
        "concat": {"pos": (6, 0), "label": "Feature\nConcatenation"},
        "encoder1": {"pos": (8, 0), "label": "MVT Encoder 1\n(Self-Attention)"},
        "encoder2": {"pos": (10, 0), "label": "MVT Encoder 2\n(Self-Attention)"},
        "encoder3": {"pos": (12, 0), "label": "MVT Encoder 3\n(Self-Attention)"},
        "output": {"pos": (14, 0), "label": "Spatial-Temporal\nFeatures"},
    }

    # Add nodes to graph
    for node_id, data in nodes.items():
        G.add_node(node_id, pos=data["pos"], label=data["label"])

    # Add edges
    edges = [
        ("input", "spatial_conv"),
        ("input", "channel_conv"),
        ("spatial_conv", "spatial_features"),
        ("channel_conv", "channel_features"),
        ("spatial_features", "concat"),
        ("channel_features", "concat"),
        ("concat", "encoder1"),
        ("encoder1", "encoder2"),
        ("encoder2", "encoder3"),
        ("encoder3", "output"),
    ]
    G.add_edges_from(edges)

    # Get node positions
    pos = nx.get_node_attributes(G, "pos")

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Define node colors by type
    node_colors = {
        "input": "#2C3E50",
        "spatial_conv": "#3498DB",
        "channel_conv": "#3498DB",
        "spatial_features": "#3498DB",
        "channel_features": "#3498DB",
        "concat": "#9B59B6",
        "encoder1": "#E74C3C",
        "encoder2": "#E74C3C",
        "encoder3": "#E74C3C",
        "output": "#F39C12",
    }

    # Draw nodes
    for node, (x, y) in pos.items():
        color = node_colors[node]
        rect = patches.FancyBboxPatch(
            (x - 0.9, y - 0.5),
            1.8,
            1,
            boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
            facecolor=color,
            edgecolor="black",
            linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(rect)

        # Add node label
        ax.text(
            x,
            y,
            nodes[node]["label"],
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    # Draw edges
    for u, v in G.edges():
        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops=dict(
                arrowstyle="->", lw=2, color="gray", shrinkA=30, shrinkB=30
            ),
        )

    # Add title
    ax.set_title("Spatial Feature Extractor Architecture", fontsize=16, pad=20)
    ax.axis("off")
    plt.tight_layout()

    return fig


def create_multiscale_graph_visualization():
    """Create visualization of the Multi-Scale Graph Module"""
    fig = plt.figure(figsize=(15, 10))

    # Define component colors
    colors = {
        "input": "#2C3E50",
        "embed": "#2ECC71",
        "attention": "#E74C3C",
        "fusion": "#9B59B6",
        "graph": "#3498DB",
        "output": "#F39C12",
    }

    # Manually define component positions and sizes
    components = [
        # Scale 3 path
        {
            "name": "Scale 3 Input\n(B×24×3)",
            "x": 1,
            "y": 8,
            "width": 2,
            "height": 1,
            "type": "input",
        },
        {
            "name": "Scale 3\nEmbedding",
            "x": 4,
            "y": 8,
            "width": 2,
            "height": 1,
            "type": "embed",
        },
        {
            "name": "Scale 3\nAttention",
            "x": 7,
            "y": 8,
            "width": 2,
            "height": 1,
            "type": "attention",
        },
        # Scale 4 path
        {
            "name": "Scale 4 Input\n(B×24×4)",
            "x": 1,
            "y": 5,
            "width": 2,
            "height": 1,
            "type": "input",
        },
        {
            "name": "Scale 4\nEmbedding",
            "x": 4,
            "y": 5,
            "width": 2,
            "height": 1,
            "type": "embed",
        },
        {
            "name": "Scale 4\nAttention",
            "x": 7,
            "y": 5,
            "width": 2,
            "height": 1,
            "type": "attention",
        },
        # Scale 5 path
        {
            "name": "Scale 5 Input\n(B×24×5)",
            "x": 1,
            "y": 2,
            "width": 2,
            "height": 1,
            "type": "input",
        },
        {
            "name": "Scale 5\nEmbedding",
            "x": 4,
            "y": 2,
            "width": 2,
            "height": 1,
            "type": "embed",
        },
        {
            "name": "Scale 5\nAttention",
            "x": 7,
            "y": 2,
            "width": 2,
            "height": 1,
            "type": "attention",
        },
        # Fusion and Graph components
        {
            "name": "Concatenation",
            "x": 10,
            "y": 5,
            "width": 2,
            "height": 1,
            "type": "fusion",
        },
        {
            "name": "Dynamic\nAdjacency",
            "x": 10,
            "y": 3,
            "width": 2,
            "height": 1,
            "type": "graph",
        },
        {
            "name": "Graph\nConvolution",
            "x": 13,
            "y": 5,
            "width": 2,
            "height": 1,
            "type": "graph",
        },
        {
            "name": "Graph Attention\nNetworks",
            "x": 16,
            "y": 5,
            "width": 2,
            "height": 1.5,
            "type": "attention",
        },
        {
            "name": "Global Pooling",
            "x": 19,
            "y": 5,
            "width": 2,
            "height": 1,
            "type": "fusion",
        },
        {
            "name": "Graph Features",
            "x": 22,
            "y": 5,
            "width": 2,
            "height": 1,
            "type": "output",
        },
    ]

    # Create a new Axes
    ax = fig.add_subplot(111)

    # Draw components
    for comp in components:
        rect = patches.FancyBboxPatch(
            (comp["x"], comp["y"]),
            comp["width"],
            comp["height"],
            boxstyle=patches.BoxStyle("Round", pad=0.6, rounding_size=0.2),
            facecolor=colors[comp["type"]],
            edgecolor="black",
            linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(rect)

        # Add component name
        ax.text(
            comp["x"] + comp["width"] / 2,
            comp["y"] + comp["height"] / 2,
            comp["name"],
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    # Define arrows
    arrows = [
        # Scale 3 path
        {"start": (3, 8.5), "end": (4, 8.5)},  # Input → Embedding
        {"start": (6, 8.5), "end": (7, 8.5)},  # Embedding → Attention
        {"start": (9, 8.5), "end": (10, 5.5)},  # Attention → Concat
        # Scale 4 path
        {"start": (3, 5.5), "end": (4, 5.5)},  # Input → Embedding
        {"start": (6, 5.5), "end": (7, 5.5)},  # Embedding → Attention
        {"start": (9, 5.5), "end": (10, 5.5)},  # Attention → Concat
        # Scale 5 path
        {"start": (3, 2.5), "end": (4, 2.5)},  # Input → Embedding
        {"start": (6, 2.5), "end": (7, 2.5)},  # Embedding → Attention
        {"start": (9, 2.5), "end": (10, 5.5), "curved": True},  # Attention → Concat
        # Graph path
        {"start": (11, 3.5), "end": (13, 5.3)},  # Adjacency → Graph Conv
        {"start": (12, 5.5), "end": (13, 5.5)},  # Concat → Graph Conv
        {"start": (15, 5.5), "end": (16, 5.5)},  # Graph Conv → GAT
        {"start": (18, 5.5), "end": (19, 5.5)},  # GAT → Pooling
        {"start": (21, 5.5), "end": (22, 5.5)},  # Pooling → Output
        # Dynamic adjacency computation
        {"start": (11, 5.0), "end": (11, 4.0)},  # From concat to dynamic adjacency
    ]

    # Draw arrows
    for arrow in arrows:
        if arrow.get("curved", False):
            # For curved arrows
            connectionstyle = "arc3,rad=0.3"
        else:
            connectionstyle = "arc3,rad=0"

        ax.annotate(
            "",
            xy=arrow["end"],
            xytext=arrow["start"],
            arrowprops=dict(
                arrowstyle="->", connectionstyle=connectionstyle, lw=2, color="gray"
            ),
        )

    # Add frequency band information
    freq_bands = [
        {"x": 1, "y": 9.2, "text": "Scale 3: [(0.5-8Hz), (8-13Hz), (13-30Hz)]"},
        {
            "x": 1,
            "y": 6.2,
            "text": "Scale 4: [(0.5-4Hz), (4-8Hz), (8-13Hz), (13-30Hz)]",
        },
        {
            "x": 1,
            "y": 3.2,
            "text": "Scale 5: [(0.5-4Hz), (4-8Hz), (8-13Hz), (13-30Hz), (30-50Hz)]",
        },
    ]

    for band in freq_bands:
        ax.text(
            band["x"],
            band["y"],
            band["text"],
            fontsize=9,
            bbox=dict(
                facecolor="white", alpha=0.7, edgecolor="gray", boxstyle="round,pad=0.5"
            ),
        )

    # Add a legend
    legend_elements = [
        patches.Patch(facecolor=colors["input"], edgecolor="black", label="Input"),
        patches.Patch(
            facecolor=colors["embed"], edgecolor="black", label="Feature Embedding"
        ),
        patches.Patch(
            facecolor=colors["attention"],
            edgecolor="black",
            label="Attention Mechanism",
        ),
        patches.Patch(
            facecolor=colors["fusion"], edgecolor="black", label="Feature Fusion"
        ),
        patches.Patch(
            facecolor=colors["graph"], edgecolor="black", label="Graph Operations"
        ),
        patches.Patch(facecolor=colors["output"], edgecolor="black", label="Output"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=3
    )

    # Add a title
    ax.set_title("Multi-Scale Graph Module Architecture", fontsize=16, pad=20)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 10)
    ax.axis("off")

    plt.tight_layout()
    return fig


def create_cross_attention_visualization():
    """Create an interactive Plotly visualization of the cross-attention mechanism"""
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scatter"}]],
        subplot_titles=["Cross-Attention Integration Mechanism"],
    )

    # Define node positions
    nodes = {
        "spatial_features": {"x": 1, "y": 8, "size": 25, "label": "Spatial Features"},
        "graph_features": {"x": 1, "y": 2, "size": 25, "label": "Graph Features"},
        "s2g_attention": {
            "x": 4,
            "y": 6.5,
            "size": 20,
            "label": "Spatial→Graph\nAttention",
        },
        "g2s_attention": {
            "x": 4,
            "y": 3.5,
            "size": 20,
            "label": "Graph→Spatial\nAttention",
        },
        "spatial_attended1": {"x": 7, "y": 8, "size": 25, "label": "Spatial Attended"},
        "graph_attended1": {"x": 7, "y": 2, "size": 25, "label": "Graph Attended"},
        "s2g_attention2": {
            "x": 10,
            "y": 6.5,
            "size": 20,
            "label": "Spatial→Graph\nAttention 2",
        },
        "g2s_attention2": {
            "x": 10,
            "y": 3.5,
            "size": 20,
            "label": "Graph→Spatial\nAttention 2",
        },
        "spatial_attended2": {
            "x": 13,
            "y": 8,
            "size": 25,
            "label": "Spatial Attended 2",
        },
        "graph_attended2": {"x": 13, "y": 2, "size": 25, "label": "Graph Attended 2"},
        "pool_spatial": {"x": 16, "y": 8, "size": 20, "label": "Attention\nPooling"},
        "pool_graph": {"x": 16, "y": 2, "size": 20, "label": "Attention\nPooling"},
        "fusion": {
            "x": 19,
            "y": 5,
            "size": 30,
            "label": "Feature Fusion\n(Multi-head Attention)",
        },
        "output": {"x": 22, "y": 5, "size": 25, "label": "Fused Representation"},
    }

    # Define node colors by type
    node_colors = {
        "spatial_features": "#3498DB",
        "graph_features": "#2ECC71",
        "s2g_attention": "#E74C3C",
        "g2s_attention": "#E74C3C",
        "spatial_attended1": "#3498DB",
        "graph_attended1": "#2ECC71",
        "s2g_attention2": "#E74C3C",
        "g2s_attention2": "#E74C3C",
        "spatial_attended2": "#3498DB",
        "graph_attended2": "#2ECC71",
        "pool_spatial": "#9B59B6",
        "pool_graph": "#9B59B6",
        "fusion": "#9B59B6",
        "output": "#F39C12",
    }

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    for node, data in nodes.items():
        node_x.append(data["x"])
        node_y.append(data["y"])
        node_text.append(data["label"])
        node_size.append(data["size"])
        node_color.append(node_colors[node])

    # Add nodes to figure
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                color=node_color, size=node_size, line=dict(width=2, color="black")
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(color="white", size=11),
            hoverinfo="text",
            name="Nodes",
        )
    )

    # Define edges
    edges = [
        ("spatial_features", "s2g_attention"),
        ("graph_features", "s2g_attention"),
        ("spatial_features", "g2s_attention"),
        ("graph_features", "g2s_attention"),
        ("s2g_attention", "spatial_attended1"),
        ("g2s_attention", "graph_attended1"),
        ("spatial_attended1", "s2g_attention2"),
        ("graph_attended1", "s2g_attention2"),
        ("spatial_attended1", "g2s_attention2"),
        ("graph_attended1", "g2s_attention2"),
        ("s2g_attention2", "spatial_attended2"),
        ("g2s_attention2", "graph_attended2"),
        ("spatial_attended2", "pool_spatial"),
        ("graph_attended2", "pool_graph"),
        ("pool_spatial", "fusion"),
        ("pool_graph", "fusion"),
        ("fusion", "output"),
    ]

    # Add edges to figure
    for edge in edges:
        source, target = edge
        fig.add_trace(
            go.Scatter(
                x=[nodes[source]["x"], nodes[target]["x"]],
                y=[nodes[source]["y"], nodes[target]["y"]],
                mode="lines",
                line=dict(width=2, color="gray"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Update layout
    fig.update_layout(
        title={
            "text": "Cross-View Attention Integration",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=18),
        },
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1000,
        height=600,
        plot_bgcolor="white",
    )

    # Add annotations to explain the process
    annotations = [
        dict(
            x=2.5,
            y=9,
            text="1. Initial features from spatial and spectral pathways",
            showarrow=False,
            font=dict(size=12, color="black"),
        ),
        dict(
            x=5.5,
            y=9,
            text="2. First cross-attention layer allows each pathway<br>to attend to features from the other",
            showarrow=False,
            font=dict(size=12, color="black"),
        ),
        dict(
            x=11.5,
            y=9,
            text="3. Second cross-attention layer deepens<br>the integration between pathways",
            showarrow=False,
            font=dict(size=12, color="black"),
        ),
        dict(
            x=17.5,
            y=9,
            text="4. Attention-based pooling and fusion<br>create the final representation",
            showarrow=False,
            font=dict(size=12, color="black"),
        ),
    ]

    for annotation in annotations:
        fig.add_annotation(annotation)

    return fig


# Create all visualizations
full_model_fig = create_mvt_full_model()
spatial_extractor_fig = create_spatial_extractor_visualization()
graph_module_fig = create_multiscale_graph_visualization()
cross_attention_fig = create_cross_attention_visualization()

# Save the figures
full_model_fig.savefig("mvt_full_model.png", dpi=300, bbox_inches="tight")
spatial_extractor_fig.savefig("spatial_extractor.png", dpi=300, bbox_inches="tight")
graph_module_fig.savefig("multiscale_graph_module.png", dpi=300, bbox_inches="tight")
cross_attention_fig.write_html("cross_attention_interactive.html")

print("Visualizations created successfully!")
