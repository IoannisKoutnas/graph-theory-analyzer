# Author: IOANNIS KOUTNAS
# Graph Theory Visualizer and Analyzer using Tkinter and NetworkX
# Features: BFS, DFS, greedy coloring, cycle and clique detection, animated traversals

import tkinter as tk                          # GUI window and widgets
from tkinter import scrolledtext              # Scrollable text output widget
import networkx as nx                         # Graph representation and algorithms
import matplotlib
matplotlib.use("TkAgg")                       # Use Tkinter-compatible backend for matplotlib
import matplotlib.pyplot as plt               # Plotting static/animated graph
from matplotlib.animation import FuncAnimation# Graph animation for traversals
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading                             # For animation without freezing GUI
from collections import deque                 # BFS queue
import matplotlib.colors as mcolors           # Flexible color palettes

def bfs_with_depth(G, start=0):
    """Breadth-First Search traversal. Returns list of (node, depth) in visit order."""
    visited = set([start])
    queue = deque([(start, 0)])
    order = []
    while queue:
        node, depth = queue.popleft()
        order.append((node, depth))
        for neighbor in sorted(G.neighbors(node)):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return order

def dfs_with_depth(G, start=0):
    """Recursive Depth-First Search traversal. Returns list of (node, depth) in visit order."""
    visited = set()
    order = []
    def dfs_rec(node, depth):
        visited.add(node)
        order.append((node, depth))
        for neighbor in sorted(G.neighbors(node)):
            if neighbor not in visited:
                dfs_rec(neighbor, depth + 1)
    dfs_rec(start, 0)
    return order

def has_cycle(G):
    """
    Detects cycles in an undirected graph using DFS and recursion stack.
    Returns True if any cycle exists, else False.
    """
    visited = set()
    rec_stack = set()
    def dfs_cycle(v, parent):
        visited.add(v)
        rec_stack.add(v)
        for neighbor in G.neighbors(v):
            if neighbor not in visited:
                if dfs_cycle(neighbor, v):
                    return True
            elif neighbor in rec_stack and neighbor != parent:
                return True
        rec_stack.remove(v)
        return False
    for n in G.nodes:
        if n not in visited:
            if dfs_cycle(n, None):
                return True
    return False

def greedy_coloring(G):
    """
    Applies greedy coloring algorithm—assigns the smallest possible color to each node
    such that no adjacent nodes share the same color. Returns node: color mapping.
    """
    result = dict()
    for node in sorted(G.nodes()):
        neighbor_colors = set(result.get(neigh) for neigh in G.neighbors(node))
        color = 0
        while color in neighbor_colors:
            color += 1
        result[node] = color
    return result

def find_max_clique(G):
    """
    Finds the largest clique (a maximal, fully connected node subset).
    Returns the node indices of the clique.
    """
    cliques = list(nx.find_cliques(G))
    max_clique = max(cliques, key=len)
    return max_clique

class GraphApp:
    """
    GUI application for graph theory analysis and demonstration on the karate club graph.
    Provides animated traversal, graph properties, and network coloring.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Theory Analysis – Karate Club")
        self.G = nx.karate_club_graph()            # Loads classic social network graph data (34 nodes)
        self.pos = nx.spring_layout(self.G, seed=42) # Consistent layout for visualization
        self.anim = None
        self.anim_thread = None
        self.current_coloring = None
        self.clique_nodes = None

        root.geometry("1300x900")

        # Split GUI into left (graph area) and right (controls and log)
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame = tk.Frame(self.root, width=350, padx=8)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Graph drawing area using matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, left_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        bold_font = ('Helvetica', 11, 'bold')
        tk.Label(right_frame, text="Choose action:", font=('Helvetica', 14, 'bold')).pack(pady=12)
        
        # Action buttons
        tk.Button(right_frame, text="Run BFS", command=self.run_bfs, height=2, width=18, font=bold_font).pack(pady=6)
        tk.Button(right_frame, text="Run DFS", command=self.run_dfs, height=2, width=18, font=bold_font).pack(pady=6)
        tk.Button(right_frame, text="Run Greedy Coloring", command=self.run_coloring, height=2, width=18, font=bold_font).pack(pady=6)
        tk.Button(right_frame, text="Check Cycle", command=self.check_cycle, height=2, width=18, font=bold_font).pack(pady=6)
        tk.Button(right_frame, text="Check Clique", command=self.check_clique, height=2, width=18, font=bold_font).pack(pady=6)
        tk.Button(right_frame, text="Clear Log", command=self.clear_log, height=2, width=18, font=bold_font).pack(pady=6)
        tk.Button(right_frame, text="Exit", command=self.exit_app, height=2, width=18, font=bold_font).pack(pady=16)

        # Log output window
        tk.Label(right_frame, text="Log Output:", font=('Helvetica', 12)).pack(pady=(18,2))
        self.log = scrolledtext.ScrolledText(right_frame, width=45, height=21, font=('Consolas', 11))
        self.log.pack(pady=(1, 10))

        self.node_colors = ["lightblue"] * 34      # Default node coloring
        self.log_msg(f"Nodes: 34, Edges: 78\n")
        self.draw_graph()

    def log_msg(self, msg):
        """Append the message to the scrollable log output area."""
        self.log.insert(tk.END, msg + '\n')
        self.log.see(tk.END)

    def clear_log(self):
        """
        Reset log output and visualization to the default uncolored state.
        Clears any coloring or clique highlights.
        """
        self.log.delete('1.0', tk.END)
        self.current_coloring = None
        self.node_colors = ["lightblue"] * 34
        self.clique_nodes = None
        self.draw_graph()

    def exit_app(self):
        """Gracefully exit the application."""
        self.root.quit()
        self.root.destroy()

    def get_color_map(self, num_colors):
        """
        Generate a flexible, visually appealing color palette of length num_colors.
        Uses tableau and CSS color lists for variety and clarity.
        """
        base_colors = list(mcolors.TABLEAU_COLORS.values())
        if num_colors <= len(base_colors):
            return base_colors[:num_colors]
        extended = base_colors + list(mcolors.CSS4_COLORS.values())
        return [extended[i % len(extended)] for i in range(num_colors)]

    def draw_graph(self, highlight_nodes=None, title=""):
        """
        Draw the current state of the graph.
        Allows overriding node colors to highlight traversals or cliques.
        """
        self.ax.clear()
        self.ax.axis("off")
        node_colors = self.node_colors.copy()
        if self.current_coloring:
            max_color = max(self.current_coloring.values())
            color_map = self.get_color_map(max_color + 1)
            for node, color_idx in self.current_coloring.items():
                node_colors[node] = color_map[color_idx]
        if self.clique_nodes:
            for n in self.clique_nodes:
                node_colors[n] = "red"
        if highlight_nodes:
            for n in highlight_nodes.get('gold', []):
                node_colors[n] = "gold"
            for n in highlight_nodes.get('lightgreen', []):
                node_colors[n] = "lightgreen"
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, alpha=0.6, width=2)
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, ax=self.ax, node_size=800)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax, font_size=16)
        if title:
            self.ax.set_title(title, fontsize=20, pad=10)
        self.fig.tight_layout(pad=0)
        self.canvas.draw_idle()

    def run_bfs(self):
        """Execute BFS traversal; animate visiting order and depths."""
        self.current_coloring = None
        self.clique_nodes = None
        order = bfs_with_depth(self.G, start=0)
        self.log_msg("=== BFS ===")
        nodes_only = [node for node, depth in order]
        self.log_msg(f"BFS order: {nodes_only} ({len(nodes_only)} nodes)")
        self.animate_traversal(order, mode='BFS')

    def run_dfs(self):
        """Execute DFS traversal; animate visiting order and depths."""
        self.current_coloring = None
        self.clique_nodes = None
        order = dfs_with_depth(self.G, start=0)
        self.log_msg("=== DFS ===")
        nodes_only = [node for node, depth in order]
        self.log_msg(f"DFS order: {nodes_only} ({len(nodes_only)} nodes)")
        self.animate_traversal(order, mode='DFS')

    def animate_traversal(self, order, mode='BFS'):
        """
        Animate traversal steps for BFS/DFS: highlights the current and previous nodes at each step.
        Ensures only one animation runs at a time to avoid GUI conflicts.
        """
        if self.anim_thread and self.anim_thread.is_alive():
            self.log_msg("Animation is already running. Wait until it finishes.")
            return

        self.node_colors = ["lightblue"] * 34
        self.clique_nodes = None

        def worker():
            step = {'gold': [], 'lightgreen': []}
            def update(frame):
                if frame > 0:
                    node, depth = order[frame - 1]
                    step['gold'].append(node)
                    if frame > 1:
                        prev_node, _ = order[frame - 2]
                        step['lightgreen'].append(prev_node)
                    title = f"{mode} – Step {frame}: Node {node} (Depth {depth})"
                else:
                    title = mode
                self.draw_graph(highlight_nodes=step, title=title)
                if frame == len(order):
                    for n, _ in order:
                        step['lightgreen'].append(n)
                    self.draw_graph(highlight_nodes=step, title=f"{mode} Completed")
                    return
            self.anim = FuncAnimation(self.fig, update, frames=len(order) + 1,
                                      interval=700, repeat=False)
            self.canvas.draw_idle()
            self.anim._start()

        self.anim_thread = threading.Thread(target=worker, daemon=True)
        self.anim_thread.start()

    def run_coloring(self):
        """Apply greedy coloring and visualize the color assignment for all nodes."""
        if self.anim_thread and self.anim_thread.is_alive():
            self.log_msg("Wait for any running animation to finish before coloring.")
            return
        self.current_coloring = greedy_coloring(self.G)
        self.clique_nodes = None
        max_color = max(self.current_coloring.values())
        self.log_msg("=== Greedy Coloring ===")
        self.log_msg(f"Colors used: {max_color + 1}")
        for node in sorted(self.current_coloring.keys()):
            self.log_msg(f"Node {node}: Color {self.current_coloring[node]}")
        self.node_colors = ["lightblue"] * 34
        self.draw_graph(title="Greedy Coloring Result")

    def check_cycle(self):
        """Check and log whether a cycle is present in the graph."""
        self.clique_nodes = None
        result = has_cycle(self.G)
        self.log_msg("=== Cycle Check ===")
        self.log_msg(f"Graph has cycle: {result}")

    def check_clique(self):
        """Find, log and highlight the largest clique (completely connected subgroup)."""
        self.current_coloring = None
        max_clique = find_max_clique(self.G)
        self.clique_nodes = max_clique
        self.log_msg("=== Clique Check ===")
        self.log_msg(f"Maximum clique size: {len(max_clique)}")
        self.log_msg(f"Clique nodes: {max_clique}")
        self.draw_graph(title="Maximum Clique Highlighted")

if __name__ == "__main__":
    # Launch the Tkinter application and print basic project info to log.
    root = tk.Tk()
    app = GraphApp(root)
    app.log_msg(f"Nodes: 34, Edges: 78\n")
    app.log_msg(
        "\n• Graph Theory Analysis | Python\n"
        "Demonstrate graph traversals, coloring, cycles, and clique detection with animation and logging.\n"
        "Adaptable for teaching, projects, or extending to new graphs and datasets."
    )
    root.mainloop()
