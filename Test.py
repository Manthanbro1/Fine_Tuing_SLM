# import json, random
# for lvl in ["Level_1.jsonl","Level_2.jsonl","Level_3.jsonl"]:
#     p = r"E:\College\2nd Year\Sem 1\EDAI\Project\Data\resume\Curriculum\{}".format(lvl)
#     with open(p,'r',encoding='utf-8') as f:
#         lines = f.readlines()
#     print("==", lvl, "sample ==")
#     print(json.loads(random.choice(lines)))
import matplotlib.pyplot as plt
import networkx as nx
import random

# ----------------- AVL Tree Helpers -----------------
def create_node(key):
    return {"key": key, "left": None, "right": None, "height": 1}

def get_height(node):
    return 0 if node is None else node["height"]

def get_balance(node):
    if node is None:
        return 0
    return get_height(node["left"]) - get_height(node["right"])

# ----------------- Rotations -----------------
def right_rotate(z):
    y = z["left"]
    T3 = y["right"]

    y["right"] = z
    z["left"] = T3

    z["height"] = 1 + max(get_height(z["left"]), get_height(z["right"]))
    y["height"] = 1 + max(get_height(y["left"]), get_height(y["right"]))

    return y

def left_rotate(z):
    y = z["right"]
    T2 = y["left"]

    y["left"] = z
    z["right"] = T2

    z["height"] = 1 + max(get_height(z["left"]), get_height(z["right"]))
    y["height"] = 1 + max(get_height(y["left"]), get_height(y["right"]))

    return y

# ----------------- Insertion -----------------
def insert(root, key):
    # Normal BST insertion
    if root is None:
        return create_node(key)
    elif key < root["key"]:
        root["left"] = insert(root["left"], key)
    else:
        root["right"] = insert(root["right"], key)

    # Update height
    root["height"] = 1 + max(get_height(root["left"]), get_height(root["right"]))

    # Get balance factor
    balance = get_balance(root)

    # Perform rotations
    # Left Left
    if balance > 1 and key < root["left"]["key"]:
        return right_rotate(root)
    # Right Right
    if balance < -1 and key > root["right"]["key"]:
        return left_rotate(root)
    # Left Right
    if balance > 1 and key > root["left"]["key"]:
        root["left"] = left_rotate(root["left"])
        return right_rotate(root)
    # Right Left
    if balance < -1 and key < root["right"]["key"]:
        root["right"] = right_rotate(root["right"])
        return left_rotate(root)

    return root

# ----------------- Visualization -----------------
def add_edges(graph, node):
    if node is not None:
        graph.add_node(node["key"])
        if node["left"]:
            graph.add_edge(node["key"], node["left"]["key"])
            add_edges(graph, node["left"])
        if node["right"]:
            graph.add_edge(node["key"], node["right"]["key"])
            add_edges(graph, node["right"])

def hierarchy_pos(G, root=None, width=1., vert_gap=0.3, vert_loc=0, xcenter=0.5):
    if root is None or root not in G:
        return {}
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.3, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    if pos is None:
        pos = {}
    pos[root] = (xcenter, vert_loc)
    neighbors = list(G.neighbors(root))
    if len(neighbors) > 0:
        dx = width / len(neighbors)
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos = _hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                 vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                 pos=pos, parent=root)
    return pos

def visualize_avl(root, title="AVL Tree Visualization"):
    if root is None:
        return
    graph = nx.DiGraph()
    add_edges(graph, root)
    pos = hierarchy_pos(graph, root["key"])
    if not pos:  # Prevent empty graph error
        return
    nx.draw(graph, pos, with_labels=True, node_size=1500, node_color="lightgreen",
            font_size=10, font_weight="bold", arrows=False)
    plt.title(title)
    plt.show()

# ----------------- Driver Code -----------------
if __name__ == "__main__":
    root = None
    values = random.sample(range(1, 50), 7)
    print("Inserted values:", values)

    for v in values:
        root = insert(root, v)
        visualize_avl(root, f"AVL Tree after inserting {v}")
