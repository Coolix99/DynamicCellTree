import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class TreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM gates
        self.W_iou = nn.Linear(input_size, 3 * hidden_size)  # Input, output, update gates
        self.U_iou = nn.Linear(hidden_size, 3 * hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)  # Forget gate
        self.U_f = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, children_states):
        # Check if the node has children
        if children_states:
            h_children = torch.stack([h for h, c in children_states], dim=0)
            c_children = torch.stack([c for h, c in children_states], dim=0)
            
            # Compute forget gates for each child
            f_gates = F.sigmoid(self.W_f(x).unsqueeze(0) + self.U_f(h_children))
            c = torch.sum(f_gates * c_children, dim=0)
        else:
            # Leaf nodes (no children) have an initial cell state of zeros
            c = torch.zeros(self.hidden_size, device=x.device)

        # Compute input, output, and update gates
        iou = self.W_iou(x) + (self.U_iou(h_children).sum(dim=0) if children_states else 0)
        i, o, u = torch.chunk(iou, 3, dim=-1)

        # Update cell state and hidden state
        c = F.sigmoid(i) * torch.tanh(u) + c
        h = F.sigmoid(o) * torch.tanh(c)

        return h, c

class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = TreeLSTMCell(input_size, hidden_size)
        self.decision_layer = nn.Linear(hidden_size, 1)  # Binary decision (split/terminate)

    def forward(self, node, tree_data):
        decisions = {}

        # Get feature vector for the current node
        if 'split_time' in tree_data[node]:
            # Use repeat to match input size for consistency
            x = tree_data[node]['split_time'].repeat(self.cell.hidden_size)
        else:
            x = tree_data[node]['data']  # Leaf node

        # Recursive TreeLSTM computation for each child
        children_states = []
        for child in tree_data[node].get('children', []):
            child_decisions, child_state = self.forward(child, tree_data)
            children_states.append(child_state)
            decisions.update(child_decisions)  # Collect decisions from children
        
        # Compute the current node's hidden and cell states
        h, c = self.cell(x, children_states)

        # Apply decision layer
        decision = torch.sigmoid(self.decision_layer(h)).item()
        decisions[node] = 1 if decision > 0.5 else 0  # Store binary decision for current node

        # If decision is split (1), continue to child nodes; else terminate this branch
        if decisions[node] == 1:
            for child in tree_data[node].get('children', []):
                child_decisions, _ = self.forward(child, tree_data)
                decisions.update(child_decisions)

        return decisions, (h, c)

# Example usage
input_size = 10  # Assume data vectors and split times are size 10
hidden_size = 20
model = TreeLSTM(input_size, hidden_size)

tree_data = {
    0: {'split_time': torch.tensor([0.5]).repeat(input_size), 'children': [1, 2]},  # Root
    1: {'data': torch.randn(input_size)},  # Leaf
    2: {'split_time': torch.tensor([0.3]).repeat(input_size), 'children': [3, 4]},  # Internal node
    3: {'data': torch.randn(input_size)},  # Leaf
    4: {'data': torch.randn(input_size)}   # Leaf
}

# Perform the forward pass on the root node
decisions, _ = model(0, tree_data)
print("Decisions for each node:", decisions)
