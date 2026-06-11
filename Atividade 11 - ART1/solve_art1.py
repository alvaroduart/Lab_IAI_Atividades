import numpy as np
import matplotlib.pyplot as plt

class ART1:
    def __init__(self, input_dim=16, rho=0.5, L=2.0):
        self.input_dim = input_dim
        self.rho = rho
        self.L = L
        
        # Bottom-up weights: shape (num_classes, input_dim)
        self.W_bu = np.empty((0, input_dim))
        # Top-down weights: shape (num_classes, input_dim)
        self.W_td = np.empty((0, input_dim))
        
    def add_category(self, x):
        """
        Add a new category and initialize its weights using the input vector x.
        """
        # Top-down weights are initialized to the input vector itself (fast commitment)
        t_new = np.copy(x)
        # Bottom-up weights are initialized: b_i = L * x_i / (L - 1 + |x|)
        norm_x = np.sum(x)
        b_new = (self.L * x) / (self.L - 1.0 + norm_x)
        
        self.W_td = np.vstack([self.W_td, t_new])
        self.W_bu = np.vstack([self.W_bu, b_new])
        return len(self.W_td) - 1

    def compute_activations(self, x):
        """
        Compute F2 activations for all existing categories.
        """
        if len(self.W_bu) == 0:
            return np.array([])
        # y_j = sum_i b_ji * x_i
        return np.dot(self.W_bu, x)

    def train(self, X, max_epochs=100):
        """
        Train the ART1 network on the dataset X until no classifications change.
        Returns a list of final class labels for each pattern in X.
        """
        N = len(X)
        prev_classifications = np.full(N, -1)
        
        for epoch in range(max_epochs):
            classifications = np.full(N, -1)
            
            for idx, x in enumerate(X):
                norm_x = np.sum(x)
                # Activation of uncommitted unit: y_uncommitted = L * |x| / (L - 1 + input_dim)
                y_uncommitted = (self.L * norm_x) / (self.L - 1.0 + self.input_dim)
                
                num_classes = len(self.W_bu)
                # Compute activations of committed units
                activations = self.compute_activations(x)
                
                # Active classes mask for search (True means not inhibited)
                active = np.ones(num_classes, dtype=bool)
                
                winner_idx = -1
                
                while True:
                    candidate_idx = -1
                    candidate_act = -1.0
                    
                    for j in range(num_classes):
                        if active[j] and activations[j] > candidate_act:
                            candidate_act = activations[j]
                            candidate_idx = j
                            
                    # If candidate committed unit activation is higher than or equal to uncommitted unit's activation,
                    # we test it. Otherwise, the uncommitted unit wins.
                    if candidate_idx != -1 and candidate_act >= y_uncommitted:
                        # Test vigilance
                        t_j = self.W_td[candidate_idx]
                        intersection = np.logical_and(x, t_j).astype(float)
                        norm_intersection = np.sum(intersection)
                        
                        if norm_intersection / norm_x >= self.rho:
                            # Resonance achieved!
                            winner_idx = candidate_idx
                            # Update weights of winning class
                            self.W_td[winner_idx] = intersection
                            self.W_bu[winner_idx] = (self.L * intersection) / (self.L - 1.0 + norm_intersection)
                            break
                        else:
                            # Inhibit candidate
                            active[candidate_idx] = False
                    else:
                        # Uncommitted unit wins
                        winner_idx = self.add_category(x)
                        break
                
                classifications[idx] = winner_idx
                
            # Check if classifications changed from previous epoch
            if np.array_equal(classifications, prev_classifications):
                break
            prev_classifications = np.copy(classifications)
            
        return prev_classifications

# ----------------- Datasets -----------------
# 10 situations with 16 binary status variables each.
SITUATIONS = np.array([
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], # Situação 1
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0], # Situação 2
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], # Situação 3
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0], # Situação 4
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1], # Situação 5
    [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], # Situação 6
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0], # Situação 7
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], # Situação 8 (identica à 3)
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], # Situação 9
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1]  # Situação 10 (identica à 5)
])

# ----------------- Plotting Helper -----------------
def plot_clusters(X, labels, rho, filename):
    unique_labels = sorted(list(set(labels)))
    
    # Reorder rows so that elements of the same class are grouped together
    sorted_indices = []
    for l in unique_labels:
        sorted_indices.extend([i for i, val in enumerate(labels) if val == l])
        
    X_sorted = X[sorted_indices]
    labels_sorted = [labels[i] for i in sorted_indices]
    sit_names_sorted = [f"Situação {i+1} (Classe {labels[i] + 1})" for i in sorted_indices]
    
    plt.figure(figsize=(12, 7), dpi=150)
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Visual heatmap
    plt.imshow(X_sorted, cmap='Blues', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    
    # Grid lines to separate cells
    plt.gca().set_xticks(np.arange(-.5, 16, 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, len(X), 1), minor=True)
    plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Title and Labels
    plt.title(f"Agrupamento ART-1 - Vigilância ($\\rho$ = {rho})", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Variáveis de Status ($x_1 \\dots x_{16}$)", fontsize=12, labelpad=10)
    plt.ylabel("Situações", fontsize=12, labelpad=10)
    
    plt.xticks(np.arange(16), [f"x{i+1}" for i in range(16)])
    plt.yticks(np.arange(len(X)), sit_names_sorted)
    
    # Draw horizontal dashed red lines to visually demarcate class boundaries
    current_idx = 0
    for l in unique_labels:
        cnt = labels_sorted.count(l)
        if current_idx + cnt < len(X):
            plt.axhline(y=current_idx + cnt - 0.5, color='#d62728', linestyle='--', linewidth=2.0)
        current_idx += cnt
        
    # Add a color bar
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['0 (Inativo)', '1 (Ativo)'])
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {filename}")

if __name__ == '__main__':
    print("=" * 70)
    print("Rede Neural ART-1 - Atividade 11")
    print("=" * 70)
    
    vigilance_values = [0.5, 0.8, 0.9, 0.99]
    
    results = {}
    
    for rho in vigilance_values:
        print(f"\n--- Simulação com Vigilância rho = {rho} ---")
        art = ART1(input_dim=16, rho=rho, L=2.0)
        labels = art.train(SITUATIONS)
        
        num_classes = len(art.W_td)
        print(f"Quantidade de Classes Ativas: {num_classes}")
        
        # Group situations by class (1-indexed classes for user display)
        clusters = {}
        for idx, lbl in enumerate(labels):
            cls_name = f"Classe {lbl + 1}"
            if cls_name not in clusters:
                clusters[cls_name] = []
            clusters[cls_name].append(f"Situação {idx + 1}")
            
        for cls_name, sits in sorted(clusters.items()):
            print(f"  {cls_name}: {', '.join(sits)}")
            
        print("\nPesos Finais:")
        for j in range(num_classes):
            print(f"  Protótipo/Template (Top-down) da Classe {j + 1}:")
            print(f"    T_{j+1} = {art.W_td[j].astype(int)}")
            print(f"    B_{j+1} = {np.round(art.W_bu[j], 4)}")
            
        plot_filename = f"agrupamento_art1_rho_{str(rho).replace('.', '')}.png"
        plot_clusters(SITUATIONS, labels, rho, plot_filename)
        
        results[rho] = {
            'labels': labels,
            'num_classes': num_classes,
            'clusters': clusters,
            'W_td': art.W_td,
            'W_bu': art.W_bu
        }
        
    print("\n" + "=" * 70)
    print("Simulação concluída com sucesso!")
    print("=" * 70)
