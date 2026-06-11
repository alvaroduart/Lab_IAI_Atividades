import numpy as np
import matplotlib.pyplot as plt

class LVQ1:
    def __init__(self, input_dim=6, num_classes=4, prototypes_per_class=1, learning_rate=0.05):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.proto_per_class = prototypes_per_class
        self.alpha_init = learning_rate
        self.alpha = learning_rate
        
        # Total number of prototype vectors
        self.num_prototypes = num_classes * prototypes_per_class
        self.W = np.zeros((self.num_prototypes, self.input_dim))
        
        # Class labels associated with each prototype vector (1-indexed: 1, 2, 3, 4)
        self.proto_classes = np.repeat(np.arange(1, num_classes + 1), prototypes_per_class)
        
    def initialize_prototypes(self, X_train, y_train, method='first'):
        """
        Initialize the prototype vectors.
        method: 
          'first' - take the first sample(s) belonging to each class.
          'mean'  - take the mean of the training samples of each class.
        """
        for i, cls in enumerate(range(1, self.num_classes + 1)):
            cls_samples = X_train[y_train == cls]
            if method == 'first':
                for p in range(self.proto_per_class):
                    self.W[i * self.proto_per_class + p] = cls_samples[p]
            elif method == 'mean':
                cls_mean = np.mean(cls_samples, axis=0)
                for p in range(self.proto_per_class):
                    # Add a tiny noise if we have multiple prototypes per class to differentiate them
                    noise = np.random.normal(0, 0.001, self.input_dim) if p > 0 else 0.0
                    self.W[i * self.proto_per_class + p] = cls_mean + noise
            else:
                raise ValueError("Initialization method must be 'first' or 'mean'")
                
    def get_winner(self, x):
        """
        Find the index of the closest prototype (BMU) to input vector x.
        """
        distances = np.linalg.norm(self.W - x, axis=1)
        return np.argmin(distances)
        
    def predict(self, x):
        """
        Predict the class label for input vector x.
        """
        winner_idx = self.get_winner(x)
        return self.proto_classes[winner_idx]
        
    def train_step(self, x, y, alpha):
        """
        Execute a single LVQ-1 training step for a sample x and its true label y.
        """
        winner_idx = self.get_winner(x)
        winner_class = self.proto_classes[winner_idx]
        
        # LVQ-1 weight update rule
        if winner_class == y:
            # Correct class: pull prototype closer
            self.W[winner_idx] += alpha * (x - self.W[winner_idx])
        else:
            # Incorrect class: push prototype away
            self.W[winner_idx] -= alpha * (x - self.W[winner_idx])
            
    def train(self, X, y, epochs=1000, decay='linear', shuffle=True, seed=None):
        """
        Train the LVQ-1 network.
        X: numpy array of shape (N, input_dim)
        y: numpy array of shape (N,) containing class labels
        epochs: number of epochs
        decay: 'linear' or 'none'
        shuffle: whether to shuffle the dataset at each epoch
        seed: random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        N = len(X)
        history_weights = [np.copy(self.W)]
        
        for epoch in range(epochs):
            if decay == 'linear':
                current_alpha = self.alpha_init * (1.0 - (epoch / epochs))
            else:
                current_alpha = self.alpha_init
                
            self.alpha = current_alpha
            
            if shuffle:
                indices = np.random.permutation(N)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
                
            for x, label in zip(X_shuffled, y_shuffled):
                self.train_step(x, label, current_alpha)
                
            history_weights.append(np.copy(self.W))
            
        return history_weights

# ----------------- Datasets -----------------

# Training dataset: 16 samples, 6 inputs (7h to 12h), 1 class label (1 to 4)
TRAINING_DATA_RAW = np.array([
    # Class 1
    [2.3976, 1.5328, 1.9044, 1.1937, 2.4184, 1.8649, 1],
    [2.3936, 1.4804, 1.9907, 1.2732, 2.2719, 1.8110, 1],
    [2.2880, 1.4585, 1.9867, 1.2451, 2.3389, 1.8099, 1],
    [2.2904, 1.4766, 1.8876, 1.2706, 2.2966, 1.7744, 1],
    # Class 2
    [1.1201, 0.0587, 1.3154, 5.3783, 3.1849, 2.4276, 2],
    [0.9913, 0.1524, 1.2700, 5.3808, 3.0714, 2.3331, 2],
    [1.0915, 0.1881, 1.1387, 5.3701, 3.2561, 2.3383, 2],
    [1.0535, 0.1229, 1.2743, 5.3226, 3.0950, 2.3193, 2],
    # Class 3
    [1.4871, 2.3448, 0.9918, 2.3160, 1.6783, 5.0850, 3],
    [1.3312, 2.2553, 0.9618, 2.4702, 1.7272, 5.0645, 3],
    [1.3646, 2.2945, 1.0562, 2.4763, 1.8051, 5.1470, 3],
    [1.4392, 2.2296, 1.1278, 2.4230, 1.7259, 5.0876, 3],
    # Class 4
    [2.9364, 1.5233, 4.6109, 1.3160, 4.2700, 6.8749, 4],
    [2.9034, 1.4640, 4.6061, 1.4598, 4.2912, 6.9142, 4],
    [3.0181, 1.4918, 4.7051, 1.3521, 4.2623, 6.7966, 4],
    [2.9374, 1.4896, 4.7219, 1.3977, 4.1863, 6.8336, 4]
])

X_train = TRAINING_DATA_RAW[:, :6]
y_train = TRAINING_DATA_RAW[:, 6].astype(int)

# Validation/Test dataset: 8 samples, 6 inputs (7h to 12h)
X_test = np.array([
    [2.9817, 1.5656, 4.8391, 1.4311, 4.1916, 6.9718], # Day 1
    [1.5537, 2.2615, 1.3169, 2.5873, 1.7570, 5.0958], # Day 2
    [1.2240, 0.2445, 1.3595, 5.4192, 3.2027, 2.5675], # Day 3
    [2.5828, 1.5146, 2.1119, 1.2859, 2.3414, 1.8695], # Day 4
    [2.4168, 1.4857, 1.8959, 1.3013, 2.4500, 1.7868], # Day 5
    [1.0604, 0.2276, 1.2806, 5.4732, 3.2133, 2.4839], # Day 6
    [1.5246, 2.4254, 1.1353, 2.5325, 1.7569, 5.2640], # Day 7
    [3.0565, 1.6259, 4.7743, 1.3654, 4.2904, 6.9808]  # Day 8
])

# ----------------- Plotting Helpers -----------------

def plot_demand_profiles(X, y, prototypes, title, filename=None):
    """
    Plot the daily electricity demand profiles for each class alongside the learned prototypes.
    """
    hours = np.array([7, 8, 9, 10, 11, 12])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Harmonious palette (Blue, Orange, Green, Red)
    classes = [1, 2, 3, 4]
    
    plt.figure(figsize=(12, 8), dpi=150)
    
    # Set Outfit or Inter font style if possible, else standard sans-serif
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Plot training profiles as thin/semi-transparent lines
    for i in range(len(X)):
        cls = y[i]
        plt.plot(hours, X[i], color=colors[cls-1], alpha=0.3, linestyle='--', label=f'Treino Classe {cls}' if i in [0, 4, 8, 12] else "")
        
    # Plot prototypes as thick solid lines with markers
    for idx, proto in enumerate(prototypes):
        cls = idx + 1
        plt.plot(hours, proto, color=colors[idx], linewidth=4.0, marker='o', markersize=8, 
                 label=f'Protótipo Classe {cls}')
        
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Hora do Dia (h)', fontsize=12, labelpad=10)
    plt.ylabel('Potência Elétrica Medida (MW)', fontsize=12, labelpad=10)
    plt.xticks(hours, [f"{h}h" for h in hours])
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Position legend cleanly
    plt.legend(loc='upper left', frameon=True, shadow=True, borderpad=1)
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    plt.close()

if __name__ == '__main__':
    print("=" * 60)
    print("Simulação da Rede LVQ-1 - Atividade 10")
    print("=" * 60)
    
    # Scenario 1: Prototype initialization using the first sample of each class
    print("\nCenário 1: Inicialização com a PRIMEIRA AMOSTRA de cada classe")
    lvq_first = LVQ1(input_dim=6, num_classes=4, prototypes_per_class=1, learning_rate=0.05)
    lvq_first.initialize_prototypes(X_train, y_train, method='first')
    print("Pesos iniciais (Protótipos):")
    for idx, w in enumerate(lvq_first.W):
        print(f"  Classe {idx+1}: {np.round(w, 4)}")
        
    # Train
    epochs = 1000
    lvq_first.train(X_train, y_train, epochs=epochs, decay='linear', shuffle=True, seed=42)
    print("\nPesos finais após treinamento (1000 épocas, decaimento linear da taxa):")
    for idx, w in enumerate(lvq_first.W):
        print(f"  Classe {idx+1}: {np.round(w, 4)}")
        
    # Test Classifications
    print("\nClassificação das amostras de teste (Cenário 1):")
    predictions_first = []
    for d_idx, x in enumerate(X_test):
        pred_cls = lvq_first.predict(x)
        predictions_first.append(pred_cls)
        winner_idx = lvq_first.get_winner(x)
        dist = np.linalg.norm(lvq_first.W[winner_idx] - x)
        print(f"  Dia {d_idx+1}: {np.round(x, 4)} -> Protótipo Classe {pred_cls} (Distância: {dist:.4f})")
        
    # Plot Scenario 1
    plot_demand_profiles(X_train, y_train, lvq_first.W, 
                         'Perfil de Demanda de Potência - LVQ-1 (Inicialização: Primeira Amostra)',
                         'perfis_demanda_inicializacao_primeira.png')
                         
    # Scenario 2: Prototype initialization using the mean of training samples of each class
    print("\n" + "=" * 60)
    print("Cenário 2: Inicialização com a MÉDIA das amostras de cada classe")
    lvq_mean = LVQ1(input_dim=6, num_classes=4, prototypes_per_class=1, learning_rate=0.05)
    lvq_mean.initialize_prototypes(X_train, y_train, method='mean')
    print("Pesos iniciais (Protótipos):")
    for idx, w in enumerate(lvq_mean.W):
        print(f"  Classe {idx+1}: {np.round(w, 4)}")
        
    # Train
    lvq_mean.train(X_train, y_train, epochs=epochs, decay='linear', shuffle=True, seed=42)
    print("\nPesos finais após treinamento (1000 épocas, decaimento linear da taxa):")
    for idx, w in enumerate(lvq_mean.W):
        print(f"  Classe {idx+1}: {np.round(w, 4)}")
        
    # Test Classifications
    print("\nClassificação das amostras de teste (Cenário 2):")
    predictions_mean = []
    for d_idx, x in enumerate(X_test):
        pred_cls = lvq_mean.predict(x)
        predictions_mean.append(pred_cls)
        winner_idx = lvq_mean.get_winner(x)
        dist = np.linalg.norm(lvq_mean.W[winner_idx] - x)
        print(f"  Dia {d_idx+1}: {np.round(x, 4)} -> Protótipo Classe {pred_cls} (Distância: {dist:.4f})")
        
    # Plot Scenario 2
    plot_demand_profiles(X_train, y_train, lvq_mean.W, 
                         'Perfil de Demanda de Potência - LVQ-1 (Inicialização: Média)',
                         'perfis_demanda_inicializacao_media.png')
                         
    # Save a comparison table of classifications
    print("\nComparativo Final de Classificações:")
    print("Dia |  Primeira Amostra | Média das Amostras | Coerente?")
    for d_idx in range(8):
        c1 = predictions_first[d_idx]
        c2 = predictions_mean[d_idx]
        print(f" {d_idx+1}  |      Classe {c1}     |      Classe {c2}     | {'Sim' if c1 == c2 else 'Não'}")
