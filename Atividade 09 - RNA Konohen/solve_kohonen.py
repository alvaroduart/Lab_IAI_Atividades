import numpy as np

class KohonenSOM:
    def __init__(self, grid_shape=(4, 4), input_dim=3, learning_rate=0.001, radius=1):
        self.grid_shape = grid_shape
        self.N = grid_shape[0] * grid_shape[1] # 16 neurons
        self.input_dim = input_dim
        self.eta = learning_rate
        self.radius = radius
        
        # Initialize weights randomly between 0 and 1
        # To ensure reproducible results for the lab, we can seed or initialize around the data range
        self.W = np.random.uniform(0.0, 1.0, (self.N, self.input_dim))
        
        # Create coordinates for each neuron in the 2D grid
        self.coords = np.array([(r, c) for r in range(grid_shape[0]) for c in range(grid_shape[1])])
        
    def get_winner(self, x):
        """
        Find the winning neuron index (closest to input x in Euclidean distance).
        """
        distances = np.linalg.norm(self.W - x, axis=1)
        return np.argmin(distances)
        
    def get_neighbors(self, winner_idx):
        """
        Find indices of all neurons within Chebyshev distance <= self.radius of the winner.
        This forms a (2*radius + 1) square centered at the winner.
        """
        winner_coord = self.coords[winner_idx]
        # Chebyshev distance: max(|r1-r2|, |c1-c2|)
        distances = np.max(np.abs(self.coords - winner_coord), axis=1)
        neighbor_indices = np.where(distances <= self.radius)[0]
        return neighbor_indices

    def train_step(self, x):
        """
        Run a single training step: find winner, find neighbors, and update weights.
        """
        winner_idx = self.get_winner(x)
        neighbors = self.get_neighbors(winner_idx)
        
        # Update weights: W = W + eta * (x - W)
        for idx in neighbors:
            self.W[idx] += self.eta * (x - self.W[idx])
            
    def train(self, data, epochs=1000, shuffle=True, seed=None):
        """
        Train the SOM on the dataset for the specified number of epochs.
        """
        if seed is not None:
            np.random.seed(seed)
            
        data = np.array(data)
        
        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(len(data))
                shuffled_data = data[indices]
            else:
                shuffled_data = data
                
            for x in shuffled_data:
                self.train_step(x)

# ----------------- Helper Functions -----------------

def get_neuron_class_mapping(som, train_data):
    """
    Classify each training sample and record which neuron wins.
    Training samples 1-20 (indices 0-19) -> Class A
    Training samples 21-60 (indices 20-59) -> Class B
    Training samples 61-120 (indices 60-119) -> Class C
    
    Returns a dictionary mapping neuron indices to (Class, count, distribution_array).
    """
    # 16 neurons, 3 classes (A, B, C)
    counts = np.zeros((som.N, 3), dtype=int)
    
    for idx, x in enumerate(train_data):
        winner = som.get_winner(x)
        if idx < 20:
            counts[winner, 0] += 1 # Class A
        elif idx < 60:
            counts[winner, 1] += 1 # Class B
        else:
            counts[winner, 2] += 1 # Class C
            
    mapping = {}
    classes = ['A', 'B', 'C']
    for neuron in range(som.N):
        neuron_distribution = counts[neuron]
        total_wins = np.sum(neuron_distribution)
        if total_wins == 0:
            mapping[neuron] = ('Unused', 0, neuron_distribution)
        else:
            best_class_idx = np.argmax(neuron_distribution)
            mapping[neuron] = (classes[best_class_idx], total_wins, neuron_distribution)
            
    return mapping

def print_grid_map(mapping, shape=(4, 4)):
    """
    Prints a 4x4 topological grid map showing which neurons represent which classes.
    """
    grid = []
    for r in range(shape[0]):
        row_str = []
        for c in range(shape[1]):
            neuron_idx = r * shape[1] + c
            cls, wins, dist = mapping[neuron_idx]
            # Print Neuron Index (1-indexed) and Class label
            # Index is 1-16 to match the document layout
            row_str.append(f"{neuron_idx+1:02d}:{cls}")
        grid.append("  |  ".join(row_str))
    print("\n".join(grid))
