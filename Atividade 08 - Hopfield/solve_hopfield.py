import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons=45):
        self.N = num_neurons
        self.W = np.zeros((self.N, self.N))
    
    def train(self, patterns, zero_diagonal=True):
        """
        Train the network on the given patterns using the outer product rule (Hebbian learning).
        patterns: List of patterns, each pattern is a 1D numpy array of size self.N with values in {-1, 1}.
        zero_diagonal: If True, set self-connections W[i, i] = 0.
        """
        patterns = np.array(patterns)
        num_patterns = len(patterns)
        
        # Outer product rule: W = (1/N) * sum(xi * xi^T)
        self.W = np.dot(patterns.T, patterns) / self.N
        
        if zero_diagonal:
            np.fill_diagonal(self.W, 0.0)
            
    def energy(self, state):
        """
        Compute the energy of the given state.
        E = -0.5 * S^T * W * S
        """
        return -0.5 * np.dot(state, np.dot(self.W, state))
        
    def update(self, initial_state, beta=100.0, mode='asynchronous', max_iter=100, tolerance=1e-5):
        """
        Update the network state until convergence.
        mode: 'synchronous' or 'asynchronous'
        beta: parameter for the hyperbolic tangent activation function (large beta approximates sign function).
        """
        state = np.array(initial_state, dtype=float)
        
        if mode == 'synchronous':
            for _ in range(max_iter):
                prev_state = np.copy(state)
                # Synchronous update: S_new = tanh(beta * W * S)
                state = np.tanh(beta * np.dot(self.W, prev_state))
                # Map back to strictly near {-1, 1}
                state = np.where(state >= 0, 1.0, -1.0)
                
                if np.all(prev_state == state):
                    break
            return state
            
        elif mode == 'asynchronous':
            # Sequential or random update of neurons
            # Let's use random ordering per epoch to avoid cycles and match physical systems
            for iteration in range(max_iter):
                prev_state = np.copy(state)
                indices = np.random.permutation(self.N)
                for idx in indices:
                    activation = np.dot(self.W[idx], state)
                    state[idx] = np.tanh(beta * activation)
                    # Map to strictly {-1, 1} to keep discrete state representation
                    state[idx] = 1.0 if state[idx] >= 0 else -1.0
                
                if np.all(prev_state == state):
                    break
            return state
        else:
            raise ValueError("Mode must be 'synchronous' or 'asynchronous'")

# ----------------- Helper Functions -----------------

def add_noise(pattern, noise_level=0.20):
    """
    Randomly flip a percentage of elements in the pattern.
    """
    noisy_pattern = np.copy(pattern)
    n_flips = int(round(noise_level * len(pattern)))
    flip_indices = np.random.choice(len(pattern), n_flips, replace=False)
    for idx in flip_indices:
        noisy_pattern[idx] = -noisy_pattern[idx]
    return noisy_pattern

def pattern_to_grid(pattern, rows=9, cols=5, dark_char='#', light_char='.'):
    """
    Convert a 1D pattern of {-1, 1} to a human-readable 2D grid string.
    """
    grid = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            val = pattern[r * cols + c]
            row_chars.append(dark_char if val == 1 else light_char)
        grid.append(" ".join(row_chars))
    return "\n".join(grid)

# Define the 4 target patterns (digits 1, 2, 3, 4) in binary grid representation.
# 1 represents dark pixel (+1), 0 represents white pixel (-1).
PATTERNS_RAW = {
    1: [
        0, 0, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 0,
        0, 0, 1, 1, 0,
        0, 0, 1, 1, 0,
        0, 0, 1, 1, 0,
        0, 0, 1, 1, 0,
        0, 0, 1, 1, 0,
        0, 0, 1, 1, 0
    ],
    2: [
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ],
    3: [
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ],
    4: [
        1, 1, 0, 1, 1,
        1, 1, 0, 1, 1,
        1, 1, 0, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 1
    ]
}

# Convert binary raw values [0, 1] to Hopfield values [-1, 1]
PATTERNS = {k: np.array([1 if x == 1 else -1 for x in v]) for k, v in PATTERNS_RAW.items()}

if __name__ == '__main__':
    # Simple self-test to verify convergence and perfect recall
    print("Testing Hopfield Network Implementation...")
    network = HopfieldNetwork()
    train_patterns = [PATTERNS[1], PATTERNS[2], PATTERNS[3], PATTERNS[4]]
    network.train(train_patterns)
    
    print("Weight matrix shape:", network.W.shape)
    print("Symmetric:", np.allclose(network.W, network.W.T))
    print("Diagonal is zero:", np.allclose(np.diag(network.W), 0.0))
    
    # Test perfect recall on clean patterns
    success = True
    for key, pattern in PATTERNS.items():
        recovered = network.update(pattern, mode='asynchronous')
        if not np.array_equal(recovered, pattern):
            print(f"Failed perfect recall for pattern {key}!")
            success = False
        else:
            print(f"Pattern {key} recalled perfectly (0% noise)")
            
    # Test with 20% noise
    print("\nTesting recall with 20% noise:")
    np.random.seed(42) # set seed for reproducibility of test
    for key, pattern in PATTERNS.items():
        noisy = add_noise(pattern, 0.20)
        recovered = network.update(noisy, mode='asynchronous')
        match = np.array_equal(recovered, pattern)
        print(f"Pattern {key} with 20% noise -> Recalled matches original: {match}")
        if match:
            print("Noisy pattern:")
            print(pattern_to_grid(noisy))
            print("Recovered pattern:")
            print(pattern_to_grid(recovered))
            print("-" * 20)
