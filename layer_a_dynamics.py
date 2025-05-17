import math
import numpy as np
import matplotlib.pyplot as plt

# --- MODEL HFM ---
def H_model_distribution(n, g):
    all_states = [np.array([int(x) for x in format(i, f'0{n}b')]) for i in range(2**n)]
    # in the state (0, 0, ..., 0) -> ms=0, in state (1, 0, ..., 0) -> ms=1, else =1 is the p(s=(0,0,0))=1
    energies = np.array([np.exp(-g * max(np.max(np.where(s == 1)[0] + 1) - 1, 0)) if np.any(s) else 1.0 for s in all_states])
    Z = np.sum(energies)
    probs = energies / Z
    return all_states, probs

def plot_HFM_distribution(n,g):
    states, probs = H_model_distribution(n=n, g=g)
    state_labels = [''.join(map(str, s)) for s in states]
    sorted_indices = np.argsort(probs)[::-1]
    plt.figure(figsize=(5, 3))
    plt.bar(range(len(states)), np.array(probs)[sorted_indices], color='purple')
    plt.title(f"P(s) from HFM with g = {g}")
    plt.xlabel("States")
    plt.ylabel("Probabilidad P(s)")
    plt.tight_layout()
    plt.show()

# --- EJEMPLO ---
plot_HFM_distribution(n=8, g=0.0)
plot_HFM_distribution(n=8, g=1.5)


func= "sigmoid"
# Sample from the model
def sample_from_HFM(n, g, T):
    """
    Sample states from HFM model and transform them to {-1,+1}
    
    Parameters:
    -----------
    n : int
        Number of neurons
    g : float
        HFM parameter
    T : int
        Number of time steps
    
    Returns:
    --------
    samples : ndarray (T, n)
        Matrix of states where each element is in {-1,+1}
    """
    states, probs = H_model_distribution(n, g)
    indices = np.random.choice(len(states), size=T, p=probs)
    samples = np.array([states[i] for i in indices])
    # Transform states from {0,1} to {-1,+1}
    samples = 2 * samples - 1
    assert np.all(np.isin(samples, [-1, 1])), "States must be -1 or +1"
    return samples

# Sigmoid Function.
def sigma(x):
    return 1 / (1 + np.exp(-x))

def activation(z, func="sigmoid"):
    """
    Activation function for the layer dynamics
    
    Parameters:
    -----------
    z : ndarray
        Input to the activation function
    func : str
        Type of activation function: "step", "sigmoid", "tanh", or "relu"
    """
    if func == "step":  # Perceptron
        return np.where(z >= 0, 1.0, -1.0)
    elif func == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif func == "tanh":
        return np.tanh(z)
    elif func == "relu":
        return np.maximum(0.0, z)
    else:
        raise ValueError("Activation function not recognized")


def awareness_layers(S, J=20, eps=0):
    """
    Compute layer dynamics with Gaussian weights
    
    Parameters:
    -----------
    S : ndarray (T, n)
        Input states matrix where each element is in {-1,+1}
    J : int
        Number of layers
    eps : float
        Learning rate
    
    Returns:
    --------
    A : ndarray (T, J)
        Activity of each layer over time
    """
    T, n = S.shape
    A = np.zeros((T, J))
    
    # Initialize weights from N(0,1)
    W = np.random.normal(0, 1, (J, n))  # Shape: (J, n) for J layers and n inputs
    
    # Verify states are in {-1,+1}
    assert np.all(np.isin(S, [-1, 1])), "Input states must be -1 or +1"
    
    for j in range(J):
        a_j = 0.0  # Initialize layer j activity
        for t in range(T):
            input_sum = np.dot(W[j], S[t])  # W[j] shape: (n,), S[t] shape: (n,)
            out_val = activation(input_sum, func="sigmoid")
            a_j = (1 - eps) * a_j + eps * out_val
            A[t, j] = a_j
    
    return A

def compute_layer_correlation(S, W, A, t):
    """
    Compute the statistical measure Y = sum_{j<j'} (a_j^t - a_j'^t)(σ(w_j⋅s_t) - σ(w_j'⋅s_t))
    
    Parameters:
    -----------
    S : ndarray (T, n)
        Input states matrix
    W : ndarray (J, n)
        Weight matrix
    A : ndarray (T, J)
        Activity matrix
    t : int
        Time step to compute the measure
        
    Returns:
    --------
    Y : float
        Statistical measure Y at time t
    """
    J = W.shape[0]
    Y = 0.0
    
    # Compute activations for all layers at time t
    activations = np.array([activation(np.dot(W[j], S[t])) for j in range(J)])
    
    # Sum over j < j' pairs
    for j in range(J):
        for jp in range(j + 1, J):  # This ensures j < j'
            # Activity difference
            da = A[t, j] - A[t, jp]
            # Activation difference
            dact = activations[j] - activations[jp]
            # Add to the sum
            Y += da * dact
    
    return Y

def plot_layer_correlations(S, W, A):
    """
    Plot the evolution of the statistical measure Y over time
    """
    T = S.shape[0]
    
    # Compute Y for all time steps
    Y_t = np.array([compute_layer_correlation(S, W, A, t) for t in range(T)])
    
    # Plot Y over time
    plt.figure(figsize=(10, 4))
    plt.plot(Y_t, label='Y(t)')
    plt.title('Statistical Measure Y Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Y(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return Y_t

if __name__ == "__main__":
    # Fixed parameters
    n = 10      # length of each state
    T = 100     # number of time steps
    J = 5      # number of layers
    eps = 0.5
    func = "sigmoid"

    # We'll explore 3 different values of g
    g_values = [0.0, 2, math.log(2.0)]

    # We'll store the mean time series for each g in a list
    mean_series_list = []

    for g in g_values:
        # 1) sample states S from HFM
        S = sample_from_HFM(n=n, g=g, T=T)
        # 2) compute awareness for J layers
        A = awareness_layers(S, J=J, eps=eps)
        
        # Get the weights used in awareness_layers
        W = np.random.normal(0, 1, (J, n))  # Need to use same distribution
        
        # 3) Compute and plot layer correlations
        Y_t = plot_layer_correlations(S, W, A)

        # 3) Plot each layer's time series (WITHOUT the mean)
        plt.figure()
        for j in range(J):
            plt.plot(A[:, j], label=f"Layer {j}")

        plt.title(f"A(t), no mean line, for g={g}\n"
                  f"(n={n}, T={T}, J={J}, func={func})")
        plt.xlabel("Time")
        plt.ylabel("a_j(t)")
        plt.tight_layout()
        plt.show()

        # Plot the states matrix with improved visualization
        plt.figure(figsize=(10, 4))
        plt.imshow(S.T, cmap='RdBu', aspect='auto', interpolation='none')
        plt.colorbar(label='State Value')
        plt.title(f'States Matrix for g={g}\n(Red: +1, Blue: -1)')
        plt.xlabel('Time Step (t)')
        plt.ylabel('Neuron Index')
        # Add grid
        plt.grid(False)
        plt.tight_layout()
        plt.show()

        # Save the mean time series for later plotting
        mean_layer = A.mean(axis=1)  # shape (T,)
        mean_series_list.append((g, mean_layer))

    # Now, in a separate figure, let's plot ONLY the means for each g
    plt.figure()
    for (g, mean_vec) in mean_series_list:
        plt.plot(mean_vec, label=f"g={g}")
    plt.title("Mean Layer for each g")
    plt.xlabel("Time")
    plt.ylabel("Mean a_j(t)")
    plt.legend()
    plt.tight_layout()
    plt.show()