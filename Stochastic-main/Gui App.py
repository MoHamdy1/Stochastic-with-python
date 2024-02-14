import numpy as np
import customtkinter as ctk
from customtkinter import CTkLabel, CTkEntry, CTkButton, CTkTextbox
from tkinter import scrolledtext
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=8)

Pi = np.array([0.99, 0.01])  # Default value for Pi


def forward_algorithm(observations, P, E, Pi):
    # Number of hidden states
    num_states = len(Pi)
    # Number of observations
    num_observations = len(observations)
    # Initialize the forward matrix
    forward_matrix = np.zeros((num_states, num_observations))
    # Init the first value of each state
    forward_matrix[:, 0] = Pi * E[:, observations[0]]
    # Print the initial forward variables
    print(f"Alpha at time 1:\n{forward_matrix[:, 0]}\n")
    # Fill in the rest of the forward matrix
    for t in range(1, num_observations):
        for j in range(num_states):
            forward_matrix[j, t] = np.sum(
                forward_matrix[i, t - 1] * P[i, j] * E[j, observations[t]]
                for i in range(num_states)
            )

        # Print the forward variables at each time step
        print(f"Alpha at time {t+1}:\n{forward_matrix[:, t]}\n")
    # Termination step: Compute the likelihood of the sequence by summation of last 2 times
    likelihood = np.sum(forward_matrix[:, num_observations - 1])
    # Print the final forward matrix and likelihood
    print(
        f"Final Forward Matrix: \n{forward_matrix}\n",
        f"Likelihood of the sequence: {likelihood}",
        sep="\n",
    )
    return forward_matrix, likelihood


def backward_algorithm(observations, P, E):
    # Number of hidden states
    num_states = len(P)
    # Number of observations
    num_observations = len(observations)
    # Initialize the backward matrix
    backward_matrix = np.zeros((num_states, num_observations))
    # Initialize the last column of the backward matrix to 1
    backward_matrix[:, num_observations - 1] = 1
    # Print the initial backward variables
    print(
        f"Beta at time {num_observations}:\n{backward_matrix[:, num_observations - 1]}\n"
    )
    # Fill in the rest of the backward matrix
    for t in range(num_observations - 2, -1, -1):
        for i in range(num_states):
            backward_matrix[i, t] = np.sum(
                backward_matrix[j, t + 1] * P[i, j] * E[j, observations[t + 1]]
                for j in range(num_states)
            )

        # Print the backward variables at each time step
        print(f"Beta at time {t+1}:\n{backward_matrix[:, t]}\n")

    # Termination step: Compute the likelihood of the sequence using the initial probabilities and the first observation
    likelihood = np.sum(
        backward_matrix[i, 0] * P[0, j] * E[j, observations[0]]
        for j in range(num_states)
    )

    # Print the final backward matrix and likelihood
    print(
        f"Final Backward Matrix: \n{backward_matrix}\n",
        f"Likelihood of the sequence: {likelihood}",
        sep="\n",
    )

    return backward_matrix, likelihood


def viterbi_algorithm(observations, P, E, Pi):
    # Number of hidden states
    num_states = len(Pi)
    # Number of observations
    num_observations = len(observations)
    # Initialize the Viterbi matrix
    viterbi_matrix = np.zeros((num_states, num_observations))
    # Initialize the backpointer matrix
    path = np.zeros((num_states, num_observations), dtype=int)
    # Initialization step
    viterbi_matrix[:, 0] = Pi * E[:, observations[0]]
    # Recursion step
    for t in range(1, num_observations):
        for j in range(num_states):
            prev_probs = viterbi_matrix[:, t - 1] * P[:, j] * E[j, observations[t]]
            viterbi_matrix[j, t] = round(max(prev_probs), 5)
            path[j, t] = np.argmax(prev_probs)

    # Termination step
    best_path_prob = max(viterbi_matrix[:, -1])
    best_last_state = np.argmax(viterbi_matrix[:, -1])

    # Backtrace to find the best path
    best_path = [best_last_state]
    for t in range(num_observations - 1, 0, -1):
        best_last_state = path[best_last_state, t]
        best_path.insert(0, best_last_state)

    return best_path, best_path_prob, viterbi_matrix


def baum_welch(
    Pi,
    Emission,
    Transition,
    observations,
    num_iterations=100,
):
    state_numbers = len(Pi)
    # Iterate through the Baum-Welch algorithm
    for iteration in range(num_iterations):
        # E-step: Use forward and backward algorithms to calculate expected counts
        forward_matrix, forward_likelihood = forward_algorithm(
            observations, Transition, Emission, Pi
        )
        backward_matrix, backward_likelihood = backward_algorithm(
            observations, Transition, Emission
        )

        # M-step: Update parameters based on expected counts
        # Update initial state probabilities
        Pi = forward_matrix[:, 0] * backward_matrix[:, 0] / forward_likelihood

        # Update transition probabilities
        for i in range(state_numbers):
            for j in range(state_numbers):
                numer = sum(
                    forward_matrix[i, t]
                    * Transition[i, j]
                    * Emission[j, observations[t + 1]]
                    * backward_matrix[j, t + 1]
                    for t in range(len(observations) - 1)
                )
                denom = sum(
                    forward_matrix[i, t] * backward_matrix[i, t]
                    for t in range(len(observations) - 1)
                )
                Transition[i, j] = numer / denom

        # Update emission probabilities
        for j in range(state_numbers):
            for k in range(state_numbers):
                numer = sum(
                    forward_matrix[i, t] * backward_matrix[i, t]
                    if observations[t] == k
                    else 0
                    for t in range(len(observations) - 1)
                )
                denom = sum(
                    forward_matrix[i, t] * backward_matrix[i, t]
                    for t in range(len(observations) - 1)
                )
                Emission[j, k] = numer / denom

    return Pi, Transition, Emission


# Function to apply the forward algorithm
def apply_forward():
    global Pi, Transition, Emission
    observations = [int(x) for x in obs_entry.get().split(",")]
    Pi = np.array([float(x) for x in pi_entry.get().split(",")])
    transition_matrix = np.array(
        [
            list(map(float, line.split(",")))
            for line in transition_entry.get("1.0", ctk.END).splitlines()
        ]
    )
    emission_matrix = np.array(
        [
            list(map(float, line.split(",")))
            for line in emission_entry.get("1.0", ctk.END).splitlines()
        ]
    )

    forward_matrix, likelihood_forward = forward_algorithm(
        observations, transition_matrix, emission_matrix, Pi
    )
    result_text.insert(
        ctk.END,
        f"Final Forward Matrix: \n{forward_matrix}\n",
        f"Likelihood of the sequence: {likelihood_forward}\n",
    )


# Function to apply the backward algorithm
def apply_backward():
    global Pi, Transition, Emission
    observations = [int(x) for x in obs_entry.get().split(",")]
    Pi = np.array([float(x) for x in pi_entry.get().split(",")])
    transition_matrix = np.array(
        [
            list(map(float, line.split(",")))
            for line in transition_entry.get("1.0", ctk.END).splitlines()
        ]
    )
    emission_matrix = np.array(
        [
            list(map(float, line.split(",")))
            for line in emission_entry.get("1.0", ctk.END).splitlines()
        ]
    )

    backward_matrix, likelihood_backward = backward_algorithm(
        observations, transition_matrix, emission_matrix
    )
    result_text.insert(
        ctk.END,
        f"Final Backward Matrix: \n{backward_matrix}\n",
        f"Likelihood of the sequence: {likelihood_backward}\n",
    )


# Function to apply the Viterbi algorithm
def apply_viterbi():
    global Pi, Transition, Emission
    observations = [int(x) for x in obs_entry.get().split(",")]
    Pi = np.array([float(x) for x in pi_entry.get().split(",")])
    transition_matrix = np.array(
        [
            list(map(float, line.split(",")))
            for line in transition_entry.get("1.0", ctk.END).splitlines()
        ]
    )
    emission_matrix = np.array(
        [
            list(map(float, line.split(",")))
            for line in emission_entry.get("1.0", ctk.END).splitlines()
        ]
    )

    best_path, best_path_prob, viterbi_matrix = viterbi_algorithm(
        observations, transition_matrix, emission_matrix, Pi
    )

    result_text.insert(
        ctk.END,
        "\nViterbi Algorithm:\n"
        f"Best Path: {best_path}\n"
        f"Best Path Probability: {best_path_prob}\n"
        f"Viterbi Matrix:\n{viterbi_matrix}\n\n",
    )


# Function to apply Baum-Welch algorithm
def apply_baum_welch():
    global Pi, Transition, Emission
    observations = [int(x) for x in obs_entry.get().split(",")]
    Pi = np.array([float(x) for x in pi_entry.get().split(",")])
    transition_matrix = np.array(
        [
            list(map(float, line.split(",")))
            for line in transition_entry.get("1.0", ctk.END).splitlines()
        ]
    )
    emission_matrix = np.array(
        [
            list(map(float, line.split(",")))
            for line in emission_entry.get("1.0", ctk.END).splitlines()
        ]
    )
    num_iterations = int(iterations_entry.get())

    Pi, Transition, Emission = baum_welch(
        Pi, emission_matrix, transition_matrix, observations, num_iterations
    )

    result_text.insert(
        ctk.END,
        "\nBaum-Welch Algorithm:\n"
        f"Final Initial State Probabilities: {Pi}\n"
        f"Final Transition Probabilities:\n{Transition}\n"
        f"Final Emission Probabilities:\n{Emission}\n"
    )


# Function to reset the result text
def reset_result():
    result_text.delete(1.0, ctk.END)


# Main Tkinter window
root = ctk.CTk()
root.title("Hidden Markov Model Algorithms")

# Labels and entry widgets
obs_CTkLabel = CTkLabel(root, text="Observations (comma-separated):")
obs_entry = CTkEntry(root)
obs_CTkLabel.grid(row=0, column=0, padx=10, pady=10)
obs_entry.grid(row=0, column=1, padx=10, pady=10)

# Entry widgets for matrices
transition_CTkLabel = CTkLabel(root, text="Transition matrix (one row per line):")
transition_entry = CTkTextbox(root, height=70, wrap=ctk.WORD)
transition_CTkLabel.grid(row=1, column=0)
transition_entry.grid(row=1, column=1)

emission_CTkLabel = CTkLabel(root, text="Emission matrix (one row per line):")
emission_entry = CTkTextbox(root, height=70, wrap=ctk.WORD)
emission_CTkLabel.grid(row=2, column=0, padx=10, pady=10)
emission_entry.grid(row=2, column=1, padx=10, pady=10)

# Entry widget for Pi
pi_CTkLabel = CTkLabel(root, text="Initial Probability (comma-separated):")
pi_entry = CTkEntry(root)
pi_CTkLabel.grid(row=3, column=0, padx=10, pady=10)
pi_entry.grid(row=3, column=1, padx=10, pady=10)

# Entry widget for number of iterations
iterations_CTkLabel = CTkLabel(root, text="Number of Iterations (Only for Baum Welch):")
iterations_entry = CTkEntry(root)
iterations_CTkLabel.grid(row=4, column=0, padx=10, pady=10)
iterations_entry.grid(row=4, column=1, padx=10, pady=10)

# Button for Baum-Welch algorithm


# Buttons for algorithms
forward_button = CTkButton(root, text="Apply Forward Algorithm", command=apply_forward)
backward_button = CTkButton(
    root, text="Apply Backward Algorithm", command=apply_backward
)
baum_welch_button = CTkButton(
    root, text="Apply Baum-Welch Algorithm", command=apply_baum_welch
)
viterbi_button = CTkButton(root, text="Apply Viterbi Algorithm", command=apply_viterbi)
reset_button = CTkButton(root, text="Reset Result", command=reset_result)
forward_button.grid(row=5, column=0, padx=10, pady=10)
backward_button.grid(row=5, column=1, padx=10, pady=10)
viterbi_button.grid(row=6, column=0, padx=10, pady=10)
reset_button.grid(row=7, column=0, padx=10, pady=10)
baum_welch_button.grid(row=6, column=1, padx=10, pady=10)
label_for_check = CTkLabel(root, text="Please don't leave Blanks!")
label_for_check.grid(row=7, column=1, padx=10, pady=10)


# Result display
result_text = CTkTextbox(root, width=350, wrap=ctk.WORD)
result_text.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

# Keep the main loop running
root.mainloop()
