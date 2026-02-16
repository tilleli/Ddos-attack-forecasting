import numpy as np

def create_sequences_multi_horizon(data, seq_length, horizons=[1, 5, 10, 20]):
    """
    Create input-output sequences for multiple prediction horizons.

    Args:
        data (array-like): Array of values to create sequences from.
        seq_length (int): Length of input sequences.
        horizons (list): List of horizon steps to predict ahead.

    Returns:
        dict: Dictionary where keys are horizon values and
              values are tuples of (X, y) numpy arrays.
    """

    # Initialize a dictionary to store results for each horizon
    results = {}

    # Iterate through each specified prediction horizon
    for horizon in horizons:

        # Initialize empty lists to store input sequences and target values
        xs = []
        ys = []

        # Iterate over valid indices to create input-output pairs
        # Stop before exceeding the limit that allows a full input sequence
        # AND a valid target value at the specified horizon
        for i in range(len(data) - seq_length - horizon):

            # Create input sequence: sliding window of length seq_length
            x = data[i:(i + seq_length)]

            # Get the target value located 'horizon' steps
            # after the end of the input sequence
            # The -1 adjusts for Python's 0-based indexing
            y = data[i + seq_length + horizon - 1]

            # Append the input-output pair to the lists
            xs.append(x)
            ys.append(y)

        # Convert lists to numpy arrays and store them in the dictionary
        results[horizon] = (np.array(xs), np.array(ys))

        # Print shape information for debugging/verification
        print(
            f"Horizon {horizon} steps - "
            f"Shape of X: {np.array(xs).shape}, "
            f"Shape of y: {np.array(ys).shape}"
        )

    # Return the complete dictionary containing all horizons
    return results
