import numpy as np
def calculate_L(X, Y):
    """
    Calculates L_b and L_w defined in traditional LDA.

    Args:
        X: Training data (NumPy array), where each row is a data point.
        Y: Labels (NumPy array).

    Returns:
        Sb: Between-class scatter matrix.
        Sw: Within-class scatter matrix.
        L_b: Between-class scatter matrix coefficient.
        L_w: Within-class scatter matrix coefficient.
    """

    data_n = X.shape[0]  # Number of data points
    class_set = np.unique(Y)
    class_n = len(class_set)

    W = np.zeros((data_n, data_n))

    for i in range(class_n):
        U = (Y == class_set[i])
        count = np.sum(U)
        index = np.where(U)[0]
        W[np.ix_(index, index)] = 1 / count

    L_w = np.eye(data_n) - W
    L_t = np.eye(data_n) - np.ones((data_n, 1)) @ np.ones((1, data_n)) / data_n
    L_b = L_t - L_w

    L_w = (L_w + L_w.T) / 2
    L_b = (L_b + L_b.T) / 2

    Sb = X.T @ L_b @ X
    Sw = X.T @ L_w @ X

    Sb = (Sb + Sb.T) / 2
    Sw = (Sw + Sw.T) / 2

    return Sb, Sw, L_b, L_w
