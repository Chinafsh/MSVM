
#==============================================================================
# shihao.feng@foxmail.com
# Shihao Feng
# 2025.4.14
#==============================================================================
# For loading datasets
import medmnist
from medmnist import BreastMNIST, PneumoniaMNIST
# For computation
import numpy as np
# For loading data
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# For plot curve
import matplotlib.pyplot as plt

# For computing with-class scatter matrix & between-class scatter matrix
from calculate_L import calculate_L

# For LSSVM 
# See https://github.com/RomuloDrumond/LSSVM
from lssvm import LSSVC

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_loader(type, root='./data/', download=True):
    """Load data

    Args:
        type (string): "BreastMNIST or PneumoniaMNIST"
        root (str, optional): directory of data. Defaults to './data/'.
        download (bool, optional): download dataset from internet or not Defaults to True.
    """
    
    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    if type == 'BreastMNIST':
        train_dataset = BreastMNIST(root='./data/', split='train', download=True, transform=data_transform)
        test_dataset = BreastMNIST(root='./data/', split='test', download=True, transform=data_transform)
    elif type == 'PneumoniaMNIST':
        train_dataset = PneumoniaMNIST(root='./data/', split='train', download=True, transform=data_transform)
        test_dataset = PneumoniaMNIST(root='./data/', split='test', download=True, transform=data_transform)
    else:
        raise("Unknown data to load!\n Please chose BreastMNIST or PneumoniaMNIST, which we support")
    
    
    train_bs = len(train_dataset)
    test_bs = len(test_dataset)
    print(f"Train dataset includes {train_bs} samples, Test dataset includes {test_bs} samples.")
    
    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=False)
    
    return train_loader, test_loader

def train(train_loader, dim=None, mu=1, beta=1, gamma=1):
    
    X,Y = next(iter(train_loader))
    
    X = X.flatten(start_dim=1).cpu().numpy().T # (features, samples)
    Y = Y.cpu().numpy() # (samples, 1)

    features = X.shape[0]
    if dim is None:
        dim = round(features/2)   # half of features
    
    # LDA's Sb and Sw
    Sb, Sw, _,_ = calculate_L(X.T, Y)
    St = Sb + Sw
    
    # arbitrary columnly orthogonal matrix
    # Random orthogonal matrix P projects data into a lower-dimensional space.
    P = np.random.rand(features, dim)
    P, _ = np.linalg.qr(P)  # QR decomposition for orthogonality    
    
    
    # Diagonal matrix D is initialized to penalize rows of P (related to L2,1-norm regularization).
    norm_P_rows = np.linalg.norm(P, axis=1)  # Calculate L2 norm of rows
    D = np.diag(1 / (norm_P_rows + 1e-12)) # Create diagonal matrix
    
    # get first LS-SVM model in the initial P subspace.
    p_train = P.T @ X
    
    
    lssvc = LSSVC(gamma=gamma, kernel='linear')
    lssvc.fit(p_train.T, Y) 
    
    alpha = lssvc.alpha.reshape(-1, 1)

    tmpv = X @ (alpha * Y)
    TSum = tmpv @ tmpv.T

    
    combined_matrix = TSum + beta * Sw + mu * D
    trace_term = np.trace(P.T @ combined_matrix @ P)
    alpha_norm_squared = np.sum(alpha ** 2)  #  ||alpha||_2^2
    regularization_term = alpha_norm_squared / (2 * gamma)
    
    obj_list = []
    obj = trace_term + regularization_term
    
    for i in range(10):

        for j in range(10):
            # Sp
            Sp = TSum + mu * D + beta * Sw

            # lambda_n
            lambda_n = np.trace(P.T@St@P) / np.trace(P.T@Sp@P)
            B = St - lambda_n*Sp

            # Calculate the eigne vector
            B = (B + B.T) / 2
            Lambda, V = np.linalg.eig(B)
            # Sort eigenvalues in descending order and get the indices
            index = np.argsort(Lambda)[::-1]  # Get indices that would sort the eigenvalues
            P_old = P
            P = V[:, index[:dim]]  # Sort columns of eigenvectors

            # singular decomposition for the sake of orthogonal transformation invariance
            Sp_p = P @ P.T @ Sp @ P @ P.T
            tempU, _, _ = np.linalg.svd(Sp_p, full_matrices=False)  # Economy SVD
            P = tempU[:, :dim]
            D = np.eye(features)
            for ii in range(features):
                norm_value = np.linalg.norm(P[ii, :]) + 1e-12
                D[ii, ii] = 1.0 / norm_value

            if np.linalg.norm(P-P_old) < np.sqrt(features*dim)*0.01:
                print("inside loop converges")
                break

        p_train = P.T @ X
        lssvc = LSSVC(gamma=gamma, kernel='linear')
        lssvc.fit(p_train.T, Y)
        alpha = lssvc.alpha.reshape(-1, 1)
        # update TSum
        tmpv = X @ (alpha * Y)
        TSum = tmpv @ tmpv.T

        obj_list.append(obj)# For recording
        obj_old = obj
        obj = np.trace(P.T @ (TSum + beta*Sw + mu*D) @ P) + np.linalg.norm(alpha,ord=2) / (2*gamma)
        if abs(obj - obj_old)/abs(obj) <= 0.1:
            print(f"outside loop converges at iter {i}")
            
        
    p_train = P.T @ X
    lssvc = LSSVC(gamma=gamma, kernel='linear')
    lssvc.fit(p_train.T, Y)
    
    return lssvc, P, obj_list
    
def predict(model, P, test_loader):
    X_test, Y_test = next(iter(test_loader))
    X_test =X_test.flatten(start_dim=1).cpu().numpy().T
    Y_test= Y_test.cpu().numpy()
    n = Y_test.shape[0]

    p_test = P.T @ X_test

    test_pred = model.predict(p_test.T)

    accuracy = 100*np.sum(test_pred==Y_test[:, 0])/n
    
    return accuracy

def plot(obj_list, data_type):
    n = len(obj_list)
    plt.figure(figsize=(8, 4))  # 设置图像尺寸
    plt.plot(obj_list, color='blue', label='Objective value')  # 绘制曲线
    plt.xlabel('Iteration')  # X 轴标签
    plt.ylabel('Objective value')  # Y 轴标签
    plt.title(f'Objective values on data {data_type}')
    plt.savefig(f"{data_type}_loss_curve.png")
    
if __name__ == '__main__':
    # Hyper parameter define    
    # ============================================================================
    # Tune these parameters 
    dim = None # None means half feature dimension of original dimension
    mu = 1
    beta = 1
    gamma = 1
    # ============================================================================
    
    data_type = "BreastMNIST" # BreastMNIST or PneumoniaMNIST
    train_loader, test_loader = data_loader(type=data_type,root='./data/')
    
    trained_model, Projection_matrix, obj_list = train(train_loader, dim, mu, beta, gamma)
    
    accuracy = predict(model=trained_model, P = Projection_matrix, test_loader = test_loader)
    
    # Draw objective curve and save it
    plot(obj_list=obj_list, data_type=data_type)
    
    print(f"Test on data {data_type}, accuracy is {accuracy:.3f}.")
    
