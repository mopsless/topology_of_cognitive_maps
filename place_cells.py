# this is a big file containing my experience in choosing specific place cell neurons, and studying them using time series and point cloud analysis approaches.
# It would contain several steps made by my students and myself. 
# Contrubutors: me - Konstantin Sorokin, Robert Drynkin, Andrey Zaitsew, Yulia Kokorina, Michail Gorbunov. 

# 1 step. Taking a dataset and computing it's bins and spatial/mutual info
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import quantile_transform
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from typing import Tuple
from ripser import Rips
from numba import njit, jit
import numba
import PIL
from dionysus import Filtration, homology_persistence, init_diagrams, Simplex
from collections import Counter
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex

@njit
def cart2pol(x, y):
    """
    Polar coordinates to an old (symmetric) binarizer of an arena. TODO: make unweighted one 
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if phi<0:
        phi+=2*3.1416
    return(rho, phi)

@njit
def pol2cart(rho, phi):
    """
    Unverse operation
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

@njit
def gen_pos_labels(Y, n_cl):
    """
    labelize the animal position
    """
    Y_list = list(Y)
    Y_cyl = []
    for i in range(len(Y_list)):
        Y_cyl.append(cart2pol(Y_list[i][0], Y_list[i][1]))
    pos_ind_cent = []
    for i in range(len(Y_cyl)):
        if abs(Y_cyl[i][0]) <= 200:
            pos_ind_cent.append(i)
    pos_labels_cyl = []

    for j in range(len(Y_cyl)):
        for i in range(n_cl):    
            if 0+i*(2*3.1416/n_cl) <= Y_cyl[j][1] <= 0+(i+1)*(2*3.1416/n_cl):
                pos_labels_cyl.append(i)
                
    for k in range(len(pos_ind_cent)):
        pos_labels_cyl[pos_ind_cent[k]] = n_cl+1
    return np.array(pos_labels_cyl) 

def euc_distances(list_frame):
    euc_dist = []
    for i in range(len(list_frame)-1):
        euc_dist.append(np.sqrt((list_frame[i+1][0]-\
            list_frame[i][0])**2 + (list_frame[i+1][1]-\
            list_frame[i][1])**2))                
    return euc_dist

def slow_mouse(dist):
    """
    find the moments in time when an animal wasn't moving - it might potentially give 
    unaccurate inromation on non-p;acee cells neurons acivity
    """
    stationary = []
    for i in range(len(dist)):
        if dist[i] < 10:
            stationary.append(i)
    return stationary

# the following section is for second approach of neurons choosing. 
# searching for neurons related to the loops travelled by the mouse.
# double ternary search; works only locally; time ~ 1min

@njit
def min_dist(x, y, Y):
    cur_min = 1000
    for n_x, n_y in Y:
        cur_min = min(cur_min, np.sqrt((n_x - x) * (n_x - x)  + (n_y - y) * (n_y - y)))
    return cur_min

@njit
def ternary_search_y(x, y_min, y_max, Y):
    eps = 0.001
    l_y = y_min
    r_y = y_max
    while (r_y - l_y > eps):
        m1_y = l_y + (r_y - l_y)/3
        m2_y = r_y - (r_y - l_y)/3
        d1 = min_dist(x, m1_y, Y)
        d2 = min_dist(x, m2_y, Y)
        if d1 >= d2:
            r_y = m2_y
        else:
            l_y = m1_y
    return r_y, min(d1, d2)

@njit
def rad_and_center(x_min, x_max, y_min, y_max, Y):
    eps = 0.001
    l_x = x_min
    r_x = x_max
    while (r_x - l_x > eps):
        m1_x = l_x + (r_x - l_x)/3
        m2_x = r_x - (r_x - l_x)/3
        m1_y, d1 = ternary_search_y(m1_x, y_min, y_max, Y)
        m2_y, d2 = ternary_search_y(m2_x, y_min, y_max, Y)
        if d1 >= d2:
            r_x = m2_x
        else:
            l_x = m1_x
    return m1_x, m1_y, min(d1, d2)
        
#x_c, y_c - center of holes; rad - hole radius; k in [0; 1]; arr - active state
@njit
def isloop(x_c, y_c, rad, k, arr):
    x_m, y_m = mass_center(arr)
    if np.sqrt((x_c - x_m)**2 + (y_c - y_m)**2) < k * rad:
        return True;
    else:
        return False;

@njit
def mass_center(arr):
    return np.mean(arr[:, 0]), np.mean(arr[:, 1])

def segment_arena_kmeans(
        position: np.ndarray,
        n_clusters: int = 15
) -> np.ndarray:
    """
    Segment arena in n_clusters segments

    n - number of events
    :param position: float array of shape (n, 2) with columns x and y coordinates of mouse
    :param n_clusters: number of segments

    :return: int array of shape (n, ) with segments corresponds to position
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(position)
    return kmeans.labels_

@njit
def compute_possible_loops1():
    has_loop : list(tuple(int, int)) = []
    for i in range(Y_mod.shape[0]):
        for j in range(i+1,  Y_mod.shape[0]):
            if isloop1(x_m1, y_m1, rad1, 0.4, Y_norm_mod[i:j]) or isloop1(x_m2, y_m2, rad2, 0.4, Y_norm_mod[i:j]):
                if len(has_loop) == 0:
                    has_loop.append((i,j))
                elif (has_loop[-1][0] != i) and (has_loop[-1][1] != j-1):
                    has_loop.append((i,j))
                    
    return has_loop

def dist(idx):
    return np.linalg.norm(Y_norm_mod[idx[0]] - Y_norm_mod[idx[1]])

def erase_unnecessary_loops1(has_loop):
    new_loops = []
    new_loops.append(list(has_loop[0]))
    
    for pair in has_loop[1:]:
        if abs(pair[0] - new_loops[-1][0]) > 20 and abs(pair[1] - new_loops[-1][1]) > 20:
            new_loops.append(list(pair))
        elif dist(pair) < dist(new_loops[-1]):
            new_loops[-1] = list(pair)
    return new_loops

def get_neurons_position_mi(
        activity: np.ndarray,
        position: np.ndarray,
        n_activity_labels: int
) -> Tuple[float, np.ndarray]:
    """
    Compute MI between neurons and mouse position

    n - number of events, m - number of neurons
    :param activity: float array of shape (n, m) of neurons activity
    :param position: int array of shape (n, 2) with columns x and y coordinates of mouse
    :param n_activity_labels: number of labels to segment neurons activity

    :return: entropy of position and float array of shape (n, ) of MI between neurons and position
    """
    mis = []
    activity_labels = (quantile_transform(activity) * n_activity_labels).astype(np.int32)

    for x in activity_labels.T:
        mis.append(mutual_info_score(x, position))
    mis = np.array(mis)

    entropy = mutual_info_score(position, position)

    return entropy, mis


def get_mi_best_neurons(
        activity: np.ndarray,
        position: np.ndarray,
        n_best: int = 20,
        n_activity_labels: int = 15
) -> np.ndarray:
    """
    Get ids if n_best neurons by the MI with position

    :param activity: float array of shape (n, m) of neurons activity
    :param position: int array of shape (n, 2) with columns x and y coordinates of mouse
    :param n_activity_labels: number of labels to segment neurons activity
    :param n_best: number of ids to return

    :return: array with ids of n_best neurons by the MI with position
    """
    _, mis = get_neurons_position_mi(activity, position, n_activity_labels)

    return np.argsort(-mis)[:n_best]


def spatial_info_score(
        activity: np.ndarray,
        position: np.ndarray
) -> float:
    """
    Compute spatial information between neuron activity and mouse position
    :param activity: float array of shape (n, ) with neuron activities
    :param position: int array of shape (n, ) with mouse position corresponding to activity
    :return: spatial information between activity and position
    """
    si = 0.0
    mean_activity = np.mean(activity)
    for pos in np.unique(position):
        mean_activity_in_pos = np.mean(activity[position == pos])
        pos_prob = np.mean(position == pos)
        if not np.allclose(mean_activity_in_pos, 0):
            si += pos_prob * mean_activity_in_pos * np.log2(mean_activity_in_pos / mean_activity)
    return si


def get_neurons_position_si(
        activity: np.ndarray,
        position: np.ndarray
) -> np.ndarray:
    """
    Compute SI between neurons and mouse position

    n - number of events, m - number of neurons
    :param activity: float array of shape (n, m) of neurons activity
    :param position: int array of shape (n, 2) with columns x and y coordinates of mouse

    :return: float array of shape (n, ) of SI between neurons and position
    """
    sis = []

    for x in activity.T:
        sis.append(spatial_info_score(x, position))
    sis = np.array(sis)

    return sis


def get_si_best_neurons(
        activity: np.ndarray,
        position: np.ndarray,
        n_best: int = 20
) -> np.ndarray:
    """
    Get ids if n_best neurons by the SI with position

    :param activity: float array of shape (n, m) of neurons activity
    :param position: int array of shape (n, 2) with columns x and y coordinates of mouse
    :param n_best: number of ids to return

    :return: array with ids of n_best neurons by the SI with position
    """
    sis = get_neurons_position_si(activity, position)

    return np.argsort(-sis)[:n_best]

# here we obtain the Vietors-Rips filtration, construct complexes which further we can analyze. 

@njit
def get_all_subsets(arr):
    n = len(arr)
    for i in range(1, 2**n):
        a = (np.array(list(reversed(bin(int(i))[2:]))) == '1')
        a.resize(n)
        yield tuple(arr[a].tolist())

@njit       
def get_cycles(filtration, end_time=np.inf):
    m = homology_persistence(filtration, prime=2)
    dgms = init_diagrams(m, filtration)
    cycles = []
    for i, dgm in enumerate(dgms):
        for pt in dgm:
            cycles.append((i, pt.birth, pt.death))
    cycles = pd.DataFrame(cycles, columns=['dimension', 'birth_time', 'death_time'])
    cycles['living_time'] = cycles['death_time'] - cycles['birth_time']
    inf_life = (cycles['living_time'] == np.inf)
    cycles['living_time'][inf_life] = end_time - cycles['birth_time'][inf_life]
    return cycles.sort_values('living_time', ascending=False)

@njit
def construct_cycles(data, alphas, common_time_borders):
    filtrations = {
        border: Filtration()
        for border in common_time_borders
    }

    for alpha in alphas:
        print(f'alpha = {alpha}')
        neuron_activity = (data > alpha)

        cur_arr = {border: [] for border in common_time_borders}
        for sim in neuron_activity:
            for s in get_all_subsets(np.where(sim)[0]):
                total_time = np.sum(np.all(neuron_activity[:, s], axis=1))
                for border in common_time_borders:
                    if total_time > border:
                        cur_arr[border].append(s)

        for border in common_time_borders:
            cur_arr[border] = sorted(cur_arr[border], key=lambda x: len(x))
            for sim in cur_arr[border]:
                filtrations[border].append(Simplex(sim, 1 - alpha))
    
    return filtrations


class place_cells:
    
    def __init__(self, address, mc_address):
        if type(address) != str:
            raise ValueError('address is a string')
        if type(mc_address) != str:
            raise ValueError('address is a string')
        self.address = address
        self.mc_address = mc_address
        
    def tracks(self):
        data = pd.read_csv(self.address) 
        X = data[[x for x in data.columns if x.replace('.', '', 1).isdigit()]].values
        Y = data[['x', 'y']].values
        #Y = Y[3000:12000]
        #X = X[3000:12000]
        dist = euc_distances(list(Y))
        to_drop = slow_mouse(dist)

        Y[:,0]-=(Y[(np.argmin(Y[:,0])),0]+Y[(np.argmax(Y[:,0])),0])/2
        Y[:,1]-=(Y[(np.argmin(Y[:,1])),1]+Y[(np.argmax(Y[:,1])),1])/2
        Y_list = list(Y)
        Y_cyl = []
        for i in range(len(Y_list)):
            Y_cyl.append(cart2pol(Y_list[i][0], Y_list[i][1]))
        n_cl = 12

        pos_ind_cent = []
        for i in range(len(Y_cyl)):
            if abs(Y_cyl[i][0]) <= 200:
                pos_ind_cent.append(i)

        pos_labels_cyl = []

        for j in range(len(Y_cyl)):
            for i in range(n_cl):    
                if 0+i*(2*3.1416/n_cl) <= Y_cyl[j][1] <= 0+(i+1)*(2*3.1416/n_cl):
                    pos_labels_cyl.append(i)

        for k in range(len(pos_ind_cent)):
            pos_labels_cyl[pos_ind_cent[k]] = n_cl+1
        pos_labels = np.array(pos_labels_cyl) 
        ### finished for running mouse
        
        Y_mod = np.delete(Y, to_drop, axis = 0)
        X_mod = np.delete(X, to_drop, axis = 0)
        print('original trace', len(list(Y)), 'trace when mouse was only running', len(list(Y_mod)))

        
        Y_list_mod = list(Y_mod)
        Y_cyl_mod = []
        for i in range(len(Y_list_mod)):
            Y_cyl_mod.append(cart2pol(Y_list_mod[i][0], Y_list_mod[i][1]))
        n_cl = 12

        pos_ind_cent_mod = []
        for i in range(len(Y_cyl_mod)):
            if abs(Y_cyl_mod[i][0]) <= 200:
                pos_ind_cent_mod.append(i)
        pos_labels_cyl_mod = []
        for j in range(len(Y_cyl_mod)):
            for i in range(n_cl):    
                if 0+i*(2*3.1416/n_cl) <= Y_cyl_mod[j][1] <= 0+(i+1)*(2*3.1416/n_cl):
                    pos_labels_cyl_mod.append(i)

        for k in range(len(pos_ind_cent_mod)):
            pos_labels_cyl_mod[pos_ind_cent_mod[k]] = n_cl+1
        pos_labels_mod = np.array(pos_labels_cyl_mod) 
        plt.figure(figsize=(8, 8))
        plt.scatter(Y[:, 0], Y[:, 1], c=pos_labels*0.1)
        plt.show()
        
        plt.figure(figsize=(8, 8))
        plt.scatter(Y_mod[:, 0], Y_mod[:, 1], c=pos_labels_mod*0.1)
        plt.show()
        
        plt.figure(figsize=(10, 5))
        fig = plt.hist(np.array([pos_labels, pos_labels_mod]), bins = n_cl+1, density = True)
        
        return [quantile_transform(X), quantile_transform(X_mod), Y, Y_mod, pos_labels, pos_labels_mod]
        
    def best_mi_si_intersect(self, X1, X1_mod, Y, Y_mod, pos_labels, pos_labels_mod):
        best_mi_ids = get_mi_best_neurons(X1, pos_labels, 80, 100)
        best_si_ids = get_si_best_neurons(X1, pos_labels, 80)
        best_mi_ids_mod = get_mi_best_neurons(X1_mod, pos_labels_mod, 80, 100)
        best_si_ids_mod = get_si_best_neurons(X1_mod, pos_labels_mod, 80,)
        best_mi_ids_int = np.intersect1d(best_mi_ids, best_mi_ids_mod)
        best_si_ids_int = np.intersect1d(best_si_ids, best_si_ids_mod)
        best_ids_int = np.intersect1d(best_si_ids_int, best_mi_ids_int)
        
        print('best neurons of spatial an mutual information')
        plt.figure(figsize=(15, 25))
        for i, neur in enumerate(best_ids_int):
            plt.subplot(10, 5, i + 1)
            plt.title(neur)

            active_state = (X1_mod[:, neur] > 0.8)
            active_state1 = (X1_mod[:, neur] > 0.95)
            plt.scatter(Y_mod[:, 0], Y_mod[:, 1], alpha=0.05)
            plt.scatter(Y_mod[active_state][:, 0], Y_mod[active_state][:, 1], alpha=.1, color = 'r')
            plt.scatter(Y_mod[active_state1][:, 0], Y_mod[active_state1][:, 1], alpha=.1, color = 'g')
            plt.axis('off')
        return best_ids_int
    
    def construct_iso(X, chosen_cells)

        embedding = Isomap(n_components=2)
        X_transformed = embedding.fit_transform(X[:, chosen_cells[:]])
        #plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=10, alpha=0.2, c=colors[:])
        
        X_transformed_pol = np.array([cart2pol(X_transformed[i][0], X_transformed[i][1]) \
                     for i in range(len(X_transformed))])
        #plt.figure(figsize=(10, 10))
        #plt.axes(polar = True)
        #plt.axes(projection = 'polar')
        #plt.scatter(X_transformed_pol[:, 1], X_transformed_pol[:, 0], c = colors[:])
        #plt.scatter((X_transformed_pol[np.argmax(X_transformed_pol[:,0]), 1]), X_transformed_pol[np.argmax(X_transformed_pol[:,0]), 0], c = 'r')
        
        threshold = np.max(X_transformed_pol[:,0])*.2 #(you might vary it dependng on dataset (the loops might be more or less prononunced))
#X_transformed_pol = np.array(((X_transformed_pol.T[0]),\
#                                  (X_transformed_pol.T[1]+np.pi/4))).T
        circ = []
        for i in range(36):
            circ.append([threshold, i])
        circ = np.array(circ)

        loop_angle = X_transformed_pol[np.argmax(X_transformed_pol[:,0]), 1]
        loop_angle_range = np.linspace(loop_angle + 1.5, loop_angle + 3.65, 20)
        rotate_angle =  2*np.pi - loop_angle_range[0]
        X_transformed_pol_tmp = X_transformed_pol[:,1] + rotate_angle
        loop_angle_range_tmp = loop_angle_range[:] + rotate_angle -2*np.pi
        for i in range(len(X_transformed_pol_tmp)):
            if X_transformed_pol_tmp[i] > np.pi*2:
                X_transformed_pol_tmp[i] -= np.pi*2
        loop_coords = []
        for i in range(len(X_transformed_pol)):
            if loop_angle_range_tmp[0] < X_transformed_pol_tmp[i] < \
            loop_angle_range_tmp[-1] and X_transformed_pol[i, 0] > threshold:
                loop_coords.append([X_transformed_pol[i,0], \
                                    X_transformed_pol[i,1], int(i)])
        loop_coords = np.array(loop_coords)
        
        plt.figure(figsize=(10, 10))
        plt.axes(polar = True)
        plt.scatter(X_transformed_pol[:, 1], X_transformed_pol[:, 0], c = "blue")
        plt.scatter((X_transformed_pol[np.argmax(X_transformed_pol[:,0]), 1]),\
                    X_transformed_pol[np.argmax(X_transformed_pol[:,0]), 0], c = 'r')
        plt.scatter(circ[:, 1], circ[:, 0])
        plt.scatter(loop_coords[:, 1], loop_coords[:,0], c = "orange")
    
    
class SimpleAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        """Create new autoencoder with pre-defined latent dimension."""
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.input_dim, self.latent_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.input_dim, self.input_dim),
        )

        self.loss_fn = torch.nn.MSELoss()

    def encode(self, x):
        """Embed data in latent space."""
        return self.encoder(x)

    def decode(self, z):
        """Decode data from latent space."""
        return self.decoder(z)

    def forward(self, x):
        """Embeds and reconstructs data, returning a loss."""
        z = self.encode(x)
        x_hat = self.decode(z)

        # The loss can of course be changed. If this is your first time
        # working with autoencoders, a good exercise would be to 'grok'
        # the meaning of different losses.
        reconstruction_error = self.loss_fn(x, x_hat)
        return reconstruction_error

    def TopologicalAutoencoderTo2D(self, Xb, n_epochs, lam):
        """
            Input:
                Xb: d-dimensional array
                n_epochs: number of iterarions in training loop
                lam: topological loss coefficient (loss = geom_loss + lam * topo_loss)
            Output:
                Z: 2-dimensional array that we got after encoding Xb
        """

        X_tensor = torch.Tensor(Xb)
        data_set = TensorDataset(X_tensor)


        train_loader = DataLoader(
            data_set,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )

        model = SimpleAutoencoder(input_dim=Xb.shape[1])
        topo_model = TopologicalAutoencoder(model, lam=lam)
        optimizer = optim.Adam(topo_model.parameters(), lr=1e-5, weight_decay=1e-5)

        progress = range(n_epochs)

        for i in progress:
            topo_model.train()

            for batch, x in enumerate(train_loader):
                loss = model(x[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #print('{}: loss = {}'.format(i, loss.item()))


        test_loader = DataLoader(
                data_set,
                shuffle=False,
                batch_size=len(data_set)
        )

        X = next(iter(test_loader))
        Z = model.encode(X[0]).detach().numpy()

        return Z

    def TopologicalAutoencoderTo2D(Xb, n_epochs, lam):
        """
            Input:
                Xb: d-dimensional array
                n_epochs: number of iterarions in training loop
                lam: topological loss coefficient (loss = geom_loss + lam * topo_loss)
            Output:
                Z: 2-dimensional array that we got after encoding Xb
        """

        X_tensor = torch.Tensor(Xb)
        data_set = TensorDataset(X_tensor)


        train_loader = DataLoader(
            data_set,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )

        model = SimpleAutoencoder(input_dim=Xb.shape[1])
        topo_model = TopologicalAutoencoder(model, lam=lam)
        optimizer = optim.Adam(topo_model.parameters(), lr=1e-5, weight_decay=1e-5)

        progress = range(n_epochs)

        for i in progress:
            topo_model.train()

            for batch, x in enumerate(train_loader):
                loss = model(x[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #print('{}: loss = {}'.format(i, loss.item()))


        test_loader = DataLoader(
                data_set,
                shuffle=False,
                batch_size=len(data_set)
        )

        X = next(iter(test_loader))
        Z = model.encode(X[0]).detach().numpy()

        return Z

    
    
    
    
    
    

