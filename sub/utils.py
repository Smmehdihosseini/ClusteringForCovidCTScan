import numpy   as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from itertools import product
import winsound

def fetch_nii(path):
    '''
    Reads .nii file and returns pixel array
    '''
    array = nib.load(path).get_fdata()
    rotated_array = np.rot90(np.array(array))
    return(rotated_array)

def sample_plot(array_list, color_map = 'nipy_spectral'):
    '''
    Plots a slice with all available annotations
    '''
    plt.figure(figsize=(18,15))
    
    original = array_list[0]
    lung = array_list[1]
    infection = array_list[2]
    lung_n_infection = array_list[3]
    
    #Original
    plt.subplot(1,4,1)
    plt.imshow(original, cmap='bone',interpolation="nearest")
    plt.title('Original Image')
    
    #Lung Mask
    plt.subplot(1,4,2)
    plt.imshow(original, cmap='bone',interpolation="nearest")
    plt.imshow(lung, alpha=0.5, cmap=color_map)
    plt.title('Lung Mask')
    
    #Infection Mask
    plt.subplot(1,4,3)
    plt.imshow(original, cmap='bone',interpolation="nearest")
    plt.imshow(infection, alpha=0.5, cmap=color_map)
    plt.title('Infection Mask')
    
    #Lung and Infection Mask
    plt.subplot(1,4,4)
    plt.imshow(original, cmap='bone',interpolation="nearest")
    plt.imshow(lung_n_infection, alpha=0.5, cmap=color_map)
    plt.title('Lung and Infection Mask')

    plt.savefig('Samples.png', dpi=400)
    plt.show()


def filterImage(D,NN):
    """D = image (matrix) to be filtered, Nr rows, N columns, scalar values (no RGB color image)
    The image is filtered using a square kernel/impulse response with side 2*NN+1"""
    E=D.copy()
    E[np.isnan(E)]=0
    Df=E*0
    Nr,Nc=D.shape
    rang=np.arange(-NN,NN+1)
    square=np.array([x for x in product(rang, rang)])
    for kr in range(NN,Nr-NN):
        for kc in range(NN,Nc-NN):
            ir=kr+square[:,0]
            ic=kc+square[:,1]
            Df[kr,kc]=np.sum(E[ir,ic])
    return Df/square.size

def MyDBSCAN(D, z, epsv, min_samplesv):
    """D is the image to process, z is the list of image coordinates to be
    clustered"""
    Nr, Nc = D.shape
    clusters = DBSCAN(eps=epsv,min_samples=min_samplesv,metric='euclidean').fit(z)
    a,Npoints_per_cluster = np.unique(clusters.labels_,return_counts=True)
    Nclust_DBSCAN = len(a) - 1
    Npoints_per_cluster = Npoints_per_cluster[1:]
    ii = np.argsort(-Npoints_per_cluster)
    Npoints_per_cluster = Npoints_per_cluster[ii]
    C = np.zeros((Nr,Nc,Nclust_DBSCAN)) * np.nan
    info = np.zeros((Nclust_DBSCAN,5), dtype=float)
    for k in range(Nclust_DBSCAN):
        i1 = ii[k] 
        index = (clusters.labels_==i1)
        jj = z[index,:]
        C[jj[:,0],jj[:,1],k] = 1
        a = np.mean(jj, axis=0).tolist()
        b = np.var(jj, axis=0).tolist()
        info[k,0:2] = a
        info[k,2:4] = b
        info[k,4] = Npoints_per_cluster[k]
    return C,info,clusters