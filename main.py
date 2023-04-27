import numpy   as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from itertools import product
from sub.utils import *
import winsound
        
# Read sample
plt.close('all')    
plotFlag=True

folder1='./data/ct_scans'
folder2='./data/lung_mask'
folder3='./data/infection_mask'
folder4='./data/lung_and_infection_mask'
name1='/coronacases_org_001.nii'
name2='/coronacases_001.nii'

ct_samps = fetch_nii(folder1 + name1 + name1)
lung_samps = fetch_nii(folder2 + name2 + name2)
infection_samps = fetch_nii(folder3 + name2 + name2)
all_samps = fetch_nii(folder4 + name2 + name2)

N_rows, N_cols, N_images = ct_samps.shape # 512 * 512 * 301

# Examine one slice of a ct scan and its annotations

index = 125
sample_ct = ct_samps[...,index]
sample_lung = lung_samps[...,index]
sample_infection = infection_samps[...,index]
sample_all = all_samps[...,index]

sample_plot([sample_ct, sample_lung, sample_infection, sample_all])

ct_hist = np.histogram(sample_ct, 200, density=True)

if plotFlag:
    plt.figure()
    plt.plot(ct_hist[1][0:200], ct_hist[0])
    plt.title('Histogram of CT values in Slice ' + str(index))
    plt.grid()
    plt.xlabel('Value')
    plt.savefig('histogram.png', dpi=400)
    plt.show()


# Use Kmeans to perform color quantization of the image

N_cluster = 5
Kmeans = KMeans(n_clusters=N_cluster, random_state=0)
sample_ct_reshaped = sample_ct.reshape(-1,1)
Kmeans.fit(sample_ct_reshaped)
Kmeans_centroids = Kmeans.cluster_centers_.flatten()

for cluster in range(N_cluster):
    k_indexes = (Kmeans.labels_ == cluster)
    sample_ct_reshaped[k_indexes] = Kmeans_centroids[cluster]
sample_ct_quantized = sample_ct_reshaped.reshape(N_rows, N_cols)

Vmin = sample_ct.min()
Vmax = sample_ct.max()

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(sample_ct, cmap='bone', interpolation="nearest")
ax1.set_title('Original Image')
ax2.imshow(sample_ct_quantized, vmin = Vmin, vmax = Vmax, cmap='bone', interpolation="nearest")
ax2.set_title('Quantized Image')
plt.savefig('org_quantize.png', dpi=400)
plt.show()

i_dark = 1

sorted_centroids = Kmeans_centroids.argsort()
index_cluster = sorted_centroids[i_dark]
index = (Kmeans.labels_ == index_cluster)
sample_ct_reshaped_D = sample_ct_reshaped * np.nan
sample_ct_reshaped_D[index] = 1
sample_ct_reshaped_D = sample_ct_reshaped_D.reshape(N_rows, N_cols)
plt.figure()
plt.imshow(sample_ct_reshaped_D, interpolation="nearest")
plt.title('Image Used to Identify Lungs')
plt.savefig('identify.png', dpi=400)
plt.show()

# DBSCAN to find the lungs in the image

epsilon = 2
min_samples = 5

C, Centroids, Clust = MyDBSCAN(sample_ct_reshaped_D, np.argwhere(sample_ct_reshaped_D == 1), epsilon, min_samples)

if Centroids[1,1] < Centroids[0,1]:
    print('Swap')
    temp = C[:,:,0] * 1
    C[:,:,0] = C[:,:,1] * 1
    C[:,:,1] = temp
    temp = Centroids[0,:] * 1
    Centroids[0,:] = Centroids[1,:] * 1
    Centroids[1,:] = temp
    
left_Lung = C[:,:,0].copy()  
right_Lung = C[:,:,1].copy()  

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(left_Lung, interpolation="nearest")
ax1.set_title('Left Lung Mask - Initial')
ax2.imshow(right_Lung, interpolation="nearest")
ax2.set_title('Right Lung Mask - Initial')
plt.savefig('right_left_lung_initial.png', dpi=400)
plt.show()


# Generate a new image with the two darkest colors of the color-quantized image

sample_ct_reshaped_D = sample_ct_reshaped * np.nan
indexes = Kmeans_centroids.argsort()
index = (Kmeans.labels_ == indexes[0])
sample_ct_reshaped_D[index] = 1
index = (Kmeans.labels_ == indexes[1])
sample_ct_reshaped_D[index] = 1
sample_ct_reshaped_D = sample_ct_reshaped_D.reshape(N_rows, N_cols)

C, Centroids2, clust = MyDBSCAN(sample_ct_reshaped_D, np.argwhere(sample_ct_reshaped_D == 1), epsilon, min_samples)
index = np.argwhere(Centroids2[:,4]<1000)
Centroids2 = np.delete(Centroids2, index, axis=0)
dist_left = np.sum((Centroids[0,0:2]-Centroids2[:,0:2])**2, axis=1)    
dist_right = np.sum((Centroids[1,0:2]-Centroids2[:,0:2])**2, axis=1)    
index_left = dist_left.argmin()
index_right = dist_right.argmin() 
left_lungmask = C[:,:,index_left].copy()
right_lungmask = C[:,:,index_right].copy()
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(left_lungmask, interpolation="nearest")
ax1.set_title('Left Lung mask - Improvement')
ax2.imshow(right_lungmask, interpolation="nearest")
ax2.set_title('Right Lung Mask - Improvement')
plt.savefig('right_left_lung_improve.png', dpi=400)
plt.show()



#%% Final lung masks

epsilon = 1
min_samples = 5

C, Centroids3, clust = MyDBSCAN(left_lungmask, np.argwhere(np.isnan(left_lungmask)), epsilon, min_samples)
left_lungmask = np.ones((N_rows, N_cols))
left_lungmask[C[:,:,0]==1] = np.nan

C, Centroids3, clust = MyDBSCAN(right_lungmask, np.argwhere(np.isnan(right_lungmask)), epsilon, min_samples)
right_lungmask = np.ones((N_rows, N_cols))
right_lungmask[C[:,:,0]==1]=np.nan

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(left_lungmask, interpolation="nearest")
ax1.set_title('Left Lung Mask')
ax2.imshow(right_lungmask, interpolation="nearest")
ax2.set_title('Right Lung Mask')
plt.savefig('right_left_lung_final.png', dpi=400)
plt.show()




fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(left_lungmask*sample_ct, vmin=Vmin, vmax=Vmax, cmap='bone', interpolation="nearest")
ax1.set_title('Left Lung')
ax2.imshow(right_lungmask*sample_ct, vmin=Vmin, vmax=Vmax, cmap='bone', interpolation="nearest")
ax2.set_title('Right Lung')
plt.savefig('right_left_lung_bone.png', dpi=400)
plt.show()

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(left_lungmask*sample_ct, interpolation="nearest")
ax1.set_title('Left Lung')
ax2.imshow(right_lungmask*sample_ct, interpolation="nearest")
ax2.set_title('Right Lung')
plt.savefig('right_left_lung_colormap.png', dpi=400)
plt.show()

# Find ground glass opacities

left_lungmask[np.isnan(left_lungmask)] = 0
right_lungmask[np.isnan(right_lungmask)] = 0
lungs_mask = left_lungmask + right_lungmask
          
low_val = -700
high_val = -350
opa_threshold = 0.11
kernel = 20
color_map = 'spring'

B = lungs_mask*sample_ct
inf_mask = 1*(B>low_val)&(B<high_val)
infection_mask = filterImage(inf_mask, kernel)
infection_mask = 1.0*(infection_mask>opa_threshold)
infection_mask[infection_mask==0] = np.nan
plt.figure()
plt.imshow(infection_mask, alpha=0.5, cmap=color_map, interpolation="nearest")
plt.title('Infection Mask')
plt.savefig('Infection Mask.png', dpi=400)
plt.show()

ground = np.nan_to_num(sample_infection.copy())
pred = np.nan_to_num(infection_mask.copy()) 

plt.figure()
plt.imshow(sample_ct, alpha=1, vmin=Vmin, vmax=Vmax, cmap='bone')
plt.imshow(ground*255, alpha=0.3, vmin=0, vmax=255, cmap='winter', interpolation="nearest")
plt.imshow(infection_mask*255, alpha=0.5, vmin=0, vmax=255, cmap=color_map, interpolation="nearest")
plt.savefig('Original Image With Ground Glass Opacities in Yellow.png', dpi=400)
plt.show()


TP = []
TN = []
FP = []
FN = []


for i in range(ground.shape[0]):
    for j in range(ground.shape[1]):
        if ground[i,j]==1 and pred[i,j]==1:
            TP.append((i,j))
        if ground[i,j]==1 and pred[i,j]==0:
            FN.append((i,j))            
        if ground[i,j]==0 and pred[i,j]==1:
            FP.append((i,j)) 
        if ground[i,j]==0 and pred[i,j]==0:
            TN.append((i,j))

sensitivity = len(TP)/(len(TP)+len(FN))
specificity = len(TN)/(len(TN)+len(FP))
TPR = len(TP)/(len(TP)+len(FN))
FPR = len(FP)/(len(FP)+len(TN))
accuracy = (len(TP)+len(TN))/(len(TP)+len(TN)+len(FP)+len(FN))
f1_score = len(TP)/(len(TP)+1/2*(len(FP)+len(FN)))

dice = np.sum(pred[ground==1])*2 / (np.sum(pred) + np.sum(ground))

print('Dice Similarity Score is {}'.format(dice))

print('Sensitivity:', sensitivity)
print('Specificity:', specificity)
print('TPR:', TPR)
print('FPR:',FPR)
print('Accuracy:', accuracy)
print('F1 Score:', f1_score)

infected_pixels = []
for i in range(lungs_mask.shape[0]):
    for j in range(lungs_mask.shape[1]):
        if lungs_mask[i,j]==1 and pred[i,j]==0:
            infected_pixels.append((i,j))
        if lungs_mask[i,j]==1 and pred[i,j]==1:
            infected_pixels.append((i,j))
            
infected_left=[]
infected_right=[]

for i in range(lungs_mask.shape[0]):
    for j in range(lungs_mask.shape[1]):
        if left_lungmask[i,j]==1 and pred[i,j]==0:
            infected_left.append((i,j))
        if left_lungmask[i,j]==1 and pred[i,j]==1:
            infected_left.append((i,j))
        if right_lungmask[i,j]==1 and pred[i,j]==0:
            infected_right.append((i,j))
        if right_lungmask[i,j]==1 and pred[i,j]==1:
            infected_right.append((i,j))
            
            
percent_infected = len(infected_pixels)*100/(lungs_mask.shape[0]*lungs_mask.shape[1])
print(f"Through the scan, About {int(percent_infected)}% of patient's lungs have been infected with Covid-19 interstitial pneumonia. \n")

if infected_left<infected_right:
    winsound.PlaySound('sub/left_lung.wav', winsound.SND_FILENAME)
    print("Most of the Covid-19 interstitial pneumonia, have been observed through the left lung! \n")
else:
    winsound.PlaySound('sub/right_lung.wav', winsound.SND_FILENAME)
    print("Most of the Covid-19 interstitial pneumonia, have been observed through the right lung! \n")

if percent_infected>=50:
    winsound.PlaySound('sub/danger.wav', winsound.SND_FILENAME)
    print("Unfortunately, patient is in a dangerous condition! Please follow the next clinical protocols as soon as possible!")
    winsound.PlaySound('sub/danger_voice.wav', winsound.SND_FILENAME)
else:
    winsound.PlaySound('sub/safe.wav', winsound.SND_FILENAME)
    print("Fortunately, patient is in a safe condition! But as long as Covid-19 has been detected through the investiations, patient needs to be quarantined! Please follow the next clinical protocols as soon as possible!")
    winsound.PlaySound('sub/safe_voice.wav', winsound.SND_FILENAME)