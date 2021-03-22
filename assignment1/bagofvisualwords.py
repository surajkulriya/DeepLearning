import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

#returns paths of individual images and corresponding class
def paths_classes(path):
    image_paths = []
    image_class = []
    
    class_names = os.listdir(path)

    for i in class_names:
        dir_ = os.path.join(path,i)
        for j in range(len(os.listdir(dir_))):
            image_paths.append(path+ "/" + i + "/"+os.listdir(dir_)[j])
            image_class.append(i)
    return image_paths, image_class

#Makes patches of size patch_size*patch_size for a image
def patches(image, patch_size):
    
    patches = []
    h,w = image.shape[0], image.shape[1]
    h_q, w_q = h//patch_size, w//patch_size
    h_r, w_r = h%patch_size, w%patch_size
    if h_r==0 and w_r==0:
        for i in range(h_q):
            for j in range(w_q):
                patches.append(image[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size])
    elif h_r!=0 and w_r==0:
        for i in range(h_q+1):
            for j in range(w_q):
                if i != h_q:
                    patches.append(image[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size])
                else:
                    x = image[i*patch_size:h,j*patch_size:(j+1)*patch_size]
                    y = image[h-patch_size:i*patch_size,j*patch_size:(j+1)*patch_size]
                    patches.append(cv2.vconcat([x,y]))
    elif h_r==0 and w_r!=0:
        for i in range(h_q):
            for j in range(w_q+1):
                if j != w_q:
                    patches.append(image[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size])
                else:
                    x = image[i*patch_size:(i+1)*patch_size,j*patch_size:w]
                    y = image[i*patch_size:(i+1)*patch_size,w-patch_size:j*patch_size]
                    patches.append(cv2.hconcat([x,y]))
    else:
        for i in range(h_q+1):
            for j in range(w_q+1):
                if i!=h_q and j!=w_q:
                    patches.append(image[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size])
                elif i==h_q and j!=w_q:
                    x = image[i*patch_size:h,j*patch_size:(j+1)*patch_size]
                    y = image[h-patch_size:i*patch_size,j*patch_size:(j+1)*patch_size]
                    patches.append(cv2.vconcat([x,y]))
                elif i!=h_q and j==w_q:
                    x = image[i*patch_size:(i+1)*patch_size,j*patch_size:w]
                    y = image[i*patch_size:(i+1)*patch_size,w-patch_size:j*patch_size]
                    patches.append(cv2.hconcat([x,y]))
                
                else:
                    x = image[i*patch_size:h,j*patch_size:w]
                    y = image[h-patch_size:i*patch_size,j*patch_size:w]
                    z = cv2.vconcat([x,y])
                    a = image[(i-1)*patch_size:i*patch_size,w-patch_size:j*patch_size]
                    patches.append(cv2.hconcat([z,a]))
                
    return patches

#Makes a colour histogram in 8 bins and concatenates the R,G,B channels
def colour_histogram(patch, patch_size):
    
    h = [0 for i in range(24)]
    for i in range(patch_size):
        for j in range(patch_size):
            h[patch[i][j][0]//32] += 1
            h[8+patch[i][j][1]//32] += 1
            h[16+patch[i][j][2]//32] += 1
    return h
#Takes in image paths and patch size and returns histogram of patches of every image
def hist_image_patch(image_paths, patch_size):
    images = []
    for i in image_paths:
        images.append(cv2.imread(i))
    image_patches = []
    for i in images:
        image_patches.append(patches(i,patch_size))
    hist_image_patches = []
    for image in image_patches:
        hist_patches = []
        for patch in image:
            hist_patches.append(colour_histogram(patch,patch_size))
        hist_image_patches.append(hist_patches)
        #print(len(hist_image_patches))

    return hist_image_patches
#Takes in the training image paths and patch size and returns the kmeans model trained on the data and histogram of patches of every image
def cluster_train(image_paths_train, patch_size):
    hist_image_patches_train = hist_image_patch(image_paths_train, patch_size)
    hist_image_patches_cluster_train = []
    for image in hist_image_patches_train:
        for patch in image:
            hist_image_patches_cluster_train.append(patch)
    kmeans = KMeans(n_clusters=32).fit(hist_image_patches_cluster_train)
    return kmeans, hist_image_patches_train
#Gives the bag of visual words representation by taking in every image and the kmeans model 
def bovw(image_patch, kmeans):
    classes = kmeans.predict(image_patch)
    num_patch = len(image_patch)
    bovw_pre = [0 for i in range(32)]
    for i in classes:
        bovw_pre[i] += 1
    bovw = [i/num_patch for i in bovw_pre]
    return bovw

if (__name__ == "__main__"):
    #Replace the paths
    path_tv = "D:/Academics/Semester 4/CS 671 Deep Learning and Applications/Labs/Lab1/Group23/Classification/Image_Group23/train"
    path_test = "D:/Academics/Semester 4/CS 671 Deep Learning and Applications/Labs/Lab1/Group23/Classification/Image_Group23/test"

    image_paths_tv, image_class_tv = paths_classes(path_tv)
    image_paths_train, image_paths_valid, image_class_train, image_class_valid = train_test_split(image_paths_tv, image_class_tv, test_size = 0.2)
    image_paths_test, image_class_test = paths_classes(path_test)
    
    
    kmeans, hist_image_patches_train = cluster_train(image_paths_train, 32)
    hist_image_patches_valid = hist_image_patch(image_paths_valid, 32)
    hist_image_patches_test = hist_image_patch(image_paths_test, 32)

    bovw_train = [bovw(i,kmeans) for i in hist_image_patches_train]
    bovw_valid = [bovw(i,kmeans) for i in hist_image_patches_valid]
    bovw_test = [bovw(i,kmeans) for i in hist_image_patches_test]

    #Converting the data to a dataframe
    train_df = pd.DataFrame(bovw_train, columns=[i for i in range(1,33)])
    valid_df = pd.DataFrame(bovw_valid, columns=[i for i in range(1,33)])
    test_df = pd.DataFrame(bovw_test, columns=[i for i in range(1,33)])
    #Added the labels of each example to the respective dataframes
    train_df["label"] = image_class_train
    valid_df["label"] = image_class_valid
    test_df["label"] = image_class_test
    #Saved the data to csv files
    train_df.to_csv("image_train.csv")
    valid_df.to_csv("image_valid.csv")
    test_df.to_csv("image_test.csv")