import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from classificationClasses import oneHiddenLayer
from classificationClasses import getY


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

def bin(colour):
    return colour//32

def colour_histogram(patch, patch_size):
    
    r = np.zeros(8)
    g = np.zeros(8)
    b = np.zeros(8)
    for i in range(patch_size):
        for j in range(patch_size):
            r[bin(patch[i][j][0])] += 1
            g[bin(patch[i][j][1])] += 1
            b[bin(patch[i][j][2])] += 1
    return np.concatenate((r,g,b))

def hist_image_patch(image_paths, patch_size):
    images = []
    for i in image_paths_train:
        images.append(cv2.imread(i))
    image_patches = []
    for i in images:
        image_patches.append(patches(i,32))
    hist_image_patches = []
    for image in image_patches:
        for patch in image:
            hist_image_patches.append(colour_histogram(patch,32))
    return hist_image_patches

def cluster_train(image_paths_train, patch_size):
    hist_image_patches_train = hist_image_patch(image_paths_train, patch_size)
    kmeans = KMeans(n_cluster=32).fit(hist_image_patches_train)
    return kmeans, hist_image_patches_train
    
def bovw(image_patch, kmeans):
    classes = kmeans.predict(image_patch)
    bovw = np.zeros(32)
    for i in classes:
        bovw[i] += 1
    return bovw

if (__name__ == "__main__"):
    path_tv = "D:/Academics/Semester 4/CS 671 Deep Learning and Applications/Labs/Lab1/Group23/Classification/Image_Group23/train"
    path_test = "D:/Academics/Semester 4/CS 671 Deep Learning and Applications/Labs/Lab1/Group23/Classification/Image_Group23/test"
    
    image_paths_tv, image_class_tv = paths_classes(path_tv)
    image_paths_test, image_class_test = paths_classes(path_test)

    image_paths_train, image_paths_valid, image_class_train, image_class_valid = train_test_split(image_paths_tv, image_class_tv, test_size = 0.2)
    
    kmeans, hist_image_patches_train= cluster_train(image_paths_train, 32)
    hist_image_patches_valid = hist_image_patch(image_paths_valid, 32)
    hist_image_patches_test = hist_image_patch(image_paths_test, 32)

    bovw_train = [bovw(i,kmeans) for i in hist_image_patches_train]
    bovw_valid = [bovw(i,kmeans) for i in hist_image_patches_valid]
    bovw_test = [bovw(i,kmeans) for i in hist_image_patches_test]

    class_dict = {"bayou":0,"desert_vegetation":1,"music_store":2}
    image_class_train = [class_dict[i] for i in image_class_train]
    image_class_valid = [class_dict[i] for i in image_class_valid]
    image_class_test = [class_dict[i] for i in image_class_test]

    image_class_train_1hot = getY(image_class_train)

    model = oneHiddenLayer(32,3,1,5)
    model.train(bovw_train,image_class_train_1hot,image_class_train, 1.0,0,bovw_test,image_class_test,bovw_valid,image_class_valid)



