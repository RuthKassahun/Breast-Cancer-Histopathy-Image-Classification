import os

import numpy as np
import pandas as pd
from PIL import Image
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.models import sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score,accuracy_score
import scipy
from tqdm import tqdm
import tensorflow as tf
from keras import backend as k
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools


def Dataset_loader(DIR,RESIZE,sigmax=10):
    IMG = []
    read = lambda imname:np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
            img = cv2.resize(img,(RESIZE,RESIZE))
            IMG.append(np.array(img))
        return IMG





