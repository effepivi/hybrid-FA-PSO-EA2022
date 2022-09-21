import math
import copy;

import numpy as np
from numpy import linalg as LA

from sklearn.utils import check_array
import skimage.measure as measure;
import skimage.metrics as metrics;
import skimage.exposure as exposure;
from skimage import filters


MINIMISATION=[];
MINIMISATION.append("SAE");
MINIMISATION.append("SSE");
MINIMISATION.append("MAE");
MINIMISATION.append("MSE");
MINIMISATION.append("RMSE");
MINIMISATION.append("NRMSE_euclidean");
MINIMISATION.append("NRMSE_mean");
MINIMISATION.append("NRMSE_min_max");
MINIMISATION.append("mean_relative_error");
MINIMISATION.append("max_relative_error");
MINIMISATION=set(MINIMISATION);

MAXIMISATION=[];
MAXIMISATION.append("cosine_similarity");
MAXIMISATION.append("SSIM");
MAXIMISATION.append("PSNR");
MAXIMISATION.append("ZNCC");
MAXIMISATION=set(MAXIMISATION);


def getEntropy(anImage):
    grayImg = (linearNormalisation(anImage, 0, 255)).astype(np.uint8);
    return measure.shannon_entropy(grayImg);

def zeroMeanNormalisation(anImage):
    return (anImage - anImage.mean()) / (anImage.std());

def linearNormalisation(anImage, aMinValue = 0, aMaxValue = 1):
    return aMinValue + (aMaxValue - aMinValue) * (anImage - anImage.mean()) / (anImage.std());

def normalise(anImage):

    #return zeroMeanNormalisation(anImage);
    return linearNormalisation(anImage);
    #return copy.deepcopy(anImage);

def productImage(anImage1, anImage2):
    check_array(anImage1, anImage2);
    return (np.multiply(anImage1, anImage2));

def getHistogram(anImage, aNumberOfBins):
    return exposure.histogram(anImage, aNumberOfBins);

def getSAE(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return np.abs(np.subtract(aReferenceVector, aTestVector)).sum();

def getMAE(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return np.abs(np.subtract(aReferenceVector, aTestVector)).mean();

def getCosineSimilarity(aReferenceVector, aTestVector):

    check_array(aReferenceVector, aTestVector);

    u = aReferenceVector.flatten();
    v = aTestVector.flatten();

    return np.dot(u, v) / (LA.norm(u) * LA.norm(v))

def getMeanRelativeError(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return np.abs(np.divide(np.subtract(aReferenceVector, aTestVector), aReferenceVector)).mean();

def getMaxRelativeError(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return np.abs(np.divide(np.subtract(aReferenceVector, aTestVector), aReferenceVector)).max();

def getSSIM(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return metrics.structural_similarity( aReferenceVector, aTestVector);

def getSSE(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return np.square(np.subtract(aReferenceVector, aTestVector)).sum();

def getMSE(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return metrics.mean_squared_error( aReferenceVector, aTestVector);

def getRMSE(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return math.sqrt(getMSE(aReferenceVector, aTestVector));

def getNRMSE_euclidean(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return metrics.normalized_root_mse(aReferenceVector, aTestVector, normalization='euclidean');

def getNRMSE_mean(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return metrics.normalized_root_mse(aReferenceVector, aTestVector, normalization='mean');

def getNRMSE_minMax(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return metrics.normalized_root_mse(aReferenceVector, aTestVector, normalization='min-max');

def getPSNR(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return metrics.peak_signal_noise_ratio(aReferenceVector, aTestVector, data_range=aReferenceVector.max() - aReferenceVector.min());

def getNCC(aReferenceVector, aTestVector):
    check_array(aReferenceVector, aTestVector);
    return productImage(zeroMeanNormalisation(aReferenceVector), zeroMeanNormalisation(aTestVector)).mean();

def getTV(anImage):
    image_prewitt_h = filters.prewitt_h(anImage);
    image_prewitt_v = filters.prewitt_v(anImage);

    return np.abs(image_prewitt_h).mean() + np.abs(image_prewitt_v).mean();

def cropCenter(img, cropx, cropy):
   y, x = img.shape
   startx = x // 2 - (cropx // 2)
   starty = y // 2 - (cropy // 2)
   return img[starty:starty + cropy, startx:startx + cropx]
