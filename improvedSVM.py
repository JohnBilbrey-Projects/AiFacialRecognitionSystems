import cv2
import numpy as np
import os
import glob
import random
from PIL import Image
from sklearn.svm import SVC
import joblib


#caching directories for improved SVM

CACHE_DIR = 'improved_SVM_cache'
DOWNSAMPLE_RATIO = 0.2
FEATURES_CACHE = os.path.join(CACHE_DIR, f'features_{int(DOWNSAMPLE_RATIO*100)}pct.npz')
MODEL_CACHE = os.path.join(CACHE_DIR, f'svm_model_{int(DOWNSAMPLE_RATIO*100)}pct.joblib')
ACC_CACHE = os.path.join(CACHE_DIR, f'accuracy_{int(DOWNSAMPLE_RATIO*100)}pctDownsample.npy')


#function to compute HOG features for a image
def computeHog(image, cellSize=(8,8), blockSize=(2,2), bins=9):
    #compute gradients using Sobel filters (converted to float32 for precision)
    image = image.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    #compute gradient magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = (np.arctan2(gy, gx) * (180 / np.pi)) % 180
    
    height, width = image.shape
    cell_h, cell_w = cellSize
    n_cells_y = height // cell_h
    n_cells_x = width // cell_w
    
    #initialize the histogram for each cell
    orientationHistogram = np.zeros((n_cells_y, n_cells_x, bins))
    binSize = 180 / bins
    
    #loop through each cell to compute the histogram of gradients
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cellMag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cellAngle = angle[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            hist = np.zeros(bins)
            #for each pixel in the cell, vote into the appropriate bin
            for y in range(cellMag.shape[0]):
                for x in range(cellMag.shape[1]):
                    a = cellAngle[y, x]
                    m = cellMag[y, x]
                    binIdx = int(a // binSize) % bins
                    hist[binIdx] += m
            orientationHistogram[i, j, :] = hist

    #group cells into blocks and perform L2 normalization
    block_h, block_w = blockSize
    hogVector = []
    for i in range(n_cells_y - block_h + 1):
        for j in range(n_cells_x - block_w + 1):
            block = orientationHistogram[i:i+block_h, j:j+block_w, :].ravel()
            norm = np.linalg.norm(block) + 1e-6  #prevent division by zero by adding very small value to denominator, as to not strongly affect outcome
            block = block / norm
            hogVector.extend(block)
    
    return np.array(hogVector)


#helper function to extract random negative patch from a sample image
#since there is no directory for negatives in the dataset, we will use these as our negatives.
def extractRandomNegativePatch(image, patchSize, bboxes, maxTrials=50):
    h, w = image.shape
    ph, pw = patchSize
    for _ in range(maxTrials):
        y = random.randint(0, h - ph)
        x = random.randint(0, w - pw)
        #define patch rect
        rect = (x, y, x+pw, y+ph)
        overlap = False
        for (bx1, by1, bx2, by2) in bboxes:
            ix1 = max(x, bx1)
            iy1 = max(y, by1)
            ix2 = min(x+pw, bx2)
            iy2 = min(y+ph, by2)
            if ix2 > ix1 and iy2 > iy1:
                overlap = True
                break
        if not overlap:
            return image[y:y+ph, x:x+pw]
    return None

# load positive face patches from YOLO annotations
def loadPositives(faces_base, split, windowSize):
    
    images_dir = os.path.join(faces_base, 'images', split)
    labels_dir = os.path.join(faces_base, 'labels', split)
    positives = []
    for imgPath in glob.glob(os.path.join(images_dir, '*.jpg')):
        img = Image.open(imgPath).convert('L')
        img_np = np.array(img)
        h, w = img_np.shape
        labelFile = os.path.join(labels_dir, os.path.splitext(os.path.basename(imgPath))[0] + '.txt')
        if not os.path.exists(labelFile):
            continue
        with open(labelFile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, xc, yc, ws, hs = parts
                xc, yc, ws, hs = map(float, (xc, yc, ws, hs))
                cx = int(xc * w)
                cy = int(yc * h)
                pw = int(ws * w)
                ph = int(hs * h)
                x1 = max(cx - pw // 2, 0)
                y1 = max(cy - ph // 2, 0)
                x2 = min(x1 + pw, w)
                y2 = min(y1 + ph, h)
                face_patch = img_np[y1:y2, x1:x2]
                if face_patch.size == 0:
                    continue
                positives.append(cv2.resize(face_patch, windowSize))
    return positives

#load negative patches by sampling around faces
def loadNegativesAroundFaces(faces_base, split, windowSize, numPatches=5):
    images_dir = os.path.join(faces_base, 'images', split)
    labels_dir = os.path.join(faces_base, 'labels', split)
    negatives = []
    for imgPath in glob.glob(os.path.join(images_dir, '*.jpg')):
        img = Image.open(imgPath).convert('L')
        img_np = np.array(img)
        h, w = img_np.shape
        #collect bboxes
        bboxes = []
        labelFile = os.path.join(labels_dir, os.path.splitext(os.path.basename(imgPath))[0] + '.txt')
        if os.path.exists(labelFile):
            with open(labelFile) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    _, xc, yc, ws, hs = parts
                    xc, yc, ws, hs = map(float, (xc, yc, ws, hs))
                    cx = int(xc * w)
                    cy = int(yc * h)
                    pw = int(ws * w)
                    ph = int(hs * h)
                    x1 = max(cx - pw // 2, 0)
                    y1 = max(cy - ph // 2, 0)
                    bboxes.append((x1, y1, x1 + pw, y1 + ph))
        for _ in range(numPatches):
            patch = extractRandomNegativePatch(img_np, windowSize, bboxes)
            if patch is not None:
                negatives.append(cv2.resize(patch, windowSize))
    return negatives



#function to perform sliding window detector over image
#for each window, compute hog features and use classifier to score it. record detections w a positive score
def slidingWindowDetector(image, clf, windowSize, detectionThreshold, stepSize=8, scaleFactors=[1.0, 1.25, 1.5]):
    detections = []
    for scale in scaleFactors:
        #create window by making resized image for each scale
        resized = cv2.resize(image, (int(image.shape[1] / scale), int(image.shape[0] / scale)))
        for y in range(0, resized.shape[0] - windowSize[1], stepSize):
            for x in range(0, resized.shape[1] - windowSize[0], stepSize):
                window = resized[y:y+windowSize[1], x:x+windowSize[0]]
                if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
                    continue
                #compute hog features and give score
                feat = computeHog(window)
                feat = feat.reshape(1, -1)
                score = clf.decision_function(feat)
                if score > detectionThreshold:  #detection threshold (this value seems to work well)
                    #map coordinates back to the original image scale
                    rx = int(x * scale)
                    ry = int(y * scale)
                    rw = int(windowSize[0] * scale)
                    rh = int(windowSize[1] * scale)
                    detections.append((rx, ry, rw, rh, score[0]))
    return detections

#function to apply nonmaximum suppresiion
def nonMaximumSuppression(detections, overlapThresh=0.3):
    if len(detections) == 0:
        return []
    
    #convert detections to an array: [x1, y1, x2, y2, score]
    boxes = np.array([[x, y, x+w, y+h, score] for (x, y, w, h, score) in detections])
    idxs = np.argsort(boxes[:, 4])[::-1]  #sort by score (high to low)
    keep = []
    
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(boxes[i])
        xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / ((boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1]))
        
        idxs = idxs[np.where(overlap <= overlapThresh)[0] + 1]
    
    finalBoxes = [(int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1]), b[4]) for b in keep]
    return finalBoxes

#function to compute accuracy score for testing
def accuracyScore(y_true, y_pred):
    #ensure both input lists have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch: y_true and y_pred must have the same number of elements.")

    #count the number of correct predictions
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    #return accuracy score (correct predictions / total samples)
    return correct / len(y_true)


#function to split the dataset into training and test sets
def splitTrainTest(X, y, test_size=0.25, random_state=None, stratify=None):
    #ensure X and y are numpy arrays
    X = np.array(X)
    y = np.array(y)
    n = len(X)
    
    if stratify is None:
        #no stratification, so generate a random permutation of indices.
        rng = np.random.RandomState(random_state)
        indices = np.arange(n)
        rng.shuffle(indices)
        test_n = int(round(n * test_size))
        
        test_idx = indices[:test_n]
        train_idx = indices[test_n:]
        
    else:
        #for each unique label, split the indices in proportion
        rng = np.random.RandomState(random_state)
        unique_labels = np.unique(stratify)
        train_idx = []
        test_idx = []
        
        for label in unique_labels:
            label_indices = np.where(np.array(stratify) == label)[0]
            rng.shuffle(label_indices)
            test_n_label = int(round(len(label_indices) * test_size))
            test_idx.extend(label_indices[:test_n_label])
            train_idx.extend(label_indices[test_n_label:])
            
        #sort indices to preserve order
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    return X_train, X_test, y_train, y_test

#function to perform hard negative mining by applying shifts to negative patches, calculate score using HOG of each shifted patch, and return negatives being classified as positive
def hardNegativeMining(negPatches, clf, shifts=[-4, 0, 4]):
    #for each negative patch, calculate score and if it is being classified as positive, add it to hardNegatives array
    hardNegatives = []
    for patch in negPatches:
        #we will slightly shift the patch in both directions to try and capture more potential false positives
        for dx in shifts:
            for dy in shifts:
                #use warpAffine function in cv2 to create shifted patches
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shiftedPatch = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]))
                #calculate score for each shifted patch, as well as original patch (because one of the potential shifts is no shift)
                feat = computeHog(shiftedPatch)
                score = clf.decision_function(feat.reshape(1, -1))
                if score > 0.0:  #if misclassified as positive
                    hardNegatives.append(shiftedPatch)
    return hardNegatives



def main():
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # delete any feature cache not matching this ratio
    for old in glob.glob(os.path.join(CACHE_DIR, 'features_*pct.npz')):
        if old != FEATURES_CACHE:
            os.remove(old)
    #same for model cache
    for old in glob.glob(os.path.join(CACHE_DIR, 'svm_model_*pct.joblib')):
        if old != MODEL_CACHE:
            os.remove(old)
    #same for accuracy cache
    for old in glob.glob(os.path.join(CACHE_DIR, 'accuracy_*pctDownsample.npy')):
        if old != ACC_CACHE:
            os.remove(old)
    
    #set detection threshold default value
    #can be adjusted w/o reaching
    #patches in sliding window detector w/ a calculated score < this value will not be recognized as faces.
    #2.0 is typically very high, but there are still false positives being drawn in the demo
    detectionThreshold = 2.0
    #set accuracy after hard mining to -1.0 initially, in rare case that no hard negative
    #patches are found
    accAug = -1.0
    
    faces_dir = 'faces'
    
    #this variable requires recaching if you change it, but could improve the detections drawn in the demo
    #significantly if you can find the right values. I think currently this window size of 64x64 may be too small. 
    #It is responsible for determining the size of patches from images used in training/testing,
    #and also the base scale for the sliding window detector
    windowSize = (64, 64)
    resizeDim = windowSize

    
    
    
    #extract HOG features and assign labels (1: face, 0: background)
    #if already cahced, retrieve
    if os.path.exists(FEATURES_CACHE):
        print("Loading cached features...")
        data = np.load(FEATURES_CACHE)
        X, y = data['X'], data['y']
    #if not, compute them
    else:
        print("Loading positive face patches...")
        positiveImages = loadPositives(faces_dir, 'train', resizeDim)
        print(f"Loaded {len(positiveImages)} positive patches.")

        print("Sampling negative patches around faces...")
        negativeImages = loadNegativesAroundFaces(faces_dir, 'train', resizeDim, numPatches=5)
        print(f"Loaded {len(negativeImages)} negative patches.")
        
        #downsample dataset
        print(f"Downsampling dataset to specified size ({DOWNSAMPLE_RATIO})")
        N_pos = len(positiveImages)
        N_neg = len(negativeImages)
        keep_pos = set(random.sample(range(N_pos), k=int(DOWNSAMPLE_RATIO*N_pos)))
        keep_neg = set(random.sample(range(N_neg), k=int(DOWNSAMPLE_RATIO*N_neg)))
        positiveImages = [positiveImages[i] for i in keep_pos]
        negativeImages = [negativeImages[i] for i in keep_neg]
        totalDownsampledSize = len(positiveImages) + len(negativeImages)
        print(f"Downsampled to {totalDownsampledSize} total images.")
        
        #compute hog features on downsampled dataset
        print("Computing HOG features...")
        X_list, y_list = [], []
        for img in positiveImages:
            X_list.append(computeHog(img)); y_list.append(1)
        for img in negativeImages:
            X_list.append(computeHog(img)); y_list.append(0)
        X, y = np.array(X_list), np.array(y_list)
        print("Caching features to", FEATURES_CACHE)
        np.savez(FEATURES_CACHE, X=X, y=y)
    
    
    #if cached SVM exists, retrieve it
    if os.path.exists(MODEL_CACHE) and os.path.exists(ACC_CACHE):
        print("Loading cached SVM model...")
        clf = joblib.load(MODEL_CACHE)
        acc = np.load(ACC_CACHE)
        print(f"Cached test accuracy: {acc*100:.2f}%")
    #train one if not
    else:
        #split the dataset (75% train, 25% test)
        print("splitting dataset and training SVM classifier...")
        X_train, X_test, y_train, y_test = splitTrainTest(X, y, test_size=0.25,
                                                        random_state=42, stratify=y)
        
        #train classifier
        clf = SVC(kernel="linear", probability=True)
        clf.fit(X_train, y_train)
        
    
        #evaluate on the test set
        yPred = clf.predict(X_test)
        acc = accuracyScore(y_test, yPred)
        print("Test Accuracy Without Hard Negative Mining: {:.2f}%".format(acc * 100))
        
        print("Performing Hard Negative Mining...")
        #run hard negative mining on the negative images
        hardNegPatches = hardNegativeMining(negativeImages, clf)
        print("Extracted {} hard negative patches.".format(len(hardNegPatches)))
        
        #if any hard negatives were found, update the training set
        if hardNegPatches:
            print("Retraining model and caching...")
            X_hard = [computeHog(patch) for patch in hardNegPatches]
            y_hard = [0] * len(X_hard)
            #augment the original training set with the hard negatives.
            X_train_aug = np.concatenate((X_train, np.array(X_hard)), axis=0)
            y_train_aug = np.concatenate((y_train, np.array(y_hard)), axis=0)
            
            #retrain the classifier with the augmented training set and cache model and accuracy
            clf = SVC(kernel="linear", probability=True)
            clf.fit(X_train_aug, y_train_aug)
            yPredAug = clf.predict(X_test)
            joblib.dump(clf, MODEL_CACHE)
            accAug = accuracyScore(y_test, yPredAug)
            print("Test Accuracy after Hard Negative Mining: {:.2f}%".format(accAug * 100))
            np.save(ACC_CACHE, accAug)
        else:
            #cache accuracy without HNM in rare case no hard negatives are found
            print("No hard negatives were found")
            np.save(ACC_CACHE, acc)
            
        #if accuracy has improved after running hard negative mining, tell the user
        if acc < accAug: 
            print("Hard Negative Mining has improved accuracy")
        else: 
            print("Hard Negative Mining has not improved accuracy")
        
    
    
    
    #DEMO
    
    
    
    #perform a demonstration on a random image
    print("Choosing random images for demonstration...")
    
    imgPaths = glob.glob(os.path.join(faces_dir, 'images', 'train', '*.jpg'))
    
    posDemoPath = random.choice(imgPaths)
    
    #use gray image to obtain windows w detections, then draw them on color image
    posDemoColor = cv2.imread(posDemoPath, cv2.IMREAD_COLOR)
    posDemoGray  = cv2.cvtColor(posDemoColor, cv2.COLOR_BGR2GRAY)
    demoImages = [(posDemoColor, posDemoGray)]

    print("Image Chosen.")
    

    for colorImg, grayImg in demoImages:
        print("applying sliding window detector...")
        #step size and scaleFactors can be adjusted w/o recaching
        detections = slidingWindowDetector(
            grayImg, clf, windowSize, detectionThreshold,
            stepSize=32, scaleFactors=[1.0, 2.5, 5.0, 10.0, 15.0]
        )
        #overlapTresh can be adjusted w/o recaching 
        print("Applying non-maximum suppression...")
        finalDetections = nonMaximumSuppression(detections, overlapThresh=0.3)

        print("drawing detections on image...")
        #draw on the true color image
        for (x, y, w, h, score) in finalDetections:
            cv2.rectangle(colorImg, (x, y), (x+w, y+h), (0, 255, 0), 2)

        title = "Face Detected" if finalDetections else "No Face Detected"
        cv2.imshow(title, colorImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()