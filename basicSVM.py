import cv2
import numpy as np
import os, glob, random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

#paths for caching
CACHE_DIR = 'basic_SVM_cache'
FEATURES_FILE = os.path.join(CACHE_DIR, 'features.npz')
MODEL_FILE = os.path.join(CACHE_DIR, 'svm_model.joblib')
ACC_FILE       = os.path.join(CACHE_DIR, 'accuracy.npy')

#feature extraction: mean shift smoothing and fixed grid raw pixel averages
def computeFeatures(patch, grid=(8,8)):
    filtered = cv2.pyrMeanShiftFiltering(patch, sp=10, sr=10)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gh, gw = grid
    cell_h, cell_w = h // gh, w // gw
    features = []
    for i in range(gh):
        for j in range(gw):
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            features.append(np.mean(cell))
    return np.array(features)

#function to load sample positive patches for demonstration. 
def samplePositivePatch(facesDir, split, windowSize):
    imagesDir = os.path.join(facesDir, 'images', split)
    labelsDir = os.path.join(facesDir, 'labels', split)
    imgPaths = glob.glob(os.path.join(imagesDir, '*.jpg'))
    while imgPaths:
        imgPath = random.choice(imgPaths)
        img = cv2.imread(imgPath)
        h, w = img.shape[:2]
        labelFile = os.path.join(labelsDir, os.path.splitext(os.path.basename(imgPath))[0] + '.txt')
        if not os.path.exists(labelFile): imgPaths.remove(imgPath); continue
        with open(labelFile) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                _, xc, yc, ws, hs = map(float, parts[:5])
                cx, cy = int(xc*w), int(yc*h)
                pw, ph = int(ws*w), int(hs*h)
                x1, y1 = max(cx-pw//2,0), max(cy-ph//2,0)
                x2, y2 = min(x1+pw, w), min(y1+ph, h)
                patch = img[y1:y2, x1:x2]
                if patch.size: return cv2.resize(patch, windowSize)
        imgPaths.remove(imgPath)
    raise RuntimeError('No positive patch')

#function to sample negative patches for demonstration.
def sampleNegativePatch(facesDir, split, windowSize):
    imagesDir = os.path.join(facesDir, 'images', split)
    imgPaths = glob.glob(os.path.join(imagesDir, '*.jpg'))
    while imgPaths:
        imgPath = random.choice(imgPaths)
        img = cv2.imread(imgPath)
        h, w = img.shape[:2]
        if h <= windowSize[1] or w <= windowSize[0]: imgPaths.remove(imgPath); continue
        y0 = random.randint(0, h-windowSize[1])
        x0 = random.randint(0, w-windowSize[0])
        patch = img[y0:y0+windowSize[1], x0:x0+windowSize[0]]
        if patch.shape[:2] == (windowSize[1], windowSize[0]): return patch
        imgPaths.remove(imgPath)
    raise RuntimeError('No negative patch')

if __name__ == '__main__':
    facesDir = 'faces'
    split = 'train'
    windowSize = (64, 128)
    negativesPerImg = 1

    os.makedirs(CACHE_DIR, exist_ok=True)

    #if features have already been computed, retireve them,
    if os.path.exists(FEATURES_FILE):
        print("Loading cached features...")
        data = np.load(FEATURES_FILE)
        X, y = data['X'], data['y']
    #if not, compute them
    else:
        X, y = [], []
        print("Computing features...")
        # positives
        for labelFile in glob.glob(os.path.join(facesDir, 'labels', split, '*.txt')):
            base = os.path.splitext(os.path.basename(labelFile))[0]
            imgPath = os.path.join(facesDir, 'images', split, base + '.jpg')
            if not os.path.exists(imgPath): continue
            img = cv2.imread(imgPath); h, w = img.shape[:2]
            with open(labelFile) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts)<5: continue
                    _, xc, yc, ws, hs = map(float, parts[:5])
                    cx, cy = int(xc*w), int(yc*h)
                    pw, ph = int(ws*w), int(hs*h)
                    patch = img[max(cy-ph//2,0):min(cy+ph//2,h), max(cx-pw//2,0):min(cx+pw//2,w)]
                    if patch.size:
                        feat = computeFeatures(cv2.resize(patch, windowSize))
                        X.append(feat); y.append(1)
        # negatives
        for imgPath in glob.glob(os.path.join(facesDir, 'images', split, '*.jpg')):
            img = cv2.imread(imgPath); h, w = img.shape[:2]
            for _ in range(negativesPerImg):
                if h<=windowSize[1] or w<=windowSize[0]: continue
                y0, x0 = random.randint(0, h-windowSize[1]), random.randint(0, w-windowSize[0])
                patch = img[y0:y0+windowSize[1], x0:x0+windowSize[0]]
                if patch.shape[:2] == (windowSize[1], windowSize[0]):
                    feat = computeFeatures(patch)
                    X.append(feat); y.append(0)
        X = np.array(X); y = np.array(y)
        np.savez(FEATURES_FILE, X=X, y=y)
        print(f"Features computed and saved: {len(y)} samples.")

    
    
    
    #if SVM has already been trained, retrieve it
    if os.path.exists(MODEL_FILE) and os.path.exists(ACC_FILE):
        print("Loading cached SVM model...")
        clf = joblib.load(MODEL_FILE)
        acc = float(np.load(ACC_FILE))
        print(f"Cached test accuracy: {acc*100:.2f}%")
    #if not, train it
    else:
        #downsample to downsample ratio of dataset to quicken runtime
        print("Downsampling dataset to 20%...")
        X, _, y, _ = train_test_split(
            np.array(X), np.array(y), train_size=0.1,
            random_state=42, stratify=y
        )
        print(f"Downsampled to {len(y)} samples total.")
        #train classifier
        print("Training SVM model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        clf = SVC(kernel='linear', max_iter=6000, tol=1e-3)
        clf.fit(X_train, y_train)
        #compute and cache accuracy
        acc = clf.score(X_test, y_test)
        np.save(ACC_FILE, acc)
        joblib.dump(clf, MODEL_FILE)
        print(f"Model trained. Test acc: {acc*100:.2f}%")

    #choose two images for demonstration (1 pos, 1 neg)
    print("Running demonstration...")
    demos = [
        (samplePositivePatch(facesDir, split, windowSize), 'True: Face'),
        (sampleNegativePatch(facesDir, split, windowSize), 'True: Non-face')
    ]
    for img, tl in demos:
        pred = clf.predict([computeFeatures(img)])[0]
        pl = 'Face' if pred==1 else 'Non-face'
        plt.figure(figsize=(4,4))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{tl}, Pred: {pl}")
        plt.axis('off')
        plt.show()
