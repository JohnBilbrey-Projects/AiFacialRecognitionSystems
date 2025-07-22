import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

#paths for caching
CACHE_DIR    = "cnn_cache"
MODEL_CACHE  = os.path.join(CACHE_DIR, "face_cnn.keras")
ACC_CACHE    = os.path.join(CACHE_DIR, "face_cnn_acc.npy")
os.makedirs(CACHE_DIR, exist_ok=True)

#adjustable parameters
#adjusting window size will require recaching
WINDOW_SIZE = (64, 64)
DETECTION_THRESHOLD = 0.75
#parameters for sliding window detector
STEP_SIZE = 32
SCALE_FACTORS =  [2.5, 5.0, 10.0]



#function to load positive face patches from YOLO annotations
def loadPositives(faces_dir, split="train", windowSize=WINDOW_SIZE):
    imgs = glob.glob(os.path.join(faces_dir, "images", split, "*.jpg"))
    patches = []
    for imgPath in imgs:
        gray = np.array(Image.open(imgPath).convert("L"))
        h, w = gray.shape
        labelFile = os.path.join(
            faces_dir, "labels", split,
            os.path.splitext(os.path.basename(imgPath))[0] + ".txt"
        )
        if not os.path.exists(labelFile):
            continue
        with open(labelFile) as f:
            for line in f:
                cls, xc, yc, ws, hs = line.split()
                xc, yc, ws, hs = map(float, (xc, yc, ws, hs))
                cx, cy = int(xc*w), int(yc*h)
                pw, ph = int(ws*w), int(hs*h)
                x1 = max(cx - pw//2, 0)
                y1 = max(cy - ph//2, 0)
                face = gray[y1:y1+ph, x1:x1+pw]
                if face.size:
                    face = cv2.resize(face, windowSize)
                    patches.append(face)
    return patches

#function to sample a negative patch from an image that does not overlap with positive face patches
def extractRandomNegativePatch(image, patchSize, bboxes, maxTrials=50):
    h, w = image.shape
    ph, pw = patchSize
    for _ in range(maxTrials):
        y = random.randint(0, h-ph)
        x = random.randint(0, w-pw)
        overlap = False
        for (bx1,by1,bx2,by2) in bboxes:
            if not (x+pw < bx1 or x > bx2 or y+ph < by1 or y > by2):
                overlap = True
                break
        if not overlap:
            return image[y:y+ph, x:x+pw]
    return None


#function to load negative samples
def loadNegatives(faces_dir, split="train", windowSize=WINDOW_SIZE, numPerImage=5):
    patches = []
    for imgPath in glob.glob(os.path.join(faces_dir, "images", split, "*.jpg")):
        gray = np.array(Image.open(imgPath).convert("L"))
        h, w = gray.shape
        # parse all bboxes so we avoid them
        bboxes = []
        labelFile = os.path.join(
            faces_dir, "labels", split,
            os.path.splitext(os.path.basename(imgPath))[0] + ".txt"
        )
        if os.path.exists(labelFile):
            with open(labelFile) as f:
                for line in f:
                    _, xc, yc, ws, hs = line.split()
                    xc, yc, ws, hs = map(float, (xc, yc, ws, hs))
                    cx, cy = int(xc*w), int(yc*h)
                    pw, ph = int(ws*w), int(hs*h)
                    x1, y1 = cx-pw//2, cy-ph//2
                    bboxes.append((x1, y1, x1+pw, y1+ph))
        for _ in range(numPerImage):
            patch = extractRandomNegativePatch(gray, windowSize, bboxes)
            if patch is not None:
                patches.append(cv2.resize(patch, windowSize))
    return patches

#function to build CNN classifier using keras
def buildModel(input_shape=(*WINDOW_SIZE, 1)):
    model = keras.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128,3, activation="relu"), layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def slidingWindowDetectorCNN(image, model, windowSize, thresh,
                            stepSize=STEP_SIZE, scaleFactors=SCALE_FACTORS):
    detections = []
    for scale in scaleFactors:
        #resize so that large faces become windowSize
        resized = cv2.resize(
            image,
            (int(image.shape[1]/scale), int(image.shape[0]/scale))
        )
        for y in range(0, resized.shape[0]-windowSize[1]+1, stepSize):
            for x in range(0, resized.shape[1]-windowSize[0]+1, stepSize):
                win = resized[y:y+windowSize[1], x:x+windowSize[0]]
                inp = win.astype("float32")/255.0
                inp = np.expand_dims(inp, axis=(0,-1))
                prob = model.predict(inp, verbose=0)[0,0]
                if prob > thresh:
                    rx = int(x*scale)
                    ry = int(y*scale)
                    rw = int(windowSize[0]*scale)
                    rh = int(windowSize[1]*scale)
                    detections.append((rx, ry, rw, rh, prob))
    return detections

def nonMaximumSuppression(detections, overlapThresh=0.3):
    if not detections:
        return []
    boxes = np.array([[x,y,x+w,y+h,s] for (x,y,w,h,s) in detections])
    idxs  = np.argsort(boxes[:,4])[::-1]
    keep  = []
    while len(idxs)>0:
        i = idxs[0]; keep.append(boxes[i])
        xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
        w   = np.maximum(0, xx2-xx1)
        h   = np.maximum(0, yy2-yy1)
        overlap = (w*h)/((boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]))
        idxs = idxs[np.where(overlap <= overlapThresh)[0]+1]
    return [(int(b[0]),int(b[1]),int(b[2]-b[0]),int(b[3]-b[1]),b[4]) for b in keep]

def main():
    faces_dir = "faces"
    #if model and accuracy exist in cache, load it
    if os.path.exists(MODEL_CACHE) and os.path.exists(ACC_CACHE):
        print("Loading cached CNN model & accuracy…")
        model = keras.models.load_model(MODEL_CACHE)
        acc   = np.load(ACC_CACHE).item()
        print(f"Cached Test accuracy = {acc*100:.2f}%")
    
    else:
        #load data
        print("loading dataset...")
        pos = loadPositives(faces_dir, "train")
        neg = loadNegatives(faces_dir, "train")
        X = np.array(pos + neg, dtype="float32")/255.0
        y = np.array([1]*len(pos) + [0]*len(neg))
        X = np.expand_dims(X, -1)  #grayscale channel
        
        #shuffle dataset to avoid only positives in training and only negatives in test, then split
        idx = np.arange(len(X)); np.random.shuffle(idx)
        X, y = X[idx], y[idx]
        split = int(0.75*len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        #train model
        print("Training CNN...")
        model = buildModel(X_train.shape[1:])
        model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=10, batch_size=32
        )
        #evaluate on test set
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy = {acc*100:.2f}%")
        
        #cache model and accuracy
        model.save(MODEL_CACHE)
        np.save(ACC_CACHE, acc)
        
    #DEMO
    
    #choose random image, store color and grayscale version
    imgPaths = glob.glob(os.path.join(faces_dir, "images", "train", "*.jpg"))
    demoPath = random.choice(imgPaths)
    colorImg = cv2.imread(demoPath, cv2.IMREAD_COLOR)
    grayImg  = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
    
    #perform sliding window detector
    print("Running sliding window detector...")
    detections = slidingWindowDetectorCNN(
        grayImg, model, WINDOW_SIZE,
        DETECTION_THRESHOLD
    )
    
    #perform non maximum suppression
    print("performing non maximum suppression...")
    final = nonMaximumSuppression(detections, overlapThresh=0.1)
    
    #draw rectangles and display image
    for (x,y,w,h,score) in final:
        cv2.rectangle(colorImg, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            colorImg, f"{score:.2f}", (x,y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1
        )
    
    title = "Face Detected" if final else "No Face Detected"
    cv2.imshow(title, colorImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
