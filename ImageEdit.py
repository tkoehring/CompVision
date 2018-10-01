import numpy as np
import cv2 as cv
import tkFileDialog
import tkMessageBox

def onBlur(x):
    if x == 0:
        return
    else:
        temp = cv.blur(img, (x, x))
        cv.imshow('Image', temp)

def onBlurCustom(x):
    filter = np.ones((2*x + 1, 2*x + 1))/(2*x + 1)**2
    temp = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    weight = 0
    weight2 = 0
    for i in range(x, height-x):
        for j in range(x, width-x):
            for k in range(1, x+1):
                for l in range(1, x+1):
                    weight += img[i-k, j-l]*filter[x-k, x-l] + img[i-k, j+l]*filter[x-k, x+l] + img[i-k, j]*filter[x-k, x]
                    weight += img[i+k, j-l]*filter[x+k, x-l] + img[i+k, j+l]*filter[x+k, x+l] + img[i+k, j]*filter[x+k, x]
                weight += img[i, j-k]*filter[x, x-k] + img[i, j+k]*filter[x, x+k]
            weight += img[i, j]*filter[x, x]

            weight2 += img[i-1, j-1]*filter[0, 0] + img[i-1, j]*filter[0, 1] + img[i-1, j+1]*filter[0, 2]
            weight2 += img[i, j-1]*filter[1, 0] + img[i, j]*filter[1, 1] + img[i, j+1]*filter[1, 2]
            weight2 += img[i + 1, j - 1]*filter[2, 0] + img[i+1, j]*filter[2, 1] + img[i + 1, j + 1]*filter[2, 2]
            temp[i, j] = weight
            weight = 0
    temp = cv.filter2D(img, -1, filter)
    cv.imshow('Image', temp)

def gray(img):
    temp = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    return temp
def reset():
    cv.destroyAllWindows()
    window = cv.namedWindow('Image', cv.WINDOW_NORMAL)
    temp = cv.imread(imgPath, cv.IMREAD_COLOR)
    return temp
def gradMag(img):
    dx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    dx = cv.pow(dx, 2)
    dy = cv.pow(dy, 2)
    img = cv.addWeighted(dx, 1, dy, 1, 0)
    img = cv.pow(img, 0.5)
    img = cv.convertScaleAbs(img)
    return img

def magVectors(N):
    img = cv.imread(imgPath, cv.IMREAD_COLOR)
    img = gray(img)
    dx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    mag = gradMag(img)
    n = 20 + N*2
    i = 0
    j = 0
    k = 10 + N
    while i < height:
        while j < width:
            m = mag[i, j]

            if m == 0:
                x = j
                y = i
            else:
                x = dx[i, j] / m
                y = dy[i, j] / m

                if x < 0:
                    x = int((x + j) - k)
                else:
                    x = int((x + j) + k)
                if y < 0:
                    y = int((y + i) - k)
                else:
                    y = int((y + i) + k)

            cv.arrowedLine(img, (j, i), (x, y), (255, 0, 0), 2)
            j += n
        i += n
        j = 0

    cv.imshow('Image', img)

def imgRot(x):
    rot = cv.getRotationMatrix2D((width / 2, height / 2), x, 1)
    temp = cv.warpAffine(img, rot, (width, height))
    cv.imshow('Image', temp)

imgPath = tkFileDialog.askopenfilename()
img = cv.imread(imgPath, cv.IMREAD_COLOR)
window = cv.namedWindow('Image', cv.WINDOW_NORMAL)
height, width, channel = img.shape
cCount = 0
trackBar = 0
cv.imshow('Image', img)

while(True):
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('i'):
        img = reset()
        cv.imshow('Image', img)
    elif key == ord('w'):
        cv.imwrite('out.jpg', img)
    elif key == ord('g'):
        img = reset()
        img = gray(img)
        cv.imshow('Image', img)
    elif key == ord('G'):
        img = reset()
        for i in range(0, height):
            for j in range(0, width):
                temp = (img[i, j, 0] + img[i, j, 1] + img[i, j, 2])/3
                img[i, j, 0] = temp
                img[i, j, 1] = temp
                img[i, j, 2] = temp
        cv.imshow('Image', img)
    elif key == ord('c'):
        img = reset()
        if cCount%3 == 0:
            img[:, :, 0] = 0
            img[:, :, 1] = 0
            cCount = 0
        elif cCount%3 == 1:
            img[:, :, 1] = 0
            img[:, :, 2] = 0
        elif cCount%3 == 2:
            img[:, :, 0] = 0
            img[:, :, 2] = 0
        cCount += 1
        cv.imshow('Image', img)
    elif key == ord('s'):
        img = reset()
        img = gray(img)
        cv.imshow('Image', img)
        cv.createTrackbar('Blur', 'Image', 0, 50, onBlur)
    elif key == ord('S'):
        img = reset()
        img = gray(img)
        cv.createTrackbar('Blur Cust', 'Image', 0, 50, onBlurCustom)

    elif key == ord('d'):
        img = cv.resize(img, None, None, 0.5, 0.5, cv.INTER_AREA)
        cv.imshow('Image', img)
   #elif key == ord('D'):
    elif key == ord('x'):
        img = reset()
        img = gray(img)
        #img = cv.blur(img, (10, 10))
        img = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
        img = cv.convertScaleAbs(img)
        cv.imshow('Image', img)
    elif key == ord('y'):
        img = reset()
        img = gray(img)
        img = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
        img = cv.convertScaleAbs(img)
        cv.imshow('Image', img)
    elif key == ord('m'):
        img = reset()
        img = gray(img)
        img = gradMag(img)
        cv.imshow('Image', img)
    elif key == ord('p'):
        img = reset()
        img = gray(img)
        magVectors(0)
        cv.createTrackbar('Vector Freq.', 'Image', 0, 50, magVectors)
    elif key == ord('r'):
        img = reset()
        imgRot(0)
        cv.createTrackbar('Rotation', 'Image', 0, 360, imgRot)
    elif key == ord('z'):
        imgPath = tkFileDialog.askopenfilename()
        img = cv.imread(imgPath, cv.IMREAD_COLOR)
        height, width, channel = img.shape
        img = reset()
        cv.imshow('Image', img)



cv.destroyAllWindows()

