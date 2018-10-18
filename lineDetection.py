import cv2 as cv
import numpy as np
import tkFileDialog
import tkMessageBox
import math
import matplotlib.pyplot as plt

def nothing(x):
    pass

def houghLines(img, rhoR, thetaR):
    thetas = np.deg2rad(np.arange(0,180, 2*thetaR))
    diag = int(round(math.sqrt(width * width + height * height)))
    diag -= diag % 2
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    accumulator = np.zeros((2*diag, num_thetas), dtype=np.uint8)
    y_idx, x_idx = np.nonzero(img)

    xSize = x_idx.shape[0]

    i = 0

    while i < xSize:
        x = x_idx[i]
        y = y_idx[i]
        t_idx = 0

        while t_idx < num_thetas:
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag
            accumulator[rho, t_idx] += 1
            t_idx += 1
        i += 1
    return accumulator, thetas, diag

def plot_hough_acc(accumulator, diag, thetas, plot_title='Hough Accumulator Plot'):
    ''' A function that plot a Hough Space using Matplotlib. '''
    fig = plt.figure(figsize=(8, 8))
    fig.canvas.set_window_title(plot_title)
    plt.imshow(accumulator, cmap='binary_r', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), -diag*0.1, diag*0.1])

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction (1/10 Scale)')
    plt.tight_layout()
    plt.savefig('accumulator.png')
    fig = cv.imread('accumulator.png', cv.IMREAD_COLOR)
    figWin = cv.namedWindow('Parameter Space', cv.WINDOW_NORMAL)
    cv.imshow('Parameter Space', fig)
    plt.clf()
def newImg():
    imgPath = tkFileDialog.askopenfilename()
    img = cv.imread(imgPath, cv.IMREAD_COLOR)
    height, width, channel = img.shape
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    cv.imshow('Image', gray)
    return gray, img, height, width

def reset():
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def help():
    tkMessageBox.showinfo("Help", 'Listed below are the key functions:\n'
                                  'g - Show grayscale image\n'
                                  'x - Show hough lines\n'
                                  'p - Show hough parameter space plot\n'
                                  'z - Load new image\n'
                                  'h - Open help dialog\n')


imgPath = tkFileDialog.askopenfilename()
img = cv.imread(imgPath, cv.IMREAD_COLOR)
height, width, channel = img.shape
window = cv.namedWindow('Image', cv.WINDOW_NORMAL)
cv.createTrackbar('Hysteresis Min.', 'Image', 50, 100, nothing)
cv.createTrackbar('Hysteresis Max', 'Image', 110, 200, nothing)
cv.createTrackbar('Rho Res.', 'Image', 1, 10, nothing)
cv.createTrackbar('Theta Res.', 'Image', 1, 10, nothing)
cv.createTrackbar('Peak Threshold', 'Image', 50, 300, nothing)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.imshow('Image', gray)
help()

while(True):
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('g'):
        gray = reset()
        cv.imshow('Image', gray)
    elif key == ord('h'):
        help()
    elif key == ord('p'):
        plot_hough_acc(accumulator, diag, thetas)
    elif key == ord('x'):
        gray = reset()

        hMin = cv.getTrackbarPos('Hysteresis Min.', 'Image')
        hMax = cv.getTrackbarPos('Hysteresis Max', 'Image')
        rhoR = cv.getTrackbarPos('Rho Res.', 'Image')
        thetaR = cv.getTrackbarPos('Theta Res.', 'Image')
        peakT = cv.getTrackbarPos('Peak Threshold', 'Image')

        edges = cv.Canny(gray, hMin, hMax, apertureSize=3)
        accumulator, thetas, diag = houghLines(edges, rhoR, thetaR)
        temp = accumulator.flatten()

        j = 0
        idx = []
        while j < temp.size:
            if temp[j] > peakT:
                idx.append(j)
            j += 1

        vals = np.vstack(np.unravel_index(idx, accumulator.shape)).T

        i = 0
        while i < vals.shape[0]:
            rho = vals[i][0] - diag
            theta = thetas[vals[i, 1]]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 1)
            i = i + 1

        cv.imshow('Image', gray)
    elif key == ord('z'):
        gray, img, height, width = newImg()

cv.destroyAllWindows()