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
    accumulator = np.zeros((2*diag, num_thetas), dtype=np.uint64)

    y_idx, x_idx = np.nonzero(img)
    pointLoc = np.zeros((x_idx.shape[0], thetas.size), dtype=np.uint64)
    xSize = x_idx.shape[0]

    i = 0

    while i < xSize:
        x = x_idx[i]
        y = y_idx[i]
        t_idx = 0

        while t_idx < num_thetas:
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag
            accumulator[rho, t_idx] += 1
            pointLoc[i, t_idx] = rho
            t_idx += 1
        i += 1
    return accumulator, thetas, x_idx, y_idx, pointLoc, diag

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
    img = cv.imread(imgPath, cv.IMREAD_COLOR)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def help():
    tkMessageBox.showinfo("Help", 'You must find the lines before running commands p, i, w, s.\n'
                                  'Listed below are the key functions:\n'
                                  'h - Open help dialog\n'
                                  'g - Show grayscale image\n'
                                  'z - Load new image\n'
                                  'x - Find lines\n'
                                  'p - Show hough parameter space plot\n'
                                  'i - Show points used in voting\n'
                                  'w - Show hough lines with line fitting\n'
                                  's - Show hough lines\n')


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
        accumulator, thetas, x_idx, y_idx, pointLoc, diag = houghLines(edges, rhoR, thetaR)
        temp = accumulator.flatten()

        j = 0
        idx = []
        while j < temp.size:
            if temp[j] > peakT:
                idx.append(j)
            j += 1

        vals = np.vstack(np.unravel_index(idx, accumulator.shape)).T

        i = 0
        temp = []
        while i < vals.shape[0]:
            j = 0
            points = []
            while j < pointLoc.shape[0]:
                if pointLoc[j, [vals[i][1]]] == vals[i][0]:
                    img[y_idx[j], x_idx[j]] = [0, 255, 0]
                    points.append((x_idx[j], y_idx[j], 1))
                j += 1
            temp.append((list(points)))
            i += 1
        cv.imshow('Image', img)

        i = 0
        while i < vals.shape[0]:
            points = np.array(temp[i])

            j = 0
            sum = np.zeros((3, 3))
            while j < points.shape[0]:
                sum += np.matmul(points[j], points[j].transpose())
                j += 1
            w, v = np.linalg.eig(sum)

            x0 = points[0][0]
            y0 = points[0][1]
            a = v[-1][0]
            b = v[-1][1]
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 1)
            i += 1

        bestFit = gray.copy()
        gray = reset()

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
            i += 1

        houghImg = gray.copy()
        cv.imshow('Image', bestFit)
    elif key == ord('i'):
        cv.imshow('Image', img)
    elif key == ord('w'):
        cv.imshow('Image', bestFit)
    elif key == ord('s'):
        cv.imshow('Image', houghImg)
    elif key == ord('z'):
        gray, img, height, width = newImg()

cv.destroyAllWindows()