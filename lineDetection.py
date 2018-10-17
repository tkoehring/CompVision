import cv2 as cv
import numpy as np
import tkFileDialog
import math
import matplotlib.pyplot as plt

def gradMag(img):
    blur = cv.GaussianBlur(img,(5,5),0)
    dx = cv.Sobel(blur, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=3)
    dx = cv.pow(dx, 2)
    dy = cv.pow(dy, 2)
    img = cv.addWeighted(dx, 1, dy, 1, 0)
    img = cv.pow(img, 0.5)
    img = cv.convertScaleAbs(img)
    return img

def houghLines(img):
    thetas = np.deg2rad(np.arange(-90,90,2))
    diag = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag, diag, diag * 2)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    accumulator = np.zeros((2 * diag, num_thetas), dtype=np.uint8)
    y_idx, x_idx = np.nonzero(img)

    xSize = x_idx.shape[0]

    i = 0
    t_idx = 0

    while i < xSize:
        x = x_idx[i]
        y = y_idx[i]

        while t_idx < num_thetas:
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag
            accumulator[rho, t_idx] += 1
            t_idx += 1
        i += 1
    return accumulator, thetas, rhos


def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    ''' A function that plot a Hough Space using Matplotlib. '''
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_title)

    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()

imgPath = tkFileDialog.askopenfilename()
img = cv.imread(imgPath, cv.IMREAD_COLOR)
window = cv.namedWindow('Image', cv.WINDOW_NORMAL)
height, width, channel = img.shape
cCount = 0
trackBar = 0

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)

accumulator, thetas, rhos = houghLines(edges)

# lines = cv.HoughLines(edges,1,np.pi/180,200)
# i = 0
# while i < lines.shape[0]:
#     rho = lines[i][0][0]
#     theta = lines[i][0][1]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     i = i + 1
#     cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#
# cv.imwrite('accumulator.jpg',accumulator)
# cv.imwrite('edges.jpg',edges)
# cv.imwrite('houghlines.jpg',img)
# cv.imshow('Image', img)

indices = np.argpartition(accumulator.flatten(), -2)[-50:]
test = np.vstack(np.unravel_index(indices, accumulator.shape)).T
#img = cv.imread(imgPath, cv.IMREAD_COLOR)

i = 0
while i < test.shape[0]:
    rho = test[i][0]
    theta = test[i][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img,(x1,y1),(x2,y2),(0,255,255),2)
    i = i + 1

cv.line(img,(0,0),(100,100),(0,255,255),2)
cv.imshow('Image', img)

plot_hough_acc(accumulator)