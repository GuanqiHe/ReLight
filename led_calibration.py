'''
LED property: light intensity/s *4 (RGBW)

Color Card Extraction

LED data analysis
'''

import numpy as np
import cv2
import rawpy
import exif
import math
from matplotlib import pyplot as plt


def ColorCardPosExtract(img, sample_scale=0.4, save=None, show=False):
    '''
    @img: a M*N*3 8-bit BGR img
    @sample_scale: get card value by taking average in width*sample_scale length square
    @save: save res as "save".npz
    @show: show mid steps when processing img

    @return: 3-dim array (center_x,center_y,square_width), start at bottom left, col first

    Canny get contour, find min-bounding rect, take average.
    '''

    (b, g, r) = cv2.split(img)
    sample_scale = sample_scale/2
    b_s = cv2.Canny(b, 20, 60)
    g_s = cv2.Canny(g, 20, 60)
    r_s = cv2.Canny(r, 20, 60)
    res = cv2.add(b_s, g_s, r_s)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dst = cv2.dilate(res, kernel)

    contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if(cv2.contourArea(cnt) < 80000):
            cv2.fillConvexPoly(dst, cnt, 0)

    dst = 255-dst

    for cnt in contours:
        if(cv2.contourArea(cnt) < 20000):
            cv2.fillConvexPoly(dst, cnt, 0)

    squares = []
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 80000 and area < 250000:
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if abs(w/h) > 1.5 or abs(h/w) > 1.5:
                continue
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            count += 1
            center = np.int0(rect[0])

            squares.append([center[0], center[1], w*sample_scale])
            cv2.putText(img, str(count), (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 20)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 20)

    print(len(squares))

    squares = np.array(squares)

    if save:
        try:
            np.save(save+".npy", squares)
        except:
            pass

    if show == True:
        img_show("Extract", img)
        #cv2.imshow("Extract Result", img)
        # cv2.waitKey(0)

    return squares


def ColorCardPosExtractClick(img, sample_scale=0.4, save=None):
    '''
    @img: a M*N*3 8-bit BGR img
    @sample_scale: get card value by taking average in width*sample_scale length square
    @save: save res as "save".npz

    @return: 3-dim array (center_x,center_y,square_width), start at bottom left, col first

    Get Card Position By Click, left button for center square, from bottom left, col first, right bottom for square width
    '''
    center = []
    w = []
    d = 0

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            center.append((x/800*ori_x, y/600*ori_y))
            cv2.circle(img, (x, y), 2, (0, 0, 255), thickness=2)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness=1)
            cv2.imshow("image", img)
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(w) < 2:
                w.append((x/800*ori_x, y/600*ori_y))
                xy = "%d,%d" % (x, y)
                cv2.circle(img, (x, y), 2, (255, 0, 0), thickness=2)
                cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (255, 255, 255), thickness=1)
                cv2.imshow("image", img)
                if len(w) == 2:
                    nonlocal d
                    d = math.sqrt((w[0][0]-w[1][0])**2+(w[0][1]-w[1][1])**2)
                    print("square_width: ", d)

    cv2.namedWindow("image")
    (ori_y, ori_x, _) = img.shape
    img = cv2.resize(img, (800, 600))
    cv2.imshow("image", img)
    print(img.shape)

    #cv2.resizeWindow("image", 800, 600)
    cv2.setMouseCallback("image", mouse)

    k = cv2.waitKey(0)
    if k:
        cv2.destroyAllWindows()
    res = np.array([[int(x[0]), int(x[1]), d*sample_scale] for x in center])
    if save:
        try:
            np.save(save+".npy", res)
        except:
            pass

    return res


def SquareAvg(img, center, r):
    '''
    @img: RGB 16 bit image

    @return: square point RBG
    '''
    r = int(r)
    tmp = img[(center[1]-r):(center[1]+r), (center[0]-r):(center[0]+r)]
    res = tmp.astype(np.float)
    res = np.sum(np.sum(res, 0), 0)/(4*r*r)
    return res


def ImgGetColorSquare(img, squares):
    '''
    @img: RGB 16 bit image
    @squares: return value of ColorCardPosExtract, (x,y,r)*24
    @return: compress RGB*24
    '''
    res = []
    for x, y, r in squares:
        res.append(SquareAvg(img, (int(x), int(y)), r))
    return np.array(res)


def ReadRaw(path):
    '''
    @path: image path
    '''
    d = {}
    d["path"] = path
    f = rawpy.imread(path+".NEF")
    d['processed'] = f.postprocess(
        gamma=(1, 1), no_auto_bright=True, output_bps=16)
    f.close()
    with open(d["path"]+".JPG", 'rb') as f:
        d['exposure_time'] = exif.Image(f)['exposure_time']
    return d


def ImageSet(prefix, start_end):
    '''
    @start_end: [(s1,e1),(s2,e2),...]
    a image path generator
    '''
    for s_e in start_end:
        for i in range(s_e[0], s_e[1]):
            yield prefix+str(i)


# image show in jupyternote book
def img_show(name, img, raw=False):
    plt.figure(name)
    if not raw:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
