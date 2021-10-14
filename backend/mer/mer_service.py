import pandas as pd
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import sys
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from os.path import isfile, join
from keras import backend as K
from os import listdir
from PIL import Image, ImageFilter
import math
import PIL.ImageOps
import numpy as np
import cv2
import glob
import os
import scipy.misc
from matplotlib import pyplot as plt
import io
from io import BytesIO
from sklearn.model_selection import train_test_split
from backend import urls

# from scipy.misc import imsave
from PIL import Image, ImageDraw

index_by_directory = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '+': 10,
    '-': 11,
    'xx': 12,
    '(': 13,
    ')': 14,
    'div': 15,
    '=': 16,
    'pm': 17,
    'i': 18,
    'X': 19,
    'A': 20,
    'int': 21,
    'N': 22,
    'o': 23,
    'T': 24,
    'C': 25,
    'S': 26,
    'd': 27
}


# In[2]:


def image_resize(image, size):
    height, width = image.shape
    if width >= height:
        nheight = int(round((size * 1.0 / width * height), 0))
        nheight = max(3, nheight)
        im_resize = cv2.resize(image, (size, nheight))  # resize size x nheight
        w_ver_top = int(round(((size - nheight) / 2), 0))
        w_ver_bot = size - nheight - w_ver_top
        im_resize = cv2.copyMakeBorder(im_resize, w_ver_top, w_ver_bot, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        nwidth = int(round((size * 1.0 / height * width), 0))
        nwidth = max(3, nwidth)

        im_resize = cv2.resize(image, (nwidth, size))  # resize size x nwidth
        w_hor_left = int(round(((size - nwidth) / 2), 0))
        w_hor_right = size - nwidth - w_hor_left
        im_resize = cv2.copyMakeBorder(im_resize, 0, 0, w_hor_left, w_hor_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return im_resize


def isolate_image_prepare(img):
    height, width = img.shape
    area = width * height
    iterations = math.ceil(area / 400000)  # 500 000 is a magical number
    img = ~img
    if (iterations < 4):
        blur_kernel = 3
    elif (4 <= iterations < 10):
        blur_kernel = 5
    elif (iterations >= 10):
        blur_kernel = 11
    elif (iterations >= 20):
        blur_kernel = 15

    #     img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0 )
    img = cv2.medianBlur(img, blur_kernel)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Set bits > 130 to 1 and <= 130 to 0
    img = cv2.dilate(img, kernel=np.ones((3, 3), np.uint8), iterations=1)
    ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    x1_min = sys.maxsize
    y1_min = sys.maxsize
    x2_max = -1
    y2_max = -1
    for c in ctrs:
        x, y, w, h = cv2.boundingRect(c)
        x1_min = min(x1_min, x)
        y1_min = min(y1_min, y)
        x2_max = max(x2_max, x + w)
        y2_max = max(y2_max, y + h)
    im_crop = img[y1_min:+y2_max + 5, x1_min:x2_max + 5]

    #     plt.imshow(im_crop, cmap='gray')
    #     plt.show()

    return im_crop


def load_images_from_folder(folder, show=False):
    train_data = []

    for idx, filename in enumerate(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  # Convert to Image to Grayscale

        #         if(show):
        #             plt.imshow(img, cmap='gray')
        #             plt.show()
        if img is not None:
            img = ~img

            im_resize = cv2.dilate(img, kernel=np.ones((2, 2), np.uint8), iterations=1)
            im_resize = cv2.resize(im_resize, (45, 45))  # Resize to (28, 28)

            #             if(show):
            #                 plt.imshow(im_resize, cmap='gray')
            #                 plt.show()

            im_resize = np.reshape(im_resize, (2025, 1))  # Flat the matrix

            train_data.append(im_resize)
    return train_data


def load_all_imgs():
    dataset_dir = "./datasets/"
    directory_list = listdir(dataset_dir)
    first = True
    data = []

    #     print('Exporting images...')
    for directory in directory_list:
        if (directory == '.DS_Store'):
            continue

        if first:
            first = False
            data = load_images_from_folder(dataset_dir + directory)
            for i in range(0, len(data)):
                data[i] = np.append(data[i], [str(get_index_by_directory(directory))])
            continue

        aux_data = load_images_from_folder(dataset_dir + directory)
        for i in range(0, len(aux_data)):
            aux_data[i] = np.append(aux_data[i], [str(get_index_by_directory(directory))])
        data = np.concatenate((data, aux_data))

    df = pd.DataFrame(data, index=None)
    df.to_csv('model/train_data.csv', index=False)


# In[3]:


# detect if input boundingBox contains a dot
def isDot(boundingBox, image_shape, printt=False):
    # image_shape = (height, width)

    (x, y), (xw, yh) = boundingBox
    area = (yh - y) * (xw - x)

    #     if(printt):
    #         print(((yh-y)/image_shape[0] * 100), xw, x, yh, y, (xw - x), (yh - y),  "percentages", 0.5 < (xw - x)/(yh - y) < 2 and ((yh-y)/image_shape[0] * 100) < 15)

    return 0.5 < (xw - x) / (yh - y) < 2 and ((yh - y) / image_shape[0] * 100) < 14  # 14 magic numero


# detect if input boundingBox contains a vertical bar
def isVerticalBar(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (yh - y) / (xw - x) > 2


# detect if a given boundingBox contains a horizontal bar
def isHorizontalBar(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (xw - x) / (yh - y) > 2


# detect if input boundingBox contains a square (regular letters, numbers, operators)
def isSquare(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (xw - x) > 8 and (yh - y) > 8 and 0.8 < (xw - x) / (yh - y) < 1.4


# detect if input three boundingBoxes are a division mark
def isDivisionMark(boundingBox, boundingBox1, boundingBox2, image_shape):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenY1 = y1 + (yh1 - y1) / 2
    cenY2 = y2 + (yh2 - y2) / 2
    # and max(y1, y2) - min(y1, y2) < 1.2 * abs(xw - x)
    return (isHorizontalBar(boundingBox) and isDot(boundingBox1, image_shape) and isDot(boundingBox2, image_shape)
            and x < x1 <= x2 < xw and max(y1, y2) > y and min(y1, y2) < y)


# detect if input two boundingBoxes are a lowercase i
def isLetterI(boundingBox, boundingBox1, image_shape):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1

    cond = False
    if (isDot(boundingBox, image_shape)):
        dot_center = x + (xw - x) / 2
        i_width = xw1 - x1

        cond = x1 - i_width * 0.2 < dot_center < xw1 + i_width * 0.2
    elif (isDot(boundingBox1, image_shape)):
        dot_center = x1 + (xw1 - x1) / 2
        i_width = xw - x

        cond = x - i_width * 0.2 < dot_center < xw + i_width * 0.2

    return (((isDot(boundingBox, image_shape) and (isVerticalBar(boundingBox1))) or (
                isDot(boundingBox1, image_shape) and (isVerticalBar(boundingBox))))
            and cond)  # 10 is a magical number


# detect if input two boundingBoxes are an equation mark
def isEquationMark(boundingBox, boundingBox1, image_shape):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1

    if (xw1 - x1 > xw - x):
        half = (xw1 - x1) / 2
        borders_condition = x1 - half < x < x1 + half and xw1 - half < xw < xw1 + half
    else:
        half = (xw - x) / 2
        borders_condition = x - half < x1 < x + half and xw - half < xw1 < xw + half

    if (y < y1):
        distance_cond = ((y1 - yh) * 100) / image_shape[0] < 20
    else:
        distance_cond = ((y - yh1) * 100) / image_shape[0] < 20

    return isHorizontalBar(boundingBox) and isHorizontalBar(boundingBox1) and borders_condition and distance_cond


# detect if input three boundingBoxes are a ellipsis (three dots)
def isDots(boundingBox, boundingBox1, boundingBox2, image_shape):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenY = y + (yh - y) / 2
    cenY1 = y1 + (yh1 - y1) / 2
    cenY2 = y2 + (yh2 - y2) / 2
    return (isDot(boundingBox, image_shape) and isDot(boundingBox1, image_shape) and isDot(boundingBox2,
                                                                                           image_shape) and max(cenY,
                                                                                                                cenY1,
                                                                                                                cenY2) - min(
        cenY, cenY1, cenY2) < 50)  # 30 is a migical number


# detect if input two boundingBoxes are a plus-minus
def isPM(boundingBox, boundingBox1, image_shape):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2

    percentageCase1 = ((y - yh1) / image_shape[0] * 100)
    percentageCase2 = ((y1 - yh) / image_shape[0] * 100)

    width_box = xw - x
    width_box1 = xw1 - x1

    prom1 = isHorizontalBar(boundingBox) and isSquare(boundingBox1)
    prom2 = isSquare(boundingBox) and isHorizontalBar(boundingBox1)

    case1 = prom1 and x < cenX1 < xw and -2 < percentageCase1 < 22 and width_box / width_box1 < 2.5
    case2 = prom2 and x1 < cenX < xw1 and -2 < percentageCase2 < 22 and width_box1 / width_box < 2.5
    return case1 or case2  # magical number


# detect if input three boundingBoxes are a fraction
def isFraction(boundingBox, boundingBox1, boundingBox2, image_shape):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    cenX2 = x2 + (xw2 - x2) / 2
    case1 = not isDot(boundingBox, image_shape) and not isDot(boundingBox1, image_shape) and isHorizontalBar(
        boundingBox2) and (y < y2 < yh1 or y1 < y2 < yh)
    case2 = not isDot(boundingBox2, image_shape) and not isDot(boundingBox, image_shape) and isHorizontalBar(
        boundingBox1) and (y2 < y1 < yh or y < y1 < yh2)
    case3 = not isDot(boundingBox1, image_shape) and not isDot(boundingBox2, image_shape) and isHorizontalBar(
        boundingBox) and (y1 < y < yh2 or y2 < y < yh1)
    return (case1 or case2 or case3) and max(cenX, cenX1, cenX2) - min(cenX, cenX1,
                                                                       cenX2) < 50  # 30 is a migical number


# return initial bounding boxes of input image
def initialBoxes(im):
    '''input: image; return: None'''

    im[im >= 127] = 255
    im[im < 127] = 0

    '''
    # set the morphology kernel size, the number in tuple is the bold pixel size
    kernel = np.ones((2,2),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    '''
    imgrey = im
    #     imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
    # remove noise

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL for only bounding outer box
    # bounding rectangle outside the individual element in image
    res = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # exclude the whole size image and noisy point
        res.append([(x, y), (x + w, y + h)])

    dump_rect = []
    # starts
    for rec1 in res:
        x = rec1[0][0]
        y = rec1[0][1]
        x1 = rec1[1][0]
        y1 = rec1[1][1]

        for rec2 in res:
            rec_x = rec2[0][0]
            rec_y = rec2[0][1]
            rec_x1 = rec2[1][0]
            rec_y1 = rec2[1][1]
            if (rec1 != rec2 and x - 5 <= rec_x and x1 + 5 >= rec_x1 and y - 5 <= rec_y and y1 + 5 >= rec_y1):
                dump_rect.append(rec2)
    #                 print("DADADA", rec1, rec2)

    #     # Discard the small collide rectangle
    #     for i in range(0, len(cnt)):
    #         for j in range(0, len(cnt)):
    #             if bool_rect[i][j] == 1:
    #                 area1 = rects[i][2] * rects[i][3]
    #                 area2 = rects[j][2] * rects[j][3]
    #                 if(area1 == min(area1,area2)):
    #                     dump_rect.append(rects[i])

    # Get the final rectangles
    #     final_rect = [i for i in rects if i not in dump_rect]

    # ends
    res = [i for i in res if i not in dump_rect]

    return res


# take in raw bounding boxes and detect components should be connected
def connect(im, res):
    '''input: image, raw rectangles; return: joint rectangles indicating detected symbols'''
    finalRes = []
    res.sort()
    i = 0
    while (i < len(res) - 1):
        (x, y), (xw, yh) = res[i]
        (x1, y1), (xw1, yh1) = res[i + 1]

        image = Image.fromarray(im)
        cropped = image.crop((x, y, xw, yh))

        equation = isEquationMark(res[i], res[i + 1], im.shape)
        letterI = isLetterI(res[i], res[i + 1], im.shape)
        pm = isPM(res[i], res[i + 1], im.shape)
        divisionMark = False
        dots = False
        fraction = False
        if i < len(res) - 2:
            (x2, y2), (xw2, yh2) = res[i + 2]
            #
            divisionMark = isDivisionMark(res[i], res[i + 1], res[i + 2], im.shape)
            dots = isDots(res[i], res[i + 1], res[i + 2], im.shape)
            fraction = isFraction(res[i], res[i + 1], res[i + 2], im.shape)

        # PM os really hard to determine, mixed with fraction

        if (equation or letterI or pm) and not fraction:
            finalRes.append([(min(x, x1), min(y, y1)), (max(xw, xw1), max(yh, yh1))])
            i += 2
        elif (divisionMark or dots) and not fraction:
            finalRes.append([(min(x, x1, x2), min(y, y1, y2)), (max(xw, xw1, xw2), max(yh, yh1, yh2))])
            i += 3
        else:
            finalRes.append(res[i])
            i += 1

    while i < len(res):
        finalRes.append(res[i])
        i += 1

    return finalRes


# slices im into smaller images based on boxes
def createSymbol(im):
    '''input: image, boxes; return: None'''
    # make a tmpelate image for next crop

    rawRes = initialBoxes(im)  # raw bounding boxes

    boxes = connect(im, rawRes)

    boxes = sorted(boxes, key=lambda box: (box[1][1] - box[0][1]) * (box[1][0] - box[0][0]))

    height, width = im.shape

    one_percent_area = width * height * 0.01

    symbol_list = []
    for box in boxes:
        (x, y), (xw, yh) = box
        w = xw - x
        h = yh - y
        # save rectangled element
        #         symbolImage = image.crop((x, y, xw, yh))

        #         if(x == 0):
        #             continue

        symbolImage = im[y:yh, x:xw]

        if (w * h < one_percent_area / 2 and float(w) / h < 1.3 and float(h) / w < 1.3):
            symbol_info = (symbolImage, "dot", x, y, xw, yh);
        elif (w * h < one_percent_area / 4 and not (float(w) / h > 1.8 or float(h) / w > 1.8)):
            symbol_info = (symbolImage, "noise", x, y, xw, yh);
        else:
            symbol_info = (symbolImage, "unknown", x, y, xw, yh);

        symbol_list.append(symbol_info)
    #         plt.imshow(symbol_info[0], cmap='gray')
    #         plt.show()

    return symbol_list


# In[4]:


sy = ['dots', 'tan', ')', '(', '+', '-', 'sqrt', '1', '0', '3', '2', '4', '6', 'mul', 'pi', '=', 'sin', 'pm', 'A',
      'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x',
      'div', '5', '7', '8', '9']

slash_sy = ['tan', 'sqrt', 'mul', 'pi', 'sin', 'pm', 'frac', 'cos', 'delta', 'bar', 'div', '^', '_']

variable = ['1', '0', '3', '2', '4', '6', '5', '7', '8', '9', 'pi', 'A', 'a', 'c', 'b', 'd', 'f', 'i', 'h', 'k', 'm',
            'o', 'n', 'p', 's', 't', 'y', 'x', '(', ')']
brules = {}

operator = ['-', '+', 'div']


def get_index_by_directory(directory):
    return index_by_directory[directory]


def update(im_name, symbol_list):
    #     im = Image.open(im_name)
    im = im_name
    list_len = len(symbol_list)
    for i in range(list_len):
        if i >= len(symbol_list): break

        symbol = symbol_list[i]
        predict_result = symbol[1]

        #         deal with division mark
        if predict_result == "-":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i + 1]
                s2 = symbol_list[i + 2]

                cond1 = (s1[3] <= symbol[3] and s2[3] >= symbol[3]) or (s1[3] >= symbol[3] and s2[3] <= symbol[3])
                cond2 = abs(s2[3] - s1[3]) < ((symbol[4] - symbol[2]) * 1.2)

                if (cond1 and cond2):

                    if s1[1] == "dot" and s2[1] == "dot":
                        updateDivision(symbol, s1, s2, symbol_list, im, i)
                        continue

        # deal with fraction
        if predict_result == "-":
            j = i
            upPart = 0
            underPart = 0
            while j < len(symbol_list):
                tmp = symbol_list[j]

                if tmp[2] > symbol[2] and tmp[4] < symbol[4] and tmp[5] > symbol[3]: upPart += 1
                if tmp[2] > symbol[2] and tmp[4] < symbol[4] and tmp[3] < symbol[5]: underPart += 1
                j += 1
            if upPart > 0 and underPart > 0:
                updateFrac(symbol, symbol_list, im, i)
                continue
    #         if predict_result == "-":
    #             if(i < len(symbol_list) -1):
    #                 s1 = symbol_list[i+1]
    #                 if (s1[1] == '+'):
    #                     s1

    return symbol_list


def toLatex(symbol_list):
    s = []
    i = 0
    while (i < len(symbol_list)):
        checkForOperations(symbol_list[i], symbol_list, i)
        symbol = symbol_list[i]
        if (isinstance(symbol, str)):
            s.append(symbol)
            continue
        value = symbol[1]

        # check for log, sin, cos, tan

        if value == 'frac':
            upper = []
            under = []
            i = i + 1
            while (i < len(symbol_list) and (
                    isUpperFrac(symbol, symbol_list[i]) or isUnderFrac(symbol, symbol_list[i]))):
                if isUpperFrac(symbol, symbol_list[i]): upper.append(symbol_list[i])
                if isUnderFrac(symbol, symbol_list[i]): under.append(symbol_list[i])
                i = i + 1
            if len(upper) > 1 and upper[len(upper) - 1][1] not in variable:
                upper.pop()
                i = i - 1
            if len(under) > 1 and under[len(under) - 1][1] not in variable:
                under.pop()
                i = i - 1

            upper_string = '{' + toLatex(upper) + '}'
            under_string = '{' + toLatex(under) + '}'
            s.append('\\frac' + upper_string + under_string)
            continue
        elif value == 'sqrt':
            outer = []
            inner = []
            i = i + 1
            while (i < len(symbol_list) and isInner(symbol, symbol_list[i])):
                inner.append(symbol_list[i])
                i = i + 1
            if len(inner) > 0 and inner[len(inner) - 1][1] not in variable:
                inner.pop()
                i = i - 1
            inner_string = '{' + toLatex(inner) + '}'
            s.append('\\sqrt' + inner_string)
            continue
        elif value in slash_sy:
            s.append('\\' + value)
            base = i
        elif i > 0 and (s[len(s) - 1] in slash_sy):
            # need to consider about range within squrt and frac
            s.append('{' + value + '}')
        elif i < len(symbol_list) - 1 and symbol[1] not in operator and isUpperSymbol(symbol, symbol_list[
            i + 1]) and not isInCenterRange(symbol, symbol_list[i + 1]):
            s.append(value)
            s.append('^{')
            i = i + 1
            upper_symbols = []
            #             uslov = isInCenterRange(symbol, symbol_list[i])
            while (i < len(symbol_list) and isUpperSymbol(symbol, symbol_list[i]) and not isInCenterRange(symbol,
                                                                                                          symbol_list[
                                                                                                              i])):
                #                 s.append(symbol_list[i][1])
                if (symbol_list[i][1] == 'frac'):
                    upper = []
                    under = []

                    x1_min = sys.maxsize
                    y1_min = sys.maxsize
                    x2_max = -1
                    y2_max = -1

                    frac_symbol = symbol_list[i]
                    i = i + 1

                    while (i < len(symbol_list) and (
                            isUpperFrac(frac_symbol, symbol_list[i]) or isUnderFrac(frac_symbol, symbol_list[i]))):
                        if isUpperFrac(frac_symbol, symbol_list[i]): upper.append(symbol_list[i])
                        if isUnderFrac(frac_symbol, symbol_list[i]): under.append(symbol_list[i])

                        im, typ, x, y, xw, yh = symbol_list[i]
                        x1_min = min(x1_min, x)
                        y1_min = min(y1_min, y)
                        x2_max = max(x2_max, xw)
                        y2_max = max(y2_max, yh)

                        i = i + 1
                    if len(upper) > 0 and upper[len(upper) - 1][1] not in variable:
                        upper.pop()
                        i = i - 1
                    if len(under) > 0 and under[len(under) - 1][1] not in variable:
                        under.pop()
                        i = i - 1

                    upper_string = '{' + toLatex(upper) + '}'
                    under_string = '{' + toLatex(under) + '}'
                    final_string = '\\frac' + upper_string + under_string

                    new_symbol = ['', final_string, x1_min, y1_min, x2_max, y2_max]
                    upper_symbols.append(new_symbol)
                else:
                    upper_symbols.append(symbol_list[i])
                    i = i + 1

            s.append(toLatex(upper_symbols))
            s.append('}')
            continue
        elif i < len(symbol_list) - 1 and isLowerSymbol(symbol, symbol_list[i + 1]) and (symbol[1] in variable) and (
                symbol_list[i + 1][1] in variable):
            s.append(value)
            s.append('_{')
            i = i + 1
            while (i < len(symbol_list) and isLowerSymbol(symbol, symbol_list[i])):
                s.append(symbol_list[i][1])
                i = i + 1
            s.append('}')
            continue
        else:
            s.append(value)
            base = i
        i = i + 1

    return "".join(s)


def isInCenterRange(cur, next):
    cur_center = cur[3] + (cur[5] - cur[3]) / 2
    next_center = next[3] + (next[5] - next[3]) / 2
    cur_height = cur[5] - cur[3]
    diff = abs(cur_center - next_center)

    #     print((100 * diff)/cur_height, cur_height, diff, cur_center, next_center, cur[1], next[1], "% ch dif cc nc")
    return ((100 * diff) / cur_height) < 28


def isVSame(cur, next):
    cur_center_x = cur[2] + (cur[4] - cur[2]) / 2
    next_center_x = next[2] + (next[4] - next[2]) / 2
    if abs(cur_center_x - next_center_x) < 30:
        return True
    else:
        return False


def isInner(cur, next):
    if next[3] < cur[5] and next[2] > cur[2] and next[4] - cur[4] < 10:
        return True
    else:
        return False


def isUpperFrac(cur, next):
    if next[5] < cur[3] and next[2] - cur[2] > -10 and next[4] - cur[4] < 10:
        return True
    else:
        return False


def isUnderFrac(cur, next):
    if next[3] > cur[5] and next[2] - cur[2] > -10 and next[4] - cur[4] < 10:
        return True
    else:
        return False


def isUpperSymbol(cur, next):
    cur_center = cur[3] + (cur[5] - cur[3]) / 2
    next_center = next[3] + (next[5] - next[3]) / 2
    cur_center_x = cur[2] + (cur[4] - cur[2]) / 2
    if next_center < cur_center - (next[5] - next[3]) / 2 and next[2] > cur_center_x:
        return True
    else:
        return False


#    if predict_result == "dot":
#             if i < (len(symbol_list) - 2):
#                 s1 = symbol_list[i+1]
#                 s2 = symbol_list[i+2]
#                 if symbol_list[i+1][1] == "dot" and symbol_list[i+2][1] == "dot":
#                     updateDots(symbol, s1, s2, symbol_list, im, i)
#                     continue


# check for sin, cos, tan
def checkForOperations(symbol, symbol_list, i):
    # check for sin
    okay = False

    if (symbol[1] == '\\int' or symbol[1] == '5'):
        if (i < (len(symbol_list) - 2)):
            s1 = symbol_list[i + 1]
            s2 = symbol_list[i + 2]
            if ((s1[1] == 'i' or s1[1] == '1') and s2[1] == 'n'):
                updateAlphaChar(symbol, symbol_list, i, 's')
                updateAlphaChar(s1, symbol_list, i + 1, 'i')

                okay = True
                return
        if (0 > i - 2):
            return
        else:
            s1 = symbol_list[i - 2]
            s2 = symbol_list[i - 1]
            if (((s1[1] == 'c' or s1[1] == 'o') and (s2[1] == 'o' or s2[1] == 'c'))):
                updateAlphaChar(symbol, symbol_list, i, 's')
                if (s1[1] == 'o'):
                    updateAlphaChar(s1, symbol_list, i - 2, 'c')
                if (s2[1] == 'c'):
                    updateAlphaChar(s2, symbol_list, i - 1, 'o')
                okay = True
                return

        if (not okay):
            return
    if (symbol[1] == 's'):
        if (i < (len(symbol_list) - 2)):
            s1 = symbol_list[i + 1]
            s2 = symbol_list[i + 2]
            if ((s1[1] == 'i' or s1[1] == '1') and s2[1] == 'n'):
                if (symbol[1] != 's'):
                    updateAlphaChar(symbol, symbol_list, i, 's')
                if (symbol[1] != 'i'):
                    updateAlphaChar(s1, symbol_list, i + 1, 'i')

                okay = True
                return

        if (0 > i - 2 and symbol[1] == 's'):
            updateAlphaChar(symbol, symbol_list, i, '8')
        else:
            s1 = symbol_list[i - 2]
            s2 = symbol_list[i - 1]
            if (((s1[1] == 'c' or s1[1] == 'o') and (s2[1] == 'o' or s2[1] == 'c'))):
                if (symbol[1] != 's'):
                    updateAlphaChar(symbol, symbol_list, i, 's')
                if (s1[1] != 'c'):
                    updateAlphaChar(s1, symbol_list, i - 2, 'c')
                if (s2[1] != 'o'):
                    updateAlphaChar(s2, symbol_list, i - 1, 'o')
                okay = True
                return

        if (not okay):
            updateAlphaChar(symbol, symbol_list, i, '8')

    elif (symbol[1] == 'c'):
        if (i < (len(symbol_list) - 2)):
            s1 = symbol_list[i + 1]
            s2 = symbol_list[i + 2]
            if ((s1[1] == 'o' or s1[1] == '0' or s1[1] == 'c') and (s2[1] == 's' or s2[1] == '\\int' or s2[1] == '5')):
                okay = True
                if (s1[1] == '0'):
                    updateAlphaChar(s1, symbol_list, i + 1, 'o')
                if (s1[1] == 'c'):
                    updateAlphaChar(s1, symbol_list, i + 1, 'o')

                if (s2[1] == '\\int'):
                    updateAlphaChar(s2, symbol_list, i + 2, 's')
                if (s2[1] == '5'):
                    updateAlphaChar(s2, symbol_list, i + 2, 's')
                return
        if (not okay):
            updateAlphaChar(symbol, symbol_list, i, '0')

    elif (symbol[1] == 't'):
        if (i < (len(symbol_list) - 2)):
            s1 = symbol_list[i + 1]
            s2 = symbol_list[i + 2]
            if ((s1[1] == 'a') and s2[1] == 'n'):
                okay = True
                return
        if (not okay):
            updateAlphaChar(symbol, symbol_list, i, '7')

    elif (symbol[1] == 'n'):
        if (0 > i - 2):

            updateAlphaChar(symbol, symbol_list, i, '0')
        else:
            s1 = symbol_list[i - 2]
            s2 = symbol_list[i - 1]

            if (((s1[1] == 's' or s1[1] == '5' or s1[1] == '\\int') and (s2[1] == 'i' or s2[1] == '1')) or (
                    (s1[1] == 't') and s2[1] == 'a')):

                if (s1[1] == '5' or s2[1] == '1' or s1[1] == '\\int'):
                    updateAlphaChar(s1, symbol_list, i - 2, 's')
                    updateAlphaChar(s2, symbol_list, i - 1, 'i')

                okay = True
                return

            if (not okay):
                if (s2[1] == 's'):
                    updateAlphaChar(symbol, symbol_list, i, 'x')
                else:
                    updateAlphaChar(symbol, symbol_list, i, '0')



    elif (symbol[1] == 'o'):
        if (0 > i - 1 or i + 1 >= len(symbol_list)):
            updateAlphaChar(symbol, symbol_list, i, '0')
        else:
            s1 = symbol_list[i - 1]
            s3 = symbol_list[i + 1]
            if ((s1[1] == 'c' or s1[1] == 'o' or s1[1] == '0' or s1[1] == '6') and s3[1] == 's'):
                if (s1[1] == '0'):
                    updateAlphaChar(s1, symbol_list, i - 1, 'c')
                if (s1[1] == 'o'):
                    updateAlphaChar(s1, symbol_list, i - 1, 'c')
                if (s1[1] == '6'):
                    updateAlphaChar(s1, symbol_list, i - 1, 'c')

                okay = True
                return
            if (not okay):
                updateAlphaChar(symbol, symbol_list, i, '0')

    elif (symbol[1] == 'a'):
        if (0 > i - 1 or i + 1 >= len(symbol_list)):
            updateAlphaChar(symbol, symbol_list, i, '9')
        else:
            s1 = symbol_list[i - 1]
            s3 = symbol_list[i + 1]
            if (s1[1] == 't' and s3[1] == 'n'):
                okay = True
                return
            if (not okay):
                updateAlphaChar(symbol, symbol_list, i, '9')

    return symbol_list


def updateAlphaChar(symbol, symbol_list, i, newSymbolString):
    x, y, xw, yh = symbol[2:]
    new_symbol = (symbol[0], newSymbolString, x, y, xw, yh)
    symbol_list[i] = new_symbol


def isLowerSymbol(cur, next):
    cur_center = cur[3] + (cur[5] - cur[3]) / 2
    next_center = next[3] + (next[5] - next[3]) / 2
    cur_center_x = cur[2] + (cur[4] - cur[2]) / 2
    if next_center > cur_center + (next[5] - next[3]) / 2 and next[2] > cur_center_x:
        return True
    else:
        return False


def area(symbol):
    return (symbol[4] - symbol[2]) * (symbol[5] - symbol[3])


def updateEqual(symbol, s1, symbol_list, im, i):
    new_x = min(symbol[2], s1[2])
    new_y = min(symbol[3], s1[3])
    new_xw = max(symbol[4], s1[4])
    new_yh = max(symbol[5], s1[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "=", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i + 1)


def updateDivision(symbol, s1, s2, symbol_list, im, i):
    new_x = min(symbol[2], s1[2], s2[2])
    new_y = min(symbol[3], s1[3], s2[3])
    new_xw = max(symbol[4], s1[4], s2[4])
    new_yh = max(symbol[5], s1[5], s2[5])
    croppedIm = im[new_y:+new_yh + 5, new_x:new_xw]
    new_symbol = (croppedIm, "div", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i + 2)
    symbol_list.pop(i + 1)


def updateDots(symbol, s1, s2, symbol_list, im, i):
    new_x = min(symbol[2], s1[2], s2[2])
    new_y = min(symbol[3], s1[3], s2[3])
    new_xw = max(symbol[4], s1[4], s2[4])
    new_yh = max(symbol[5], s1[5], s2[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "dots", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i + 2)
    symbol_list.pop(i + 1)


def updateI(symbol, s1, symbol_list, im, i):
    new_x = min(symbol[2], s1[2])
    new_y = min(symbol[3], s1[3])
    new_xw = max(symbol[4], s1[4])
    new_yh = max(symbol[5], s1[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "i", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i + 1)


def updatePM(symbol, s1, symbol_list, im, i):
    new_x = min(symbol[2], s1[2])
    new_y = min(symbol[3], s1[3])
    new_xw = max(symbol[4], s1[4])
    new_yh = max(symbol[5], s1[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "pm", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i + 1)


def updateBar(symbol, symbol_list, im, i):
    x, y, xw, yh = symbol[2:]
    new_symbol = (symbol[0], "bar", x, y, xw, yh)
    symbol_list[i] = new_symbol


def updateFrac(symbol, symbol_list, im, i):
    x, y, xw, yh = symbol[2:]
    new_symbol = (symbol[0], "frac", x, y, xw, yh)
    symbol_list[i] = new_symbol


def predict(img):
    #         imm = Image.open(operationBytes)
    #         imm.save('aux.png')
    #         img = cv2.imread('aux.png',cv2.IMREAD_GRAYSCALE)
    #         os.remove('aux.png')
    #         img = ~img

    #         plt.imshow(img, cmap='gray')
    #         plt.show()
    if img is not None:

        img_data = []
        img = isolate_image_prepare(img)
        symbols = createSymbol(img)
        symbols = sorted(symbols, key=lambda s: s[2])  # Sort by x
        for symbol in symbols:
            im_resize = symbol[0]
            width = symbol[4] - symbol[2]
            height = symbol[5] - symbol[3]

            #                 plt.imshow(im_resize, cmap='gray')
            #                 plt.show()

            area = width * height

            iterations = math.floor(area / 15000)  # 20 000 is a magical number

            iterations = min(15, iterations)
            if (iterations == 0 and area > 3000):
                iterations = 1

            im_resize = cv2.dilate(im_resize, kernel=np.ones((2, 2), np.uint8), iterations=iterations)
            if (area < 2700):
                im_resize = cv2.erode(im_resize, kernel=np.ones((2, 2), np.uint8), iterations=3)

            #                 plt.imshow(im_resize, cmap='gray')
            #                 plt.show()
            im_resize = image_resize(im_resize, 45)  # Resize to (28, 28)
            _, im_resize = cv2.threshold(im_resize, 40, 255, cv2.THRESH_BINARY)  # Set bits > 40 to 1 and <= 40 to 0
            #                 plt.imshow(im_resize, cmap='gray')
            #                 plt.show()

            im_resize = np.reshape(im_resize, (1, 45, 45))  # Flat the matrix
            img_data.append(im_resize)

        for i in range(len(img_data)):
            img_data[i] = np.array(img_data[i])
            img_data[i] = img_data[i].reshape(-1, 45, 45, 1)
            operation = ''
            if symbols[i][1] != "dot" and symbols[i][1] != "noise":
                result = urls.get_model().predict_classes(img_data[i])

            else:

                if (symbols[i][1] == "noise"):
                    result = ["noise"]
                    continue
                else:
                    result = ["dot"]

            if result[0] == 10:
                operation = '+'
            elif result[0] == 11:
                operation = '-'
            elif result[0] == 12:
                operation = 'x'
            elif result[0] == 13:
                operation = '('
            elif result[0] == 14:
                operation = ')'
            elif result[0] == 15:
                operation = 'div'
            elif result[0] == 16:
                operation = '='
            elif result[0] == 17:
                operation = 'pm'
            elif result[0] == 18:
                operation = 'i'
            elif result[0] == 19:
                operation = 'x'
            elif result[0] == 20:
                operation = 'a'
            elif result[0] == 21:
                operation = '\\int'
            elif result[0] == 22:
                operation = 'n'
            elif result[0] == 23:
                operation = 'o'
            elif result[0] == 24:
                operation = 't'
            elif result[0] == 25:
                operation = 'c'
            elif result[0] == 26:
                operation = 's'
            elif result[0] == 27:
                operation = 'd'
            else:
                operation = str(result[0])

            symbols[i] = (symbols[i][0], operation, symbols[i][2], symbols[i][3], symbols[i][4], symbols[i][5])

        remove_noise_symbols = []

        for i in range(len(symbols)):
            symbol = symbols[i]
            if (not (symbol[1] == 'noise')):
                remove_noise_symbols.append(symbol)

        updated_symbol_list = update(img, remove_noise_symbols)
        #             print([x[1] for x in updated_symbol_list])
        latex_string = toLatex(updated_symbol_list).replace("sin", "\\sin").replace("cos", "\\cos").replace("tan", "\\tan").replace('dot', '')
        if '\\int' in latex_string:
            latex_string = latex_string.replace('\\int', '\\int{')
            latex_string = latex_string + '}'
#             print(latex_string)
        return latex_string

