import cv2
import numpy as np


def make_q(image, line_per):
    slope, intercept = line_per
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return x1, y1,x2, y2


def average_func(image, lines):
    left_fit = []
    right_fit = []
    try:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2),(y1,y2),1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope< 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit,axis = 0)
        right_fit_average = np.average(right_fit,axis = 0)
        left_line = make_q(image, left_fit_average)
        right_line = make_q(image, right_fit_average)

        

        return np.array((left_line,right_line))
    except:
        return lines
    
    


def canny(image):    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur, 50,150)
    return canny


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height),(1100,height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    bitAnd = cv2.bitwise_and(image, mask)
    return bitAnd



def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)

    return line_image



##img = cv2.imread('test_image.jpg')
##frame = np.copy(img)
##Canny = canny(frame)
##mask = region_of_interest(Canny)
##lines = cv2.HoughLinesP(mask, 2, np.pi/180, 100, np.array([]),40, 5)
##average_lines = average_func(img, lines)
##line_image = display_lines(img, average_lines)
##combo = cv2.addWeighted(line_image,0.8,img,1,0)
##cv2.imshow('img',combo)
##cv2.waitKey()

cap = cv2.VideoCapture('test2.mp4')

while cap.isOpened():
    _, img = cap.read()
    frame = np.copy(img)
    Canny = canny(frame)
    mask = region_of_interest(Canny)
    lines = cv2.HoughLinesP(mask, 2, np.pi/180, 100, np.array([]),10, 2)
    average_lines = average_func(img, lines)
    line_image = display_lines(img, average_lines)
    combo = cv2.addWeighted(line_image,0.8,img,1,0)
    
    cv2.imshow('Image',combo)
    if cv2.waitKey(1) == 27:
        break

    
cv2.destroyAllWindows()
