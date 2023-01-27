import cv2
import numpy as np

def rotateImage(image, angle):
    row = 0
    col = 0
    chan = 0
    if len(image.shape) == 2:
        row, col = image.shape
    elif len(image.shape) == 3:
        row, col, chan = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def extend_image_size(image: np.ndarray, size: int, color):
    image_extended = []
    if len(image.shape) == 2:
        image_extended = np.ndarray((image.shape[0] + size * 2,) + (image.shape[1] + size * 2,), dtype=image.dtype)
    elif len(image.shape) == 3:
        image_extended = np.ndarray((image.shape[0] + size * 2,) + (image.shape[1] + size * 2,) + (4,), dtype=image.dtype)
        
    image_extended[:, :] = color
    image_extended[size:image.shape[0] + size, size:image.shape[1] + size] = image
    return image_extended

def distance_(center, counter):
    dis = ((center[0] - counter[0]) ** 2 + (center[1] - counter[1]) ** 2) ** 0.5
    return dis

def max_distance_on_counter(center, counter, height, width, part):
    diff_w = width / 2 / 1.9
    diff_h = height / 2 / 1.9
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    if part == 1:
        x_min = width - diff_w
        x_max = width
        y_min = 0
        y_max = diff_h
    elif part == 2:
        x_min = width - diff_w
        x_max = width
        y_min = height - diff_h
        y_max = height
    elif part == 3:
        x_min = 0
        x_max = diff_w
        y_min = height - diff_h
        y_max = height
    elif part == 4:
        x_min = 0
        x_max = diff_w
        y_min = 0
        y_max = diff_h

    max_dist = 0
    point = (0, 0)
    for c_point in counter[0]:
        c_point = c_point[0]
        if c_point[0] < x_min or c_point[0] > x_max or c_point[1] < y_min or c_point[1] > y_max:
            continue
        dist = distance_(center, c_point)
        if dist > max_dist:
            max_dist = dist
            point = c_point

    return (point, max_dist)

def draw_counter_part(image, counter, start, end, reverse, color):
    line = np.array([], np.int32)
    isLine = False
    if reverse:
        isLine = True
    for c_point in counter[0]:
        c_point = c_point[0]
        if c_point[0] == end[0] and c_point[1] == end[1]:
            isLine = True
        if c_point[0] == start[0] and c_point[1] == start[1]:
            isLine = False
        if isLine:
            line = np.append(line, [c_point])
    line = line.reshape((-1, 1, 2))
    return cv2.polylines(image, [line], False, color, 1)

images = [
        '20230125_082359.jpg'
    ]

for image in images:
    print(image)

## Fill background to black
    # Read image
    img = cv2.imread(image)
    hh, ww = img.shape[:2]

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([180, 180, 180])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)
    # thresh = cv2.blur(thresh, ksize=(3, 3))

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    masked_background_result = cv2.bitwise_and(img, img, mask=mask)

    # black background to transparent
    tmp = cv2.cvtColor(masked_background_result, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(masked_background_result)
    rgba = [b, g, r, alpha]
    alpha_result = cv2.merge(rgba, 4)

    # save results
    cv2.imwrite('res/' + image + '_result.jpg', masked_background_result)
    cv2.imwrite('res/' + image + '_alpha_result.png', alpha_result)

## Find tiles
    # Read input image as Grayscale
    masked_background_gray_result = cv2.cvtColor(masked_background_result, cv2.COLOR_BGR2GRAY)

    # Convert img to uint8 binary image with values 0 and 255
    # All pixels above 1 goes to 255, and other pixels goes to 0
    ret, thresh_gray = cv2.threshold(masked_background_gray_result, 0, 255, cv2.THRESH_BINARY_INV)

    # Inverse polarity
    thresh_gray = 255 - thresh_gray

    # Find contours in thresh_gray.
    contours = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # [-2] indexing takes return value before last (due to OpenCV compatibility issues).

    corners = []

    # Iterate contours, find bounding rectangles, and add corners to a list
    for c in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(c)

        # Append corner to list of corners - format is corners[i] holds a tuple: ((x0, y0), (x1, y1))
        if w > 150:
            corners.append(((x, y), (x+w, y+h)))

    print('Found', len(corners), 'tiles')

    # Prepare tiles
    counter = 0
    for c in corners:
        print('====', counter)
        # cv2.rectangle(out, c[0], c[1], (0, 255, 0), thickness = 2)
        tile_gray = masked_background_gray_result[c[0][1]:c[1][1], c[0][0]:c[1][0]]
        hh, ww = tile_gray.shape[:2]
        tile_gray = extend_image_size(tile_gray, ww, 0)
        tile_color = alpha_result[c[0][1]:c[1][1], c[0][0]:c[1][0]]
        tile_color = extend_image_size(tile_color, ww, 0)
        angle = -90
        max_angle = 0
        width = 10000
        coords = ()
        cont = ()
        # Align image
        while True:
            _tile_gray = rotateImage(tile_gray, angle)
            contours = cv2.findContours(_tile_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # [-2] indexing takes return value before last (due to OpenCV compatibility issues).
            # Iterate contours, find bounding rectangles, and add corners to a list
            for c in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(c)

                # Append corner to list of corners - format is corners[i] holds a tuple: ((x0, y0), (x1, y1))
                if w > 150:
                    if w < width:
                        width = w
                        max_angle = angle
                        coords = ((x, y), (x+w, y+h))
                        cont = c
            angle = angle + 1
            if angle == 90:
                break
        tile_gray = rotateImage(tile_gray, max_angle)
        tile_gray = tile_gray[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
        _, tile_gray_tresh = cv2.threshold(tile_gray, 0, 255, cv2.THRESH_BINARY)
        # tile_gray_tresh = cv2.blur(tile_gray_tresh, ksize=(3, 3))
        cont = cv2.findContours(tile_gray_tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tile_counter = np.zeros(tile_gray.shape, dtype=np.uint8)
        cv2.drawContours(tile_counter, cont[0], -1, (255, 255, 255))

        hh_c, ww_c = tile_counter.shape[:2]
        center = (int(ww_c / 2), int(hh_c / 2))
        tile_corners = []
        for i in range(4):
            point, distance = max_distance_on_counter(center, cont[0], hh_c, ww_c, i + 1)
            tile_corners.append(point)
            if distance != 0:
                cv2.line(tile_counter, center, point, (255, 0, 0), 1)
            else:
                print('no max')

        tile_counter = cv2.cvtColor(tile_counter, cv2.COLOR_GRAY2BGR)
        tile_counter = draw_counter_part(tile_counter, cont[0], tile_corners[0], tile_corners[1], False, (255, 0, 0))
        tile_counter = draw_counter_part(tile_counter, cont[0], tile_corners[1], tile_corners[2], False, (255, 255, 0))
        tile_counter = draw_counter_part(tile_counter, cont[0], tile_corners[2], tile_corners[3], False, (255, 0, 255))
        tile_counter = draw_counter_part(tile_counter, cont[0], tile_corners[3], tile_corners[0], True, (0, 255, 255))

        cv2.imwrite('res/tiles/' + image + '_tile' + str(counter) + '.jpg', tile_counter)
        # tile_color = rotateImage(tile_color, max_angle)
        # tile_color = tile_color[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
        # cv2.imwrite('res/tiles/' + image + '_tile' + str(counter) + '.png', tile_color)
        counter = counter + 1

exit
