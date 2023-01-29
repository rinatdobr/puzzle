import cv2
import numpy as np

def rotate_image(image, angle):
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

def extend_image_size(image, size, color):
    image_extended = []
    if len(image.shape) == 2:
        image_extended = np.ndarray((image.shape[0] + size * 2,) + (image.shape[1] + size * 2,), dtype=image.dtype)
    elif len(image.shape) == 3:
        image_extended = np.ndarray((image.shape[0] + size * 2,) + (image.shape[1] + size * 2,) + (4,), dtype=image.dtype)
        
    image_extended[:, :] = color
    image_extended[size:image.shape[0] + size, size:image.shape[1] + size] = image
    return image_extended

def dist_from_center(center, counter):
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
        dist = dist_from_center(center, c_point)
        if dist > max_dist:
            max_dist = dist
            point = c_point

    return (point, max_dist)

def move(img, x, y):
    move_matrix = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, move_matrix, dimensions)

def to_masked_image(image):
    # threshold on white
    # Define lower and uppper limits
    lower = np.array([180, 180, 180])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(image, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    return cv2.bitwise_and(image, image, mask=mask)

def add_alpha_channel_to_image(image):
    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    return cv2.merge(rgba, 4)

def to_gray_image(masked_image):
    # Read input image as Grayscale
    return cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

def get_tiles_counters(gray_image):
    # Convert img to uint8 binary image with values 0 and 255
    # All pixels above 1 goes to 255, and other pixels goes to 0
    _, thresh_gray = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV)

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
            
    return corners

def align_tile(gray_tile, color_tile):
    angle = 0
    max_angle = 0
    width = 10000
    coords = ()

    # Align image
    while True:
        _gray_tile = rotate_image(gray_tile, angle)
        contours = cv2.findContours(_gray_tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # [-2] indexing takes return value before last (due to OpenCV compatibility issues).
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
        if angle == 360:
            break

    aligned_tile = rotate_image(gray_tile, max_angle)
    color_tile = rotate_image(color_tile, max_angle)
    return (aligned_tile[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]], color_tile[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]])

def to_contour_tile(aligned_tile):
    _, tile_gray_tresh = cv2.threshold(aligned_tile, 0, 255, cv2.THRESH_BINARY)
    contour = cv2.findContours(tile_gray_tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tile_counter = np.zeros(aligned_tile.shape, dtype=np.uint8)
    cv2.drawContours(tile_counter, contour[0], -1, (255, 255, 255))
    return (tile_counter, contour)

def get_tile_corners(countered_tile, counter):
    hh_c, ww_c = countered_tile.shape[:2]
    center = (int(ww_c / 2), int(hh_c / 2))
    tile_corners = []
    is_ok = True
    for i in range(4):
        point, distance = max_distance_on_counter(center, counter[0], hh_c, ww_c, i + 1)
        tile_corners.append(point)
        if distance != 0:
            cv2.line(countered_tile, center, point, (255, 0, 0), 1)
        else:
            print('No corner')
            is_ok = False
    return (tile_corners, is_ok)

def get_sidecounter(counter, start, end, reverse):
    sidecounter = np.array([], np.int32)
    is_subcounter = False
    if not reverse:
        for c_point in counter[0]:
            c_point = c_point[0]
            if c_point[0] == end[0] and c_point[1] == end[1]:
                is_subcounter = True
            if c_point[0] == start[0] and c_point[1] == start[1]:
                is_subcounter = False
            if is_subcounter:
                sidecounter = np.append(sidecounter, [c_point])
    if reverse:
        is_subcounter = True
        rightPart = []
        leftPart = []
        step = 1
        for c_point in counter[0]:
            c_point = c_point[0]
            if c_point[0] == end[0] and c_point[1] == end[1]:
                is_subcounter = True
            if c_point[0] == start[0] and c_point[1] == start[1]:
                is_subcounter = False
                step = 2
            if is_subcounter and step == 1:
                rightPart.append(c_point)
            if is_subcounter and step == 2:
                leftPart.append(c_point)
        sidecounter = np.append(sidecounter, leftPart)
        sidecounter = np.append(sidecounter, rightPart)
    sidecounter = sidecounter.reshape((-1, 1, 2))
    return sidecounter

def draw_subcounter(tile, sidecounter, side, cp):
    tile_subcounter = np.zeros(tile.shape, dtype=np.uint8)
    if cp:
        tile_subcounter = tile
    tile_subcounter = cv2.polylines(tile_subcounter, [sidecounter], False, (255, 255, 255), 1)
    x, y, w, h = cv2.boundingRect(sidecounter)
    tile_subcounter = move(tile_subcounter, -x, -y)
    tile_subcounter = tile_subcounter[:h, :w]
    if side == 1: # right
        tile_subcounter = cv2.rotate(tile_subcounter, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif side == 2: # bottom
        tile_subcounter = cv2.rotate(tile_subcounter, cv2.ROTATE_180)
    elif side == 3: # left
        tile_subcounter = cv2.rotate(tile_subcounter, cv2.ROTATE_90_CLOCKWISE)
    elif side == 4: # top
        True
    return tile_subcounter

def tile_sides(counter, corners, countered_tile, color_aligned_tile, prefix, tile_index):
    for side_index in range(4):
        side_index = side_index + 1
        corner_0 = 0
        corner_1 = 0
        is_reverse = False
        side_prefix = ''

        if side_index == 1: # right
            corner_0 = corners[0]
            corner_1 = corners[1]
            side_prefix = 'r'
        elif side_index == 2: # bottom
            corner_0 = corners[1]
            corner_1 = corners[2]
            side_prefix = 'b'
        elif side_index == 3: # left
            corner_0 = corners[2]
            corner_1 = corners[3]
            side_prefix = 'l'
        elif side_index == 4: # top
            corner_0 = corners[3]
            corner_1 = corners[0]
            side_prefix = 't'
            is_reverse = True

        sidecounter = get_sidecounter(counter[0], corner_0, corner_1, is_reverse)
        subcountered_tile = draw_subcounter(countered_tile, sidecounter, side_index, False)
        color_subcountered_tile = draw_subcounter(color_aligned_tile, sidecounter, side_index, True)
        cv2.imwrite('res/tiles/' + prefix + '_tile' + str(tile_index) + '_countered_' + side_prefix + '.jpg', subcountered_tile)
        cv2.imwrite('res/tiles/' + prefix + '_tile' + str(tile_index) + '_color_countered_' + side_prefix + '.jpg', color_subcountered_tile)

def process_tile(counter, alpha_image, gray_image, prefix, index):
    gray_tile = gray_image[counter[0][1]:counter[1][1], counter[0][0]:counter[1][0]]
    color_tile = alpha_image[counter[0][1]:counter[1][1], counter[0][0]:counter[1][0]]
    color_tile = extend_image_size(color_tile, gray_tile.shape[1], 0)
    gray_tile = extend_image_size(gray_tile, gray_tile.shape[1], 0)
    aligned_tile, color_aligned_tile = align_tile(gray_tile, color_tile)
    countered_tile, counter = to_contour_tile(aligned_tile)
    corners, is_valid_corenera = get_tile_corners(countered_tile, counter)
    cv2.imwrite('res/tiles/' + prefix + '_tile' + str(index) + '_gray.jpg', gray_tile)
    cv2.imwrite('res/tiles/' + prefix + '_tile' + str(index) + '_color.jpg', color_tile)
    cv2.imwrite('res/tiles/' + prefix + '_tile' + str(index) + '_aligned.jpg', aligned_tile)
    cv2.imwrite('res/tiles/' + prefix + '_tile' + str(index) + '_color_aligned.jpg', color_aligned_tile)
    cv2.imwrite('res/tiles/' + prefix + '_tile' + str(index) + '_countered.jpg', countered_tile)

    if is_valid_corenera:
        tile_sides(counter, corners, countered_tile, color_aligned_tile, prefix, index)

images = [
        '20230125_082359.jpg'
    ]

for image in images:
    print(image)

    original_image = cv2.imread(image)
    masked_image = to_masked_image(original_image)
    alpha_image = add_alpha_channel_to_image(masked_image)
    gray_image = to_gray_image(masked_image)

    # save results
    cv2.imwrite('res/' + image + '_masked.jpg', masked_image)
    cv2.imwrite('res/' + image + '_alpha.png', alpha_image)

    corners = get_tiles_counters(gray_image)

    print('Found', len(corners), 'tiles')

    # Prepare tiles
    counter = 0
    for c in corners:
        print('====', counter)
        process_tile(c, alpha_image, gray_image, image, counter)
        counter = counter + 1

exit
