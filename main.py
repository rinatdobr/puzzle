import cv2
import numpy as np

images = [
        '20230125_082359.jpg'
    ]

for image in images:
    print(image)
    # Read image
    img = cv2.imread(image)
    hh, ww = img.shape[:2]
    print(hh, ww)

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([180, 180, 180])
    upper = np.array([255, 255, 255])
    print(lower, upper)

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)
    cv2.imshow('thresh', thresh)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)

    # save results
    cv2.imwrite('res/' + image + '_thresh.jpg', thresh)
    cv2.imwrite('res/' + image + '_morph.jpg', morph)
    cv2.imwrite('res/' + image + '_mask.jpg', mask)
    cv2.imwrite('res/' + image + '_result.jpg', result)

    # Read input image as Grayscale
    img = cv2.imread('res/' + image + '_result.jpg', cv2.IMREAD_GRAYSCALE)

    # Convert img to uint8 binary image with values 0 and 255
    # All pixels above 1 goes to 255, and other pixels goes to 0
    ret, thresh_gray = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('res/' + image + '_out_thresh_gray.jpg', thresh_gray)

    # Inverse polarity
    thresh_gray = 255 - thresh_gray

    # Find contours in thresh_gray.
    contours = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # [-2] indexing takes return value before last (due to OpenCV compatibility issues).

    corners = []

    print(len(contours))

    # Iterate contours, find bounding rectangles, and add corners to a list
    for c in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(c)

        # Append corner to list of corners - format is corners[i] holds a tuple: ((x0, y0), (x1, y1))
        corners.append(((x, y), (x+w, y+h)))

    # Convert grayscale to BGR (just for testing - for drawing rectangles in green color).
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw green rectangle (for testing)
    counter = 0
    for c in corners:
        if abs(c[0][0] - c[1][0]) > 150:
            cv2.rectangle(out, c[0], c[1], (0, 255, 0), thickness = 2)
            tile = result[c[0][1]:c[1][1], c[0][0]:c[1][0]]
            cv2.imwrite('res/tiles/' + image + '_out' + str(counter) + '.jpg', tile)
            counter = counter + 1

    cv2.imwrite('res/' + image + '_out.jpg', out)
exit
