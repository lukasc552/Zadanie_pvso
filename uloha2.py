import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

img = cv.imread('circles.jpg')
img_width, img_height = img.shape[:2]
edge_frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edge_frame = cv.medianBlur(edge_frame, 5)
edge_frame = cv.Canny(edge_frame, 0, 50)

# copy_frame = img.copy()
# h_frame = cv.cvtColor(copy_frame, cv.COLOR_BGR2GRAY)
# h_frame = cv.medianBlur(h_frame, 5)
# h_circles = cv.HoughCircles(h_frame, cv.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=0, maxRadius=0)
# h_circles = np.uint16(np.around(h_circles))
# for i in h_circles[0, :]:
#     cv.circle(copy_frame, (i[0], i[1]), i[2], BLUE)
#
# cv.imshow("CV HOUGH", copy_frame)


# ============ Houghova transformacia -> kruznica
radius_range = [i for i in range(5, int(img_width/2))]

accumulator = np.zeros((img_height, img_width))
circles = []

max_thresh = 160
avg_thresh = 33

for r in range(89, 91):
    # vytvorenie akumulatora
    for x in range(0, img_height):
        for y in range(0, img_width):

            if edge_frame[y][x] == 255:
                for angle in range(0, 360):
                    b = y - round(r * np.sin(angle * np.pi/180))
                    a = x - round(r * np.cos(angle * np.pi/180))
                    if 0 < a < img_width and 0 < b < img_height:
                        accumulator[a][b] += 1

    print('For radius: ', r)
    maximum = np.amax(accumulator)

    if (maximum > max_thresh):

        print("Detecting the circles for radius: ", r)

        for i in range(img_width-1):
            for j in range(img_height-1):
                if accumulator[i][j] >= max_thresh:
                    avg = 0.0
                    for ii in [-1, 0, 1]:
                        for jj in [-1, 0, 1]:
                            avg += accumulator[i+ii][j+jj]

                    avg_sum = avg/9

                    if avg_sum >= avg_thresh:
                        circles.append((i, j, r))
                        accumulator[i:i + 5, j:j + 5] = 0

print("Number of detected circles: ", len(circles))
for circle in circles:
    cv.circle(img, (circle[0], circle[1]), circle[2], GREEN, 1)

plt.figure()
plt.imshow(accumulator)
plt.show()

# ===============================================

cv.imshow("Frame", img)

cv.waitKey(0)

cv.destroyAllWindows()
