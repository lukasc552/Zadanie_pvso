import cv2 as cv
import numpy as np


def power_aproximator(x_log):
    if x_log == 0.0:
        return -1
    return 92.055 * x_log ** (-0.97)


def power_diff_aproximator(x_log):  # area
    if x_log == 0.0:
        return 0.0
    return 4.0978 * x_log ** (-0.269)


def get_length_of_trajectory(trajectory):
    dist = 0.0
    x_old, y_old = trajectory[0]
    for point in trajectory:
        x, y = point
        dist += np.sqrt((x - x_old) ** 2 + (y - y_old) ** 2)
        x_old, y_old = x, y
    return dist


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
                 )

feature_params = dict(maxCorners=150,
                      qualityLevel=0.15,
                      minDistance=10,
                      blockSize=7)

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

trajectory_len = 20
detect_interval = 1
trajectories = []
frame_idx = 0

cap = cv.VideoCapture(0)

values = []
results = []
string2 = ""
start_printing = False

while True:

    _, frame = cap.read()
    img = frame.copy()
    moving_trajectories = []


    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, (3, 3), 3)

    main_width = int(cap.get(3))
    main_heigth = int(cap.get(4))


    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _str, _errr = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if good_flag:
                trajectory.append((x, y))
                if len(trajectory) > trajectory_len:
                    del trajectory[0]
                new_trajectories.append(trajectory)

        trajectories = new_trajectories

        # chceme zobrat len pohybujuce sa body
        for trajectory in trajectories:
            if get_length_of_trajectory(trajectory) > 50:
                moving_trajectories.append(trajectory)
                x, y = trajectory[-1]
                cv.circle(img, (int(x), int(y)), 2, RED, -1)

        # vypociatnie hranicnych bodov pre oramovanie objektu
        lx, ly, ux, uy = main_width, main_heigth, 0, 0
        for move_traj in moving_trajectories:
            x, y = move_traj[-1]
            if x < lx:
                lx = int(x)
            if y < ly:
                ly = int(y)
            if x > ux:
                ux = int(x)
            if y > uy:
                uy = int(y)

        roi = frame_gray[lx:ux, ly:uy]
        contours, _ = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        area = 0.0
        for contour in contours:
            area = cv.contourArea(contour)

        # r_width_px = abs(lx - ux)
        if len(moving_trajectories) > 1:
            # string = str(r_width_px)
            if len(values) == 10:
                for i in range(len(values) - 1):
                    values[i] = values[i + 1]
                values[-1] = area
            else:
                values.append(area)
                continue

            result = 0.0
            if len(results) == 10:
                for i in range(len(results) - 1):
                    results[i] = results[i + 1]
                tmp = power_diff_aproximator(abs(values[-1] - values[0]))
                if tmp != 0:
                    results[-1] = tmp
                result = sum(results) / 10
                start_printing = True

            else:
                tmp = power_diff_aproximator(abs(values[-1] - values[0]))
                if tmp != 0:
                    results.append(power_diff_aproximator(abs(values[-1] - values[0])))
                continue

            if frame_idx % 10 == 0:
                result = int(result * 100) / 100
                string2 = str(result)

            if start_printing:
                cv.putText(img, "Dist: " + string2, (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, GREEN, 2)
                cv.rectangle(img, (lx, ly), (ux, uy), BLUE)

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv.circle(mask, (x, y), 5, 0, -1)

        p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray

    cv.imshow("Frame", img)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
