"""
#######################################
    @ Author : The DemonWolf
#######################################
"""


# Import necessary libraries
import operator
import time

import cv2
import numpy as np
from keras.models import load_model

from Algorithm.sudokuAlgorithm import solveSudoku


def expandLine(line, base):
    return line[0] + line[5:9].join([line[1:5] * (base - 1)] * base) + line[9:13]


def printBoard(board):
    base = 3
    side = base * base
    line0 = expandLine("╔═══╤═══╦═══╗", base)
    line1 = expandLine("║ . │ . ║ . ║", base)
    line2 = expandLine("╟───┼───╫───╢", base)
    line3 = expandLine("╠═══╪═══╬═══╣", base)
    line4 = expandLine("╚═══╧═══╩═══╝", base)

    symbol = "1234567890"
    nums = [[""] + [symbol[n] for n in row] for row in board]
    print(line0)
    for r in range(1, side + 1):
        print("".join(n + s for n, s in zip(nums[r - 1], line1.split("."))))
        print([line2, line3, line4][(r % side == 0) + (r % base == 0)])


if __name__ == '__main__':
    # Load digit recognition model
    classifier = load_model("Model/digit_model.h5")

    margin = 4
    box = 28 + 2 * margin
    grid_size = 9 * box

    # Start web cam
    cap = cv2.VideoCapture(0)
    # Save video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    flag = 0
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1080, 620))

    while True:
        # Read video frame by frame
        _, frame = cap.read()
        # Add text to the image on run time
        cv2.putText(frame, "Press 'q' to quit", (int(175 / 10), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Convert frame image as gray scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert image into blur image
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
        # Find the contour of the image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_grid = None
        maxArea = 0
        # Go through every contour to identify the largest rectangle polygon as Sudoku Puzzle
        for c in contours:
            area = cv2.contourArea(c)
            if area > 25000:
                peri = cv2.arcLength(c, True)
                polygon = cv2.approxPolyDP(c, 0.01 * peri, True)
                if area > maxArea and len(polygon) == 4:
                    contour_grid = polygon
                    maxArea = area

        if contour_grid is not None:
            # Draw the outline of the puzzle
            cv2.drawContours(frame, [contour_grid], 0, (0, 255, 0), 2)
            # Extract the pixel values of the corner of the puzzle
            points = np.vstack(contour_grid).squeeze()
            """ According to those points extract the puzzle grid """
            points = sorted(points, key=operator.itemgetter(1))
            if points[0][0] < points[1][0]:
                if points[3][0] < points[2][0]:
                    pts1 = np.float32([points[0], points[1], points[3], points[2]])
                else:
                    pts1 = np.float32([points[0], points[1], points[2], points[3]])
            else:
                if points[3][0] < points[2][0]:
                    pts1 = np.float32([points[1], points[0], points[3], points[2]])
                else:
                    pts1 = np.float32([points[1], points[0], points[2], points[3]])
            pts2 = np.float32([[0, 0], [grid_size, 0], [0, grid_size], [grid_size, grid_size]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            # Extract the grid
            grid = cv2.warpPerspective(frame, M, (grid_size, grid_size))
            grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
            grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)

            cv2.imshow("Grid", grid)
            msg = None
            if flag == 0:
                grid_txt = []
                Input = []
                """ Extract the puzzle cells and make predictions """
                for y in range(9):
                    Line = ""
                    temp = []
                    for x in range(9):
                        y2min = y * box + margin
                        y2max = (y + 1) * box - margin
                        x2min = x * box + margin
                        x2max = (x + 1) * box - margin
                        img = grid[y2min:y2max, x2min:x2max]
                        x = img.reshape(1, 28, 28, 1)
                        if x.sum() > 10000:
                            prediction = classifier.predict_classes(x)
                            Line += "{:d}".format(prediction[0])
                            temp.append(int(prediction[0]))
                        else:
                            Line += "{:d}".format(0)
                            temp.append(0)
                    grid_txt.append(Line)
                    Input.append(temp)
                t0 = time.time()
                # Solving the sudoku puzzle
                result, done, msg = solveSudoku(Input, verbose=False, all_solutions=False)
                deltaT = time.time() - t0

            x, y, w, h = 0, 0, 175, 75
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # Add text
            cv2.putText(frame, "Time taken to solve the puzzle : " + str(deltaT)[:7] + "s",
                        (x + int(w / 10), y + int(h / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if msg is not None:
                # Add text
                cv2.putText(frame, "Error on board : " + msg, (int(width / 4), int(height / 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if done:
                print("[INFO] Result...")
                printBoard(result)
                flag = 1
                # Add results to the a temporary image (numbers that not in the puzzle).
                fond = np.zeros(shape=(grid_size, grid_size, 3), dtype=np.float32)
                for y in range(len(result)):
                    for x in range(len(result[y])):
                        if grid_txt[y][x] == "0":
                            cv2.putText(fond, str(result[y][x]), (x * box + margin + 3,
                                                                  (y + 1) * box - margin - 3),
                                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 255, 0))
                # Add the results to the frame that has real size of the frame
                M = cv2.getPerspectiveTransform(pts2, pts1)
                h, w, c = frame.shape
                fondP = cv2.warpPerspective(fond, M, (w, h))
                # Convert the image into grayscale image
                img2gray = cv2.cvtColor(fondP, cv2.COLOR_BGR2GRAY)
                # Thresholding the grayscale image
                _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask = mask.astype('uint8')
                # Inverse the mask image
                mask_inv = cv2.bitwise_not(mask)
                # Create background & foreground images
                img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                img2_fg = cv2.bitwise_and(fondP, fondP, mask=mask).astype('uint8')
                # Add background & foreground images to build the final image
                dst = cv2.add(img1_bg, img2_fg)
                dst = cv2.resize(dst, (1080, 620))
                cv2.imshow("frame", dst)
                out.write(dst)
            else:
                # In case there is no solution display the image with reason
                frame = cv2.resize(frame, (1080, 620))
                cv2.imshow("frame", frame)
                out.write(frame)
        else:
            flag = 0
            frame = cv2.resize(frame, (1080, 620))
            cv2.imshow("frame", frame)
            out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
