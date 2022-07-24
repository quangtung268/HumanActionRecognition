import cv2
import os
import numpy as np


def extract_frames(root_path):
    classes = os.listdir(root_path)
    class_count = 0
    for i in range(len(classes)):
        class_count += 1
        class_path = os.path.join(root_path, classes[i])
        each_class = os.listdir(class_path)

        for j in range(len(each_class)):
            video_path = os.path.join(class_path, each_class[j])

            vidcap = cv2.VideoCapture(video_path)

            success, image = vidcap.read()
            count = 0
            new_dir = f'data/RGB/{each_class[j]}/'
            os.mkdir(new_dir)
            while success:
                write_path = new_dir + f'img_{count:05d}.jpg'
                cv2.imwrite(write_path, image)
                success, image = vidcap.read()
                count += 1
            print(f"Class subset: {class_count}", each_class[j])
        print(f"Success: {classes[i]}", class_count)
        print("---------------------------------")
        print()


def extract_optical_flow(root_path):
    class_names = ['BaseballPitch', 'Basketball', 'BenchPress', 'Billards', 'Biking', 'BreastStroke', 'CleanAndJerk',
                   'Diving', 'Drumming', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop']

    classes = os.listdir(root_path)
    class_count = 0
    for i in range(len(classes)):
        class_count += 1
        class_path = os.path.join(root_path, classes[i])
        each_class = os.listdir(class_path)

        for j in range(len(each_class)):
            video_path = os.path.join(class_path, each_class[j])

            vidcap = cv2.VideoCapture(video_path)

            success, first_frame = vidcap.read()
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

            first_gray = prev_gray.copy()

            count = 0
            new_dir = f'data/flow/{each_class[j]}/'
            # os.mkdir(new_dir)

            mask = np.zeros_like(first_frame)

            # Sets image saturation to maximum
            mask[..., 1] = 255

            while (success):
                write_path = new_dir + f'flow_{count:05d}.jpg'
                success, frame = vidcap.read()
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except:
                    break
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # Computes the magnitude and angle of the 2D vectors
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # Sets image hue according to the optical flow
                # direction
                mask[..., 0] = angle * 180 / np.pi / 2

                # Sets image value according to the optical flow
                # magnitude (normalized)
                mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

                # Converts HSV to RGB (BGR) color representation
                rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

                cv2.imwrite(write_path, rgb)
                prev_gray = gray
                count += 1

            flow = cv2.calcOpticalFlowFarneback(prev_gray, first_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            write_path = new_dir + f'img_{count:05d}.jpg'
            # cv2.imwrite(write_path, rgb)

            print(f"Class subset: {class_count}", each_class[j])
        print(f"Success: {classes[i]}", class_count)
        print("---------------------------------")
        print()


if __name__ == '__main__':
    root_path = 'data/UCF50'
    extract_frames(root_path)
    # root_path = 'C:/Users/Quangtung/PycharmProjects/Project2/UCF50/UCF50'
    # extract_optical_flow(root_path)
