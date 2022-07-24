import os
import random

root_path = 'data/RGB/'
videos = os.listdir(root_path)

class_name_50 = os.listdir('C:/Users/Quangtung/PycharmProjects/Project2/UCF50/UCF50')

class_names = {}
for i, name in enumerate(class_name_50[:15]):
    class_names[name] = i

class_names['Billards'] = class_names.pop('Billiards')

print(list(class_names.keys()))

written_file = []
for vid_name in videos:
    for class_name in class_names.keys():
        if class_name in vid_name:
            vid_path = os.path.join(root_path, vid_name)
            count = len(os.listdir(vid_path)) - 1
            row = vid_path + ' ' + str(count) + ' ' + str(class_names[class_name])
            written_file.append(row)
random.shuffle(written_file)


len = len(written_file)
train_len = len * 3 // 5
val_len = len * 1 // 5

with open('data_split/train_RGB.txt', 'w') as f:
    for line in written_file[:train_len]:
        f.write(line)
        f.write('\n')

with open('data_split/test_RGB.txt', 'w') as f:
    for line in written_file[train_len: train_len + val_len]:
        f.write(line)
        f.write('\n')

with open('data_split/val_RGB.txt', 'w') as f:
    for line in written_file[train_len + val_len:]:
        f.write(line)
        f.write('\n')

print(len, train_len, val_len)