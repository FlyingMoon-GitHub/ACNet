import os

ANNOS_DIR = './annos'

NAMES = 'names.txt'
ANNOS_TRAIN = 'annos_train.txt'
ANNOS_TEST = 'annos_test.txt'

if not os.path.exists(ANNOS_DIR):
    os.mkdir(ANNOS_DIR)

classes = []
with open('./classes.txt', 'r') as f:
    line = f.readline()
    while line:
        line = line.strip().split(' ')
        classes.append((int(line[0]), line[1].split('.')[-1]))
        line = f.readline()

with open(os.path.join(ANNOS_DIR, NAMES), 'w') as f:
    for class_ in classes:
        f.write(str(class_[0] - 1) + ' ' + class_[1] + '\n')

name_dict = {}
with open('./images.txt', 'r') as f:
    line = f.readline()
    while line:
        line = line.strip().split(' ')
        name_dict[line[0]] = line[1]
        line = f.readline()

train_test_dict = {}
with open('./train_test_split.txt', 'r') as f:
    line = f.readline()
    while line:
        line = line.strip().split(' ')
        train_test_dict[line[0]] = int(line[1])
        line = f.readline()

train_data = {}
test_data = {}

with open('./image_class_labels.txt', 'r') as f:
    line = f.readline()
    while line:
        line = line.strip().split(' ')
        if train_test_dict[line[0]] == 1:
            train_data[name_dict[line[0]]] = int(line[1]) - 1
        else:
            test_data[name_dict[line[0]]] = int(line[1]) - 1
        line = f.readline()

with open(os.path.join(ANNOS_DIR, ANNOS_TRAIN), 'w') as f_train:
    for name in train_data.keys():
        f_train.write(name + ' ' + str(train_data[name]) + '\n')
        
with open(os.path.join(ANNOS_DIR, ANNOS_TEST), 'w') as f_test:
    for name in test_data.keys():
        f_test.write(name + ' ' + str(test_data[name]) + '\n')

print('Generated')