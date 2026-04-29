BACKGROUND_CATEGORY = [
    'ground', 'land', 'grass', 'tree', 'building', 'wall', 'sky', 'lake',
    'water', 'river', 'sea', 'railway', 'railroad', 'keyboard', 'helmet',
    'cloud', 'house', 'mountain', 'ocean', 'road', 'rock', 'street',
    'valley', 'bridge', 'sign',
]

class_names = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

new_class_names = [
    'aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair seat', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
]


class_names_coco = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

new_class_names_coco = [
    'person with clothes,people,human', 'bicycle', 'car', 'motorbike', 'aeroplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird avian',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack,bag',
    'umbrella,parasol', 'handbag,purse', 'necktie', 'suitcase', 'frisbee',
    'skis', 'sknowboard', 'sports ball', 'kite', 'baseball bat',
    'glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'dessertspoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair seat', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'toilet', 'tvmonitor screen', 'laptop', 'mouse',
    'remote control', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hairdrier,blowdrier', 'toothbrush',
]


BACKGROUND_CATEGORY_COCO = [
    'ground', 'land', 'grass', 'tree', 'building', 'wall', 'sky', 'lake',
    'water', 'river', 'sea', 'railway', 'railroad', 'helmet',
    'cloud', 'house', 'mountain', 'ocean', 'road', 'rock', 'street',
    'valley', 'bridge',
]


# --- GTA5 / Cityscapes ---

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

CITYSCAPES_PROMPT_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person with clothes,people,human", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}


# --- SYNTHIA (16-class subset of Cityscapes, missing terrain/truck/train) ---

SYNTHIA_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "sky", "person", "rider", "car",
    "bus", "motorcycle", "bicycle",
]

SYNTHIA_PROMPT_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "sky", "person with clothes,people,human", "rider", "car",
    "bus", "motorcycle", "bicycle",
]

SYNTHIA_16_TO_CITYSCAPES_19 = {
    0: 0,    # road
    1: 1,    # sidewalk
    2: 2,    # building
    3: 3,    # wall
    4: 4,    # fence
    5: 5,    # pole
    6: 6,    # traffic light
    7: 7,    # traffic sign
    8: 8,    # vegetation
    9: 10,   # sky (skip terrain=9)
    10: 11,  # person
    11: 12,  # rider
    12: 13,  # car
    13: 15,  # bus (skip truck=14)
    14: 17,  # motorcycle (skip train=16)
    15: 18,  # bicycle
}
