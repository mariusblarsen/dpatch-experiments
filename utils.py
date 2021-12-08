from random import randint
import matplotlib.pyplot as plt
import cv2
import numpy as np
from datetime import datetime
today = datetime.today().strftime("%d-%m %H:%M:%S")

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]

def save_figure(fig, path=None):
    fig = fig.copy()
    fig = bgr_to_rgb(fig)
    # TODO: Color back to RGB
    plt.axis("off")
    plt.imshow(fig.astype(np.uint8), cmap='gray', interpolation="nearest")
    #plt.imshow(fig, cmap='binary')
    #plt.imshow(fig, cmap='gray')
    plt.savefig(path)

def plot_image_with_boxes(img, boxes=[], pred_cls=[], pred_score=[]):
    text_size = 1
    text_th = 1
    rect_th = 1

    for i in range(len(boxes)):
        # Draw Rectangle with the coordinates
        x1, y1 = boxes[i][0]
        x2, y2 = boxes[i][1]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=rect_th)

        # Write the prediction class
        text = "{} {:.2f}".format(pred_cls[i], pred_score[i])
        cv2.putText(img, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    return img


def extract_predictions(predictions_):
    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    print("\npredicted classes:", predictions_class)

    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    threshold = 0.5
    try:
        predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

        predictions_boxes = predictions_boxes[: predictions_t + 1]
        predictions_class = predictions_class[: predictions_t + 1]
    except IndexError:
        pass
    return predictions_class, predictions_boxes, predictions_score

def write_attack_config(root, attack):
    with open(root + 'attack_cfg.txt', 'w+') as f:
        f.write("attack.patch_shape:\t{}\n".format(attack.patch_shape))
        f.write("attack.learning_rate:\t{}\n".format(attack.learning_rate))
        f.write("attack.max_iter:\t{}\n".format(attack.max_iter))
        f.write("attack.batch_size:\t{}\n".format(attack.batch_size))
        f.write("attack.estimator:\n{}".format(attack.estimator))