from random import randint
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from datetime import datetime
today = datetime.today().strftime("%d-%m %H:%M:%S")

run_number = len(os.listdir("./results"))
run_root = "results/{}/".format(run_number)

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
def rgb_to_bgr(img):
    return img[:, :, ::-1]

def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]

def save_figure(fig, path=None):
    fig = fig.copy()
    fig = bgr_to_rgb(fig)
    plt.axis("off")
    plt.imshow(fig.astype(np.uint8), interpolation="nearest")
    plt.savefig(path, bbox_inches='tight')

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
        offset = 20
        # Label background
        labelSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_size, thickness=text_th)
        text_x2 = x1 + labelSize[0][0]
        text_y2 = y2 - int(labelSize[0][1])
        cv2.rectangle(img,(x1,y1),(text_x2,text_y2),(0,255,0),cv2.FILLED)
        # Write label
        #cv2.putText(img, text, (int(x1), int(y1)+offset), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), thickness=2*text_th)
        cv2.putText(img, text, (int(x1), int(y1)+offset), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), thickness=text_th)
    return img

def write_predictions(cls, conf, filename, iterations=0):
    with open(run_root + filename, 'a+') as f:
        f.write("\nIterations: {}\n".format(iterations))
        f.write("Classes:\t{}\n".format(cls))
        f.write("Confidence:\t{}\n".format(conf[:len(cls)]))

def extract_predictions(predictions_):
    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    # print("\npredicted classes({}):".format(len(predictions_class)), predictions_class)

    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    # print("predicted score:", predictions_score)
    
    write_predictions(predictions_class, predictions_score, 'all_adversarial_predictions.txt')
    
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
        
def make_predictions(model, images):
    predictions = model.predict(x=images)
    #print('predictions: {}'.format(predictions))
    #print('images.shape[0]: {}'.format(images.shape[0]))
    prediction_plots = []
    prediction_classes = []
    prediction_scores = []
    for i in range(images.shape[0]):
        # print("\nPredictions image {}:".format(i))

        # Process predictions
        predictions_class, predictions_boxes, predictions_score = extract_predictions(predictions[i])

        # Plot predictions
        prediction_plot = plot_image_with_boxes(
            img=images[i].copy(), boxes=predictions_boxes, pred_cls=predictions_class, pred_score=predictions_score
        )
        prediction_plots.append(prediction_plot)
        prediction_classes.append(predictions_class)
        prediction_scores.append(predictions_score)
    return prediction_plots, prediction_classes, prediction_scores

def save_images(imgs, path_prefix=None):
    if path_prefix == None:
        print("ERROR: No path_prefix included")
        return None
    
    for i in range(len(imgs)):
        save_path = path_prefix + "_{}.png".format(i)
        save_figure(imgs[i], save_path)
