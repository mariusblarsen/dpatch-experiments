import os
import argparse

import numpy as np
from art.attacks.evasion import DPatch
from art.estimators.object_detection import PyTorchFasterRCNN
from PIL import Image
from torch.utils.data import DataLoader
from utils import save_figure, plot_image_with_boxes, COCO_INSTANCE_CATEGORY_NAMES

def rgb_to_bgr(img):
    return img[:, :, ::-1]


def NMS(boxes, overlapThresh = 0.4):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    #return only the boxes at the remaining indices
    return boxes[indices].astype(int)

def get_voc(split):
    if split == "training":
        voc_path = "data/VOCdevkit/VOC2007/JPEGImages/" 
    else:
        voc_path = "data/test/" 
        
    image_path = os.listdir(voc_path)[:25]
    images = []
    for image in image_path:
        img = Image.open(voc_path + image).convert('RGB')
        img = img.resize((416, 416))
        img = np.array(img).astype(np.float32)
        img = rgb_to_bgr(img)
        images.append(img)
    return np.array(images)

def extract_predictions(predictions_):
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]
    predictions_score = list(predictions_["scores"])
    
    keep = NMS(predictions_boxes)
    
    # Get a list of index with score greater than threshold
    threshold = 0.5
    try:
        predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

        keep = keep[: predictions_t + 1]
        predictions_class = predictions_class[: predictions_t + 1]
    except IndexError:
        pass
    return predictions_class, keep, predictions_score

def make_predictions(model, images):
    predictions = model.predict(x=images)
    prediction_plots = []
    for i in range(images.shape[0]):
        # Process predictions
        predictions_class, predictions_boxes, predictions_score = extract_predictions(predictions[i])

        # Plot predictions
        prediction_plot = plot_image_with_boxes(
            img=images[i].copy(), boxes=predictions_boxes, pred_cls=predictions_class, pred_score=predictions_score
        )
        prediction_plots.append(prediction_plot)
    return prediction_plots, predictions_class

run_number = len(os.listdir("./test"))
run_root = "test/{}/".format(run_number)

# Model
frcnn = PyTorchFasterRCNN(
    clip_values=(0, 255),
    attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
)

# Iterations
batch_size = 50

# Attack
attack = DPatch(
    frcnn,
    patch_shape=(80, 80, 3),
    verbose=True
)


# Run predictions

# Save:
#   Clean beneign image

#   Clean adversarial image

def evaluate(dataloader):
    # target = "toaster"  # TODO: Pass as arg?
    for i, x in enumerate(dataloader):
        x = np.array(x)
        x_adv = attack.apply_patch(x=x)
        # target_in_beneign = False
        beneign_prediction_plots, beneign_predictions_class = make_predictions(frcnn, x)    
        
        #TODO: Find if target in beneign
        #TODO: Count if target in adversarial
            
        adversarial_prediction_plots, adversarial_predictions_class = make_predictions(frcnn, x_adv)
    
        for j in range(len(adversarial_prediction_plots)):
            beneign_path = run_root + "{}".format(j)
            adversarial_path = run_root + "{}_adv".format(j)
            
            save_figure(beneign_prediction_plots[j], path=beneign_path)
            save_figure(adversarial_prediction_plots[j], path=adversarial_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', default='np_patch.npy', type=str, help='Path to patch to continue training')
    args = parser.parse_args()
    
    # Load patch
    patch = np.load(args.patch)
    attack._patch = patch
    
    voc_training = get_voc("training")
    # voc_validation = get_voc("validation")

    training_dataloader = DataLoader(
        dataset=voc_training,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    
    # validation_dataloader = DataLoader(
    #     dataset=voc_validation,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=1
    # )
    
    os.makedirs(run_root)
    evaluate(training_dataloader)
    # evaluate(validation_dataloader)
  
    