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

def get_voc():
    voc_path = "data/test/" 
    image_path = os.listdir(voc_path)[:50]
    images = []
    for image in image_path:
        img = Image.open(voc_path + image).convert('RGB')
        #img = img.resize((416, 416))
        img = np.array(img).astype(np.float32)
        img = rgb_to_bgr(img)
        images.append(img)
    return np.array(images)

def extract_predictions(predictions_):
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]
    predictions_score = list(predictions_["scores"])
    
    # Get a list of index with score greater than threshold
    threshold = 0.5
    try:
        predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

        predictions_boxes = predictions_boxes[: predictions_t + 1]
        predictions_class = predictions_class[: predictions_t + 1]
    except IndexError:
        pass
    return predictions_class, predictions_boxes, predictions_score

def make_predictions(model, images):
    predictions = model.predict(x=images)
    #print('predictions: {}'.format(predictions))
    #print('images.shape[0]: {}'.format(images.shape[0]))
    prediction_plots = []
    for i in range(images.shape[0]):
        # print("\nPredictions image {}:".format(i))

        # Process predictions
        predictions_class, predictions_boxes, predictions_score = extract_predictions(predictions[i])

        # Plot predictions
        prediction_plot = plot_image_with_boxes(
            img=images[i].copy(), boxes=predictions_boxes, pred_cls=predictions_class, pred_score=predictions_score
        )
        prediction_plots.append(prediction_plot)
    return prediction_plots

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
#   Beneign predictions
#   Clean adversarial image
#   Adversarial predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', default='np_patch.npy', type=str, help='Path to patch to continue training')
    args = parser.parse_args()
    
    # Load patch
    patch = np.load(args.patch)
    attack._patch = patch
    
    voc_dataset = get_voc()
    print('Fetched dataset of size: {}'.format(len(voc_dataset)))
    dataloader = DataLoader(
        dataset=voc_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    os.makedirs(run_root)
    for i, x in enumerate(dataloader):
        x = np.array(x)
        x_adv = attack.apply_patch(x=x)
        adversarial_prediction_plots = make_predictions(frcnn, x_adv)
        image_counter = 0
        for j in range(len(adversarial_prediction_plots)):
            adversarial_path = run_root + "x_adv_{}".format(image_counter)
            save_figure(adversarial_prediction_plots[j], path=adversarial_path)
            image_counter+=1