import os
import argparse

import numpy as np
from art.attacks.evasion import DPatch
from art.estimators.object_detection import PyTorchFasterRCNN
from PIL import Image
from torch.utils.data import DataLoader
from utils import make_predictions, save_figure

def rgb_to_bgr(img):
    return img[:, :, ::-1]

def get_voc():
    voc_path = "data/test/" 
    image_path = os.listdir(voc_path)[:50]
    images = []
    for image in image_path:
        img = Image.open(voc_path + image).convert('RGB')
        img = img.resize((416, 416))
        img = np.array(img).astype(np.float32)
        img = rgb_to_bgr(img)
        images.append(img)
    return np.array(images)

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
        adversarial_prediction_plots, _, _ = make_predictions(frcnn, x_adv)
        print("d")
        image_counter = 0
        print("e")
        for j in range(len(adversarial_prediction_plots)):
            print("f")
            adversarial_path = run_root + "x_adv_{}".format(image_counter)
            save_figure(adversarial_prediction_plots[j], path=adversarial_path)
            image_counter+=1