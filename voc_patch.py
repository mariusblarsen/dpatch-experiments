import os

import art
import math
import numpy as np
import argparse
from art.attacks.evasion import DPatch
from art.estimators.object_detection import PyTorchFasterRCNN
from art.utils import load_dataset, load_mnist, load_stl
from torch.utils.data import DataLoader
from PIL import Image
from datetime import datetime

from utils import (COCO_INSTANCE_CATEGORY_NAMES, make_predictions, save_figure,
                   write_attack_config, write_predictions)

run_number = len(os.listdir("./results"))
run_root = "results/{}/".format(run_number)

# Model
frcnn = PyTorchFasterRCNN(
    clip_values=(0, 255),
    attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
)

# Iterations
attack_iterations = 1
training_iterations = 1
batch_size = 25

# Attack
attack = DPatch(
    frcnn,
    patch_shape=(80, 80, 3),
    #learning_rate=1.0,
    max_iter=attack_iterations,
    batch_size=batch_size,
    verbose=True,
)
attack._targeted = True

def rgb_to_bgr(img):
    return img[:, :, ::-1]

def get_voc():
    voc_path = "data/VOCdevkit/VOC2007/JPEGImages/" 
    image_path = os.listdir(voc_path)
    images = []
    for image in image_path:
        img = Image.open(voc_path + image).convert('RGB')
        img = img.resize((416, 416))
        img = np.array(img).astype(np.float32)
        img = rgb_to_bgr(img)
        images.append(img)
    return np.array(images)

def save_patch(patch):
    """
    Saves patch in directory with patch configs
    """
    iterations = training_iterations * attack_iterations
    size = attack.patch_shape
    patch_dir = "patches/"
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    
    patch_num = len(os.listdir(patch_dir))
    patch_path = patch_dir + "{}_{}x{}_{}-iterations/".format(
        patch_num,
        size[0],
        size[1],
        iterations
        )
    os.makedirs(patch_path)
    patch_name = "np_patch"
    np.save(os.path.join(patch_path, patch_name), patch)
    
    with open(patch_path + 'patch_cfg.txt', 'w+') as f:
        f.write("patch_shape:\t{}\n".format(size))
        f.write("iterations:\t{}\n".format(iterations))    

def attack_dpatch(dataloader):
    total_samples = len(dataloader.dataset)
    total_steps = math.ceil(total_samples/batch_size)
    save_batches = 1
    
    write_attack_config(run_root, attack)
    image_counter = 0
    # Make prediction on beneign examples
    for i, x in enumerate(dataloader):
        if i >= save_batches:
            continue
        x = np.array(x)
        beneign_prediction_plots, pred_cls, pred_scores = make_predictions(frcnn, x)
        for j in range(len(beneign_prediction_plots)):
            write_predictions(pred_cls[j], pred_scores[j], 'beneign_predictions.txt')
            beneign_path = run_root + "x_{}".format(image_counter)
            save_figure(beneign_prediction_plots[j], path=beneign_path)
            image_counter+=1
    
    # Setup
    target_label = COCO_INSTANCE_CATEGORY_NAMES.index("toaster")
    patch_path = run_root + "patch"

    # Generate patch
    for epoch in range(training_iterations):
        start_time = datetime.now()
        print('\n----------- epoch {}/{} -----------'.format(epoch+1, training_iterations))
        print('cumulative training iterations: {}'.format(epoch*attack.max_iter*total_samples))
        image_counter = 0
        for i, x in enumerate(dataloader):        
            if (i + 1) % 50 == 0:
                print('\n\t------ step {}/{} ------'.format(i+1, total_steps))
            x = np.array(x)
            patch = attack.generate(x=x, target_label=[target_label]*len(x))
            np.save(os.path.join(run_root, "np_patch_{}".format(epoch)), attack._patch)
            # Apply patch to image,
            # And run prediction for the first n batches
            n = 1  # How many batches to save iamges from
            if i < n:
                x_adv = attack.apply_patch(x=x)
                adversarial_prediction_plots, pred_cls, pred_scores = make_predictions(frcnn, x_adv)
                for j in range(len(adversarial_prediction_plots)):
                    adversarial_path = run_root + "x_adv_{}_{}".format(image_counter, epoch)
                    save_figure(adversarial_prediction_plots[j], path=adversarial_path)
                    write_predictions(pred_cls[j], pred_scores[j], 'adversarial_predictions_{}.txt'.format(image_counter), i*attack.max_iter)
                    image_counter+=1
        print("Epoch completed in: " + str(datetime.now() - start_time))
    print('Total training iterations: {}'.format((epoch+1)*attack.max_iter*total_samples))
    save_figure(patch, path=patch_path)
    save_patch(attack._patch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=False, type=bool, help='Resume from an old patch')
    parser.add_argument('-p', '--patch', default='np_patch.npy', type=str, help='Path to patch to continue training')

    args = parser.parse_args()

    if args.resume:
        patch = np.load(args.patch)
        attack._patch = patch

    voc_dataset = get_voc()
    print('Fetched dataset of size: {}'.format(len(voc_dataset)))
    dataloader = DataLoader(
        dataset=voc_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    os.makedirs(run_root)
    attack_dpatch(dataloader)
    print("\nfinished run nr. {}".format(run_number))
    exit(1)
    
    