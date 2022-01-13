import argparse
import os

import art
import numpy as np
from art.attacks.evasion import DPatch
from art.estimators.object_detection import PyTorchFasterRCNN
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
attack_iterations = 1000
training_iterations = 180

# Attack
attack = DPatch(
    frcnn,
    patch_shape=(80, 80, 3),
    learning_rate=1.0,
    max_iter=attack_iterations,
    verbose=True,
)
attack._targeted = True

def rgb_to_bgr(img):
    return img[:, :, ::-1]

def get_x(image):
    img = Image.open(image).convert('RGB')
    img = np.array(img).astype(np.float32)
    img = rgb_to_bgr(img)
    if len(img.shape) < 4:
        img = np.expand_dims(img, axis=0)
    print("img.shape:", img.shape)
    return img
        

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

def attack_dpatch(x):
    write_attack_config(run_root, attack)
    # Make prediction on beneign examples
    beneign_prediction_plots, pred_cls, pred_scores = make_predictions(frcnn, x)
    write_predictions(pred_cls, pred_scores, 'beneign_predictions.txt')
    for i in range(len(beneign_prediction_plots)):
        beneign_path = run_root + "x_{}".format(i)
        save_figure(beneign_prediction_plots[i], path=beneign_path)
    
    # Generate patch
    for epoch in range(training_iterations):
        start_time = datetime.now()
        print('\n----------- iteration {} -----------'.format(epoch))
        print('total training iterations: {}'.format(epoch*attack.max_iter))
        target_label = COCO_INSTANCE_CATEGORY_NAMES.index("toaster")
        patch = attack.generate(x=x, target_label=[target_label]*len(x))
        patch_path = run_root + "patch"
        np.save(os.path.join(run_root, "np_patch_{}".format(epoch)), attack._patch)
        # Apply patch to image,
        x_adv = attack.apply_patch(x=x)
        # And run prediction
        if (epoch % 10 == 0):
            adversarial_prediction_plots, pred_cls, pred_scores = make_predictions(frcnn, x_adv)
            for j in range(len(adversarial_prediction_plots)):
                adversarial_path = run_root + "x_adv_{}_{}".format(j, epoch)
                save_figure(adversarial_prediction_plots[j], path=adversarial_path)
                write_predictions(pred_cls, pred_scores, 'adversarial_predictions_{}.txt'.format(j), epoch*attack.max_iter)
        print("Epoch completed in: " + str(datetime.now() - start_time))
    save_figure(patch, path=patch_path)
    save_patch(attack._patch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=False, type=bool, help='Resume from an old patch')
    parser.add_argument('-p', '--patch', default='np_patch.npy', type=str, help='Path to patch to continue training')
    parser.add_argument('-e', '--epochs', default=0, type=int, help='Number of training iterations')
    parser.add_argument('-i', '--image', default='', type=str, help='Target image')
    
    args = parser.parse_args()
    
    if args.epochs:
        training_iterations = args.epochs
    
    if args.resume:
        patch = np.load(args.patch)
        attack._patch = patch

    x = get_x(args.image)

    if len(x.shape) != 4:
        print("Abort, x.shape = {}".format(x.shape))
        exit(0)

    os.makedirs(run_root)
    attack_dpatch(x)
    print("\nfinished run nr. {}".format(run_number))
    exit(1)
    
    
