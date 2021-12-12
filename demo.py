import numpy as np
import art
import os

from art.utils import load_dataset, load_mnist, load_stl
from art.attacks.evasion import DPatch
from art.estimators.object_detection import PyTorchFasterRCNN
from utils import COCO_INSTANCE_CATEGORY_NAMES, save_figure, write_attack_config, make_predictions
from PIL import Image

run_number = len(os.listdir("./results"))
run_root = "results/{}/".format(run_number)

# Model
frcnn = PyTorchFasterRCNN(
    clip_values=(0, 255),
    attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
)

# Attack
attack = DPatch(
    frcnn,
    patch_shape=(100, 100, 3),
    learning_rate=1.0,
    max_iter=1000,
    #batch_size=1,
    verbose=True,
)
attack._targeted = True

def rgb_to_bgr(img):
    return img[:, :, ::-1]

def get_x(dataset=None, n=0, img=None):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
        return x_test[:n].astype(np.float32)
    if dataset == 'stl':
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_stl()
        return x_test[:n].astype(np.float32)
    #image_paths = ['4.png', '2.png']
    #image_paths = os.listdir("images/")
    #images = []
    #for path in image_paths:
    #    img = Image.open("images/" + path).convert('RGB')
    #    img = img.resize((224, 224))
    #    img = np.array(img).astype(np.float32)
    #    img = rgb_to_bgr(img)
    #    images.append(img)
    #return np.array(images)
    img = img if img else "2.png"
    img = Image.open(img).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32)
    img = rgb_to_bgr(img)
    if len(img.shape) < 4:
        img = np.expand_dims(img, axis=0)
    print("img.shape:", img.shape)
    return img
        
def write_predictions(cls, conf, iterations=0, beneign=True):
    filename = 'adversarial_predictions.txt'
    if beneign:
        filename = 'beneign_predictions.txt'
    with open(run_root + filename, 'a+') as f:
        f.write("\nIterations: {}\n".format(iterations))
        f.write("Classes:\t{}\n".format(cls))
        f.write("Confidence:\t{}\n".format(conf[:len(cls)]))
    

def attack_dpatch(x):
    write_attack_config(run_root, attack)
    # Make prediction on beneign examples
    beneign_prediction_plots, pred_cls, pred_scores = make_predictions(frcnn, x)
    write_predictions(pred_cls, pred_scores)
    for i in range(len(beneign_prediction_plots)):
        beneign_path = run_root + "x_{}".format(i)
        save_figure(beneign_prediction_plots[i], path=beneign_path)
    # Generate patch
    # TODO: x[[0]] or just x?
    # TODO: target_label no effect?
    for i in range(9):
        print('\n----------- iteration {} -----------'.format(i))
        print('total training iterations: {}'.format(i*attack.max_iter))

        target_label = COCO_INSTANCE_CATEGORY_NAMES.index("cow")
        patch = attack.generate(x=x, target_label=[target_label])
        patch_path = run_root + "patch"
        np.save(os.path.join(run_root, "np_patch_{}".format(i)), attack._patch)
        # Apply patch to image,
        x_adv = attack.apply_patch(x=x)
        # And run prediction
        adversarial_prediction_plots, pred_cls, pred_scores = make_predictions(frcnn, x_adv)
        for j in range(len(adversarial_prediction_plots)):
            adversarial_path = run_root + "x_adv_{}_{}".format(j, i)
            save_figure(adversarial_prediction_plots[j], path=adversarial_path)
            write_predictions(pred_cls, pred_scores, i*attack.max_iter, beneign=False)
        save_figure(patch, path=patch_path)

if __name__ == "__main__":
    resume = True
    if resume:
        patch = np.load(os.path.join(".", "big_np_patch.npy"))
        attack._patch = patch
    # None, mnist or stl
    # dataset = 'stl'
    # dataset = 'mnist'  # NB! Patch-shape: (H, W, 1)
    dataset = None
    n = 2
    import sys
    img = None
    if len(sys.argv) > 0:
        img = sys.argv[1]

    x = get_x(dataset, n, img)
    print('x.shape: {}'.format(x.shape))
    if len(x.shape) != 4:
        print("Abort, x.shape = {}".format(x.shape))
        exit(0)
    os.makedirs(run_root)
    attack_dpatch(x)
    print("\nfinished run nr. {}".format(run_number))
    exit(1)
    
    