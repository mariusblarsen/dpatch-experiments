from random import randint
import numpy as np
import art
import os
import cv2

from tensorflow.keras.preprocessing import image
from art.utils import load_dataset, load_mnist, load_stl
from art.attacks.evasion import DPatch
from art.estimators.object_detection import PyTorchFasterRCNN
from utils import plot_image_with_boxes, extract_predictions, save_figure, write_attack_config
from datetime import datetime
today = datetime.today().strftime("%d-%m %H:%M:%S")

run_number = len(os.listdir("./results"))
run_root = "results/{}/".format(run_number)

# Model
frcnn = PyTorchFasterRCNN(
    clip_values=(0, 255),
    attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
)

# Attack
attack = DPatch(
    frcnn,
    patch_shape=(20, 20, 3),
    learning_rate=1.0,
    max_iter=10,
    #batch_size=1,
    verbose=True,
)

def rgb_to_bgr(img):
    return img[:, :, ::-1]

# TODO preprocess
# See step 1a from: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_feature_adversaries_pytorch.ipynb
#   - Transpose axes to PyTorchs NCHW Format
#   - Might only be for  MNIST with color-channel 1

def preprocess_image(img):
    # For using pretrained models in pytorch
    #   Source: https://github.com/jwyang/faster-rcnn.pytorch/issues/10
    pixel_means = [0.485, 0.456, 0.406]
    pixel_stdvs = [0.229, 0.224, 0.225]
    img -= pixel_means
    img /= 255.
    img /= pixel_stdvs
    img = img[:, :, ::-1]
    return img

def pytorch_preprocess_image(img_path):
    from torchvision import transforms
    from PIL import Image
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path)
    input_tensor = preprocess(img)
    return input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


def get_x(dataset=None, n=0, img=None):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
        return x_test[:n].astype(np.float32)
    if dataset == 'stl':
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_stl()
        return x_test[:n].astype(np.float32)

    img = img if img else "2.png"
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = rgb_to_bgr(img)
    if len(img.shape) < 4:
        img = np.expand_dims(img, axis=0)
    print("img.shape:", img.shape)
    return img

def make_predictions(model, images):
    predictions = model.predict(x=images)
    print('predictions: {}'.format(predictions))
    print('images.shape[0]: {}'.format(images.shape[0]))
    prediction_plots = []
    for i in range(images.shape[0]):
        print("\nPredictions image {}:".format(i))

        # Process predictions
        predictions_class, predictions_boxes, predictions_score = extract_predictions(predictions[i])

        # Plot predictions
        prediction_plot = plot_image_with_boxes(
            img=images[i].copy(), boxes=predictions_boxes, pred_cls=predictions_class, pred_score=predictions_score
        )
        prediction_plots.append(prediction_plot)
    return prediction_plots, predictions_class, predictions_score
        
def write_predictions(cls, conf, beneign=True):
    filename = 'adversarial_predictions.txt'
    if beneign:
        filename = 'beneign_predictions.txt'
    with open(run_root + filename, 'w+') as f:
        f.write("Classes:\t{}\n".format(cls))
        f.write("Conficedence:\t{}\n".format(conf))
    

def attack_dpatch(x):
    # Make prediction on beneign examples
    beneign_prediction_plots, pred_cls, pred_scores = make_predictions(frcnn, x)
    write_predictions(pred_cls, pred_scores)
    for i in range(len(beneign_prediction_plots)):
        beneign_path = run_root + "x_{}".format(i)
        save_figure(beneign_prediction_plots[i], path=beneign_path)
    
    # Generate patch
    # TODO: x[[0]] or just x?
    # TODO: target_label no effect?
    patch = attack.generate(x=x)
    bgr_patch = rgb_to_bgr(patch)
    patch_path = run_root + "patch"
    save_figure(patch, path=patch_path)
    
    # Apply patch to image,
    # And run prediction
    x_adv = attack.apply_patch(x=x, patch_external=bgr_patch)
    adversarial_prediction_plots, pred_cls, pred_scores = make_predictions(frcnn, x_adv)
    for i in range(len(adversarial_prediction_plots)):
        adversarial_path = run_root + "x_adv_{}".format(i)
        save_figure(adversarial_prediction_plots[i], path=adversarial_path)
    write_predictions(pred_cls, pred_scores, beneign=False)
    write_attack_config(run_root, attack)

if __name__ == "__main__":
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
    if len(x.shape) != 4:
        print("Abort, x.shape = {}".format(x.shape))
        exit(0)
    os.makedirs(run_root)
    attack_dpatch(x)
    print("\nfinished run nr. {}".format(run_number))
    exit(1)
    
    