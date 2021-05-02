import os

import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from six import BytesIO
import numpy as np

import datetime

import random


import matplotlib
import matplotlib.image
import matplotlib.pyplot as plt

from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

#matplotlib.use( 'tkagg' )

category_index = {
  0: {'id': 0, 'name': 'speed limit 20 (prohibitory)'},
  1: {'id': 1, 'name': 'speed limit 30 (prohibitory)'},
  2: {'id': 2, 'name': 'speed limit 50 (prohibitory)'},
  3: {'id': 3, 'name': 'speed limit 60 (prohibitory)'},
  4: {'id': 4, 'name': 'speed limit 70 (prohibitory)'},
  5: {'id': 5, 'name': 'speed limit 80 (prohibitory)'},
  6: {'id': 6, 'name': 'restriction ends 80 (other)'},
  7: {'id': 7, 'name': 'speed limit 100 (prohibitory)'},
  8: {'id': 8, 'name': 'speed limit 120 (prohibitory)'},
  9: {'id': 9, 'name': 'no overtaking (prohibitory)'},

  10: {'id': 10, 'name': 'no overtaking (trucks) (prohibitory)'},
  11: {'id': 11, 'name': 'priority at next intersection (danger)'},
  12: {'id': 12, 'name': 'priority road (other)'},
  13: {'id': 13, 'name': 'give way (other)'},
  14: {'id': 14, 'name': 'stop (other)'},
  15: {'id': 15, 'name': 'no traffic both ways (prohibitory)'},
  16: {'id': 16, 'name': 'no trucks (prohibitory)'},
  17: {'id': 17, 'name': 'no entry (other)'},
  18: {'id': 18, 'name': 'danger (danger)'},
  19: {'id': 19, 'name': 'bend left (danger)'},

  20: {'id': 20, 'name': 'bend right (danger)'},
  21: {'id': 21, 'name': 'bend (danger)'},
  22: {'id': 22, 'name': 'uneven road (danger)'},
  23: {'id': 23, 'name': 'slippery road (danger)'},
  24: {'id': 24, 'name': 'road narrows (danger)'},
  25: {'id': 25, 'name': 'construction (danger)'},
  26: {'id': 26, 'name': 'traffic signal (danger)'},
  27: {'id': 27, 'name': 'pedestrian crossing (danger)'},
  28: {'id': 28, 'name': 'school crossing (danger)'},
  29: {'id': 29, 'name': 'cycles crossing (danger)'},

  30: {'id': 30, 'name': 'snow (danger)'},
  31: {'id': 31, 'name': 'animals (danger)'},
  32: {'id': 32, 'name': 'restriction ends (other)'},
  33: {'id': 33, 'name': 'go right (mandatory)'},
  34: {'id': 34, 'name': 'go left (mandatory)'},
  35: {'id': 35, 'name': 'go straight (mandatory)'},
  36: {'id': 36, 'name': 'go right or straight (mandatory)'},
  37: {'id': 37, 'name': 'go left or straight (mandatory)'},
  38: {'id': 38, 'name': 'keep right (mandatory)'},
  39: {'id': 39, 'name': 'keep left (mandatory)'},

  40: {'id': 40, 'name': 'roundabout (mandatory)'},
  41: {'id': 41, 'name': 'restriction ends (overtaking) (other)'},
  42: {'id': 42, 'name': 'restriction ends (overtaking (trucks)) (other)'}
}

num_classes = len(category_index)
batch_size = 100
learning_rate = 0.001
num_batches = 1000


class Box:
  def __init__(self, left, top, right, bottom, category):
    self.left = left
    self.top = top
    self.right = right
    self.bottom = bottom
    self.category = category
    self.gt_box = np.array([[top, left, bottom , right]], dtype=np.float32)

  def getGtBoxTensor(self):
    self.gt_box_tensor = tf.convert_to_tensor(self.gt_box, dtype=tf.float32)
    return self.gt_box_tensor

  def getGtClass(self):
    zero_indexed_groundtruth_classes = tf.convert_to_tensor(
      np.full(
        shape=[self.gt_box.shape[0]],
        fill_value=self.category,
        dtype=np.int32
      )
    )
    self.gt_classes_one_hot_tensor = tf.one_hot(
      zero_indexed_groundtruth_classes,
      num_classes
    )
    return self.gt_classes_one_hot_tensor


class TrainImage:
  def __init__(self, filename):
    self.filename = filename
    # img_data = tf.io.gfile.GFile(filename, 'rb').read()
    # image = Image.open(BytesIO(img_data))
    # (self.width, self.height) = image.size
    # np_image = self.getNpImage(image)
    # image.close()
    self.np_image = matplotlib.image.imread(filename)
    (self.height, self.width, _) = self.np_image.shape
    self.boxes = []
    self.train_image_tensor = self.getTensor()

  def addGtBox(self, left, top, right, bottom, category):
    box = Box(
      left / self.width,
      top / self.height,
      right / self.width,
      bottom / self.height,
      category
    )
    self.boxes.append(box)

  def getNpImage(self, image):
    return np.array(image.getdata()).reshape(
      (self.height, self.width, 3)).astype(np.uint8)

  def getTensor(self):
    return tf.expand_dims(
      tf.convert_to_tensor(self.np_image, dtype=tf.float32),
      axis=0
    )


def loadMetadata(path):
  images = {}
  gt = open(path + "/gt.txt")
  lines = gt.readlines()

  for line in lines:
    (filename, left, top, right, bottom, category) = line.split(";")
    cat = int(category)
    fullFileName = path + "/" + filename
    image = None
    if (filename not in images):
      images[filename] = TrainImage(fullFileName)
    image = images[filename]
    image.addGtBox(int(left), int(top), int(right), int(bottom), cat)
    print("loaded image", fullFileName)
  return images


def plot_detection(image, box, classes, scores, category_index, fig_size=(12, 16)):
  np_image_with_annotations = image.np_image.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
    np_image_with_annotations,
    box,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=True,
    min_score_thresh=0.8
  )
  plt.imshow(np_image_with_annotations)


images = loadMetadata("TrainIJCNN2013")

dummy_scores = np.array([1.0], dtype=np.float32)  # give boxes a score of 100%



tf.keras.backend.clear_session()

print('Building model and restoring weights for fine-tuning...', flush=True)
pipeline_config = 'models/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config'
checkpoint_path = 'models/research/object_detection/test_data/checkpoint/ckpt-0'

output_directory = 'output/'
output_checkpoint_dir = os.path.join(output_directory, 'checkpoint')
configs = config_util.get_configs_from_pipeline_file(pipeline_config)

model_config = configs['model']
model_config.ssd.num_classes = num_classes
model_config.ssd.freeze_batchnorm = True
detection_model = model_builder.build(
      model_config=model_config, is_training=True)
# Save new pipeline config
pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
config_util.save_pipeline_config(pipeline_proto, output_directory)
fake_box_predictor = tf.compat.v2.train.Checkpoint(
    _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
    # _prediction_heads=detection_model._box_predictor._prediction_heads,
    #    (i.e., the classification head that we *will not* restore)
    _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
fake_model = tf.compat.v2.train.Checkpoint(
          _feature_extractor=detection_model._feature_extractor,
          _box_predictor=fake_box_predictor)
ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
ckpt.restore(checkpoint_path).expect_partial()

# To save checkpoint for TFLite conversion.
exported_ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt_manager = tf.train.CheckpointManager(
    exported_ckpt, output_checkpoint_dir, max_to_keep=1)

# Run model through a dummy image so that variables are created
image, shapes = detection_model.preprocess(tf.zeros([1, 320, 320, 3]))
prediction_dict = detection_model.predict(image, shapes)
_ = detection_model.postprocess(prediction_dict, shapes)
print('Weights restored!')



trainable_variables = detection_model.trainable_variables
to_fine_tune = []

prefixes_to_train = [
  'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
  'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']

for var in trainable_variables:
  if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
    to_fine_tune.append(var)


def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
  """Get a tf.function for training step."""

  # Use tf.function for a bit of speed.
  # Comment out the tf.function decorator if you want the inside of the
  # function to run eagerly.
  @tf.function
  def train_step_fn(image_tensors,
                    groundtruth_boxes_list,
                    groundtruth_classes_list):
    """A single training iteration.

    Args:
      image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
        Note that the height and width can vary across images, as they are
        reshaped within this function to be 320x320.
      groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
        tf.float32 representing groundtruth boxes for each image in the batch.
      groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
        with type tf.float32 representing groundtruth boxes for each image in
        the batch.

    Returns:
      A scalar tensor representing the total loss for the input batch.
    """
    shapes = tf.constant(len(image_tensors) * [[320, 320, 3]], dtype=tf.int32)
    model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_boxes_list,
        groundtruth_classes_list=groundtruth_classes_list)
    with tf.GradientTape() as tape:
      sss = [detection_model.preprocess(image_tensor)[0]
           for image_tensor in image_tensors]
      preprocessed_images = tf.concat(sss, axis=0)
      prediction_dict = model.predict(preprocessed_images, shapes)
      losses_dict = model.loss(prediction_dict, shapes)
      total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
      gradients = tape.gradient(total_loss, vars_to_fine_tune)
      optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
    return total_loss

  return train_step_fn


# i = 1
# plt.figure(figsize=(20, 10))
# for key in images:
#   print("getting key", key)
#   image = images[key]
#   print("boxes of image", key, image.boxes)
#   for box in image.boxes:
#     plt.subplot(4, 5, i)
#     i = i + 1
#     print("image", box.category, key)
#     plot_detection(
#       image,
#       box.gt_box,
#       np.full(shape=[box.gt_box.shape[0]], fill_value=int(box.category), dtype=np.int32),
#       dummy_scores,
#       category_index
#       )
# plt.show()


train_image_tensors = []
train_boxes = []
train_classes = []

for img_key in images:
  image = images[img_key]
  for box in image.boxes:
    train_image_tensors.append(image.train_image_tensor)
    train_boxes.append(box.getGtBoxTensor())
    train_classes.append(box.getGtClass())

print("train_image_tensors", len(train_image_tensors))
print("train_boxes", len(train_boxes))
print("train_classes", len(train_classes))


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_fn = get_model_train_step_function(
    detection_model, optimizer, to_fine_tune)


print("starging training")

for idx in range(num_batches):
  # Grab keys for a random subset of examples
  all_keys = list(range(len(train_image_tensors)))
  random.shuffle(all_keys)
  example_keys = all_keys[:batch_size]

  # Note that we do not do data augmentation in this demo.  If you want a
  # a fun exercise, we recommend experimenting with random horizontal flipping
  # and random cropping :)
  gt_boxes_list = [train_boxes[key] for key in example_keys]
  gt_classes_list = [train_classes[key] for key in example_keys]
  image_tensors = [train_image_tensors[key] for key in example_keys]

  # Training step (forward pass + backwards pass)
  total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

  print(datetime.datetime.now().isoformat(), 'batch ' + str(idx) + ' of ' + str(num_batches)
    + ', loss=' +  str(total_loss.numpy()), flush=True)

print('Done fine-tuning!')

ckpt_manager.save()
print('Checkpoint saved!')
