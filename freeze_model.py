import keras.backend as K
import tensorflow as tf

# I needed to add this
sess = tf.Session()
K.set_session(sess)

from mrcnn import model as modellib
from mrcnn.config import Config
# my config subclass
#from network_configs import ExampleConfig


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = sess.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))

        output_names = output_names or []
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def freeze_model(model, name):
    frozen_graph = freeze_session(
        sess,
        output_names=[out.op.name for out in model.outputs][:4])
    directory = './'
    tf.train.write_graph(frozen_graph, directory, name + '.pb', as_text=False)

class SPConfig(Config):
    # Give the configuration a recognizable name
    NAME = "sp"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 60% confidence
    DETECTION_MIN_CONFIDENCE = 0.6

config = SPConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
MODEL_DIR = 'logs'
H5_WEIGHT_PATH = 'logs/idv1703/mask_rcnn_sp_0020_17_03.h5'
FROZEN_NAME = 'frozen_17_03_graph'
model = modellib.MaskRCNN(
    mode="inference",
    config=config,
    model_dir=MODEL_DIR)
model.load_weights(H5_WEIGHT_PATH, by_name=True)
freeze_model(model.keras_model, FROZEN_NAME)
