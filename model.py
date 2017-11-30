#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
import datetime
import random
import time
import json
import sys
import os

sys.path.append("../common")
from utils import Utils as U
from config import Config
from logger import Logger

log = Logger("Model")


class Model(object):
    """
        Class used to build the graph/model
    """

    # Range value
    # Min inception value: 0.0, max_inception_values: 9.81479549408,
    # Max_camera_position: 2.26729202271, min_camera_position: -2.49186825752, max_faces_value: 3

    # Min inception value
    MIN_IV = 0.0
    # Max inception value
    MAX_IV = 10.
    # Max camera position
    MAX_CP = 3.
    # Min camera position
    MIN_CP = -3.
    # Max face value
    MAX_FV = 3

    INPUT_SIZE = 2056

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINTS_FOLDER = os.path.join(CURRENT_DIR, Config.CHECKPOINTS_FOLDER)

    HYPERPARAMETERS_CONFIG_PATH = os.path.join(
        CURRENT_DIR, "settings/default_hyperparameters.json")
    ATTENTION_HYPERPARAMETERS_CONFIG_PATH = os.path.join(
        CURRENT_DIR, "settings/attention_hyperparameters.json")

    def __init__(self, file_hyperparameters_config=None, hyperparameters_config=None, seed=None,
                 group_name="unknow", training=False):
        """
            **input: **
                *file_hyperparameters_config (String) Settings file path used to set hyperparameters
                    used to train the model.

                *hyperparameters_config (Dict) Hyperparameters used to train the model.

                *file_hyperparameters_config and *hyperparameters_config are similar.
                One set hyperparameters inside a file, the other one set hyperparameters inside
                a Python dict. However, you can set only one of both.

                *seed: (Integer|None)  None by default. Otherwise this value will be used to set the
                random seed of the model. This is used to generate pseudo-random numbers.7
                TODO : Same seed for DropoutWrapper

                *training (Boolean) If true, the graph is used for training,
                otherwise, for inference.
        """
        super(Model, self).__init__()

        if file_hyperparameters_config is None:
            file_hyperparameters_config = self.HYPERPARAMETERS_CONFIG_PATH

        # Tensorflow session
        self.session = None

        # Compute delta of features between the maximum value and the minimum
        # Delta camera positions
        delta_cp = self.MAX_CP - self.MIN_CP
        # delta inception values
        delta_iv = self.MAX_IV - self.MIN_IV
        # Formula used to normalize each camera position to the same range of inception values
        self.normalize_camera_position = lambda c: (
            self.MIN_IV + ((c - self.MIN_CP) / delta_cp * delta_iv))
        # Formula used to normalize each face value to the same range of inception values
        self.normalize_faces = lambda f: self.MIN_IV + (f / self.MAX_FV * (delta_iv))

        self.group_name = group_name
        self.seed = seed
        if self.seed is not None:
            self._set_seed()

        self.training = training

        # Load hyperparameters from the dict
        if hyperparameters_config is not None:
            self.hyperparameters = hyperparameters_config
        else:
            # Load hyperparameters from the file
            self.hyperparameters = self._load_hyperparameters(file_hyperparameters_config)

        # TODO : We should not need this parameters
        self.batch_size = 1 if not self.training else self.hyperparameters["batch_size"]
        # Create a name from hyperparameters
        self._set_hyperparameters_name()
        # Set model names
        self._set_names()

    def _set_names(self):
        """
            Set all model names
        """
        name_time = time.time()
        # model_name is used to set the ckpt name
        self.model_name = "%s--%s" % (self.hyperparameters_name, name_time)
        # sub_train_log_name is used to set the name of the training part in tensorboard
        self.sub_train_log_name = "%s-train--%s" % (self.hyperparameters_name, name_time)
        # sub_test_log_name is used to set the name of the testing part in tensorboard
        self.sub_test_log_name = "%s-test--%s" % (self.hyperparameters_name, name_time)

    def _set_hyperparameters_name(self):
        """
            Convert hyperparameters dict to a string
            This string will be used to set the models names
        """
        hyperparameters_names = [
            ("gn", self.group_name),
            ("bs", self.hyperparameters["batch_size"]),
            ("ss", self.hyperparameters["sequence_size"]),
            ("hs", self.hyperparameters["hidden_layer_size"]),
            ("lr", self.hyperparameters["learning_rate"]),
            ("idp", self.hyperparameters["input_dropout"]),
            ("odp", self.hyperparameters["output_dropout"]),
            ("ds", self.hyperparameters["deep_size"]),
        ]

        self.hyperparameters_name = ""
        for index_hyperparameter, hyperparameter in enumerate(hyperparameters_names):
            short_name, value = hyperparameter
            prepend = "" if index_hyperparameter == 0 else "_"
            self.hyperparameters_name += "%s%s_%s" % (prepend, short_name, value)

    def get_features(self, obj, path, flip=False, low_brightness=False, high_brightness=False,
                     low_contrast=False, high_contrast=False):
        """
            Get features of an image.

            **input:**
                *obj: (dict) Dict associated with an image inside the dataset json file.
                *path (String) Path to the frame
                *flip (Boolean) Used to flip the image
                *low_brightness (Boolean) Used to change the image brightness. False by default.
                *high_brightness (Boolean) Used to change the image brightness. False by default.
                *low_contrast (Boolean) Used to change the image contrast. False by default.
                *high_contrast (Boolean) Used to change the image contrast. False by default.
            **return:(Tuple (Numpy array, Numpy array)) **
                *First tuple element: All features concatenated together
                *Second tuple element: Only inception features
        """
        # Retrive inception values from inception model
        inception_f = self.inception.get_features_from_path(
            path, reload_data=False, flip=flip, low_brightness=low_brightness,
            high_brightness=high_brightness, low_contrast=low_contrast,
            high_contrast=high_contrast)

        camera_position = np.array(obj["info"]["camera_position"])
        hidden = float(obj["info"]["hidden"])
        view = float(obj["info"]["view"])

        # Normalize camera position in the same range than inception features
        camera_position = self.normalize_camera_position(camera_position)
        # Do the same for hidden and view faces
        hidden = self.normalize_faces(hidden)
        view = self.normalize_faces(view)

        # Concat all features together
        final_features = np.concatenate(
            [inception_f, camera_position, np.array([hidden]), np.array([view])])

        return final_features, inception_f

    def _set_seed(self):
        """
            Set random seed
        """
        log.info("Set seed: %s" % self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def set_hyperparameters(self, hyperparameters_config):
        """
            Set new hyperparameters on this model
            **input: **
                *hyperparameters_config (Dict)
        """
        self._set_hyperparameters(hyperparameters_config)

    def _set_hyperparameters(self, hyperparameters_config):
        """
            Set new hyperparameters on this model
            **input: **
                *hyperparameters_config (Dict) Dictionary with new hyperparameters
                    Example : {"epoch": 100, ... "sequence_size": 17}
        """
        log.info("Set new hyperparameters: %s" % hyperparameters_config)
        self.hyperparameters = hyperparameters_config
        self.batch_size = 1 if not self.training else self.hyperparameters["batch_size"]
        self._set_hyperparameters_name()
        # Since hyperparameters had changed, we need to set again each name
        self._set_names()
        self.reset_model()
        # Init the graph with new parameters
        self.init_model()

    def _load_hyperparameters(self, path):
        """
            Load the content of the hyperparameters file (json content)
            **input: **
                *path (String) path to the file
            **return: (Dict) **
                Return all hyperparameters
        """
        settings = None

        with open(path, "r") as m:
            settings = json.loads(m.read())
            m.close()

        return settings

    def build_inputs(self):
        """
            Build tensorflow inputs
            **return (Tuple (... 5 elements)) **
                *tf_x : Graph inputs
                *tf_y : Graph targets
                *tf_py : Graph predictions
                *output_keep_prob && *input_keep_prob : Float value between 0. and 1
                applying on each cell on the dropout.
        """
        with tf.name_scope("placeholder"):
            tf_x = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.hyperparameters["sequence_size"], self.INPUT_SIZE],
                name="inputs")
            tf_y = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.hyperparameters["output_size"]],
                name="targets")
            tf_py = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.hyperparameters["output_size"]],
                name="predictions")
            output_keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob_dropout_output")
            input_keep_prob = tf.placeholder(tf.float32, name="keep_prob_dropout_input")

        return tf_x, tf_y, tf_py, output_keep_prob, input_keep_prob

    def build_lstm(self, tf_x, output_keep_prob, input_keep_prob):
        """
            Build LSTM (RNN)
            **input: **
                *tf_x (tf.placeholder) Graph input
                *output_keep_prob (tf.placeholder) Dropout value applying on each cell output
                *input_keep_prob (tf.placeholder) Dropout value applying on each cell input
            **return (Tuple (three tensor)) **
                *lstm_output : Output of each cell
                *final_state : Final state of each cell
                *initial_state : Initial state of each cell
        """
        with tf.name_scope("LSTM"):

            def create_cell_first_layer():
                """
                    Create the first cell
                """
                # On the first layer, the dropout in added on the inputs and on the outputs
                cell = rnn.BasicLSTMCell(num_units=self.hyperparameters["hidden_layer_size"])
                dropout = tf.contrib.rnn.DropoutWrapper(
                    cell=cell,
                    input_keep_prob=input_keep_prob,
                    output_keep_prob=output_keep_prob)
                return dropout

            def create_cell_second_layer():
                """
                    Create the cell stacked on the first cell
                """
                # For the second layer, we only apply the dropout on the output.
                # Since the dropout have already been apply on the inputs from the previous layer
                cell = rnn.BasicLSTMCell(num_units=self.hyperparameters["hidden_layer_size"])
                dropout = tf.contrib.rnn.DropoutWrapper(
                    cell=cell, output_keep_prob=output_keep_prob)
                return dropout

            cells = [create_cell_first_layer(), create_cell_second_layer()]
            cells = tf.contrib.rnn.MultiRNNCell(cells=cells)
            initial_state = cells.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            lstm_output, final_state = tf.nn.dynamic_rnn(cells, tf_x, initial_state=initial_state)

        return lstm_output, final_state, initial_state

    def build_outputs(self, lstm_output):
        """
            Build graph output
            **input: **
                *lstm_output (tf.tensor) Output of each cell
            **return (Tuple (two tensors)) **
                *linear_output: Logit output
                *softmax : Softmax output
        """
        with tf.name_scope("outputs"):
            # Weights of the final layer
            weights = tf.Variable(
                tf.random_normal([self.hyperparameters["hidden_layer_size"],
                                  self.hyperparameters["output_size"]]))
            # Bias of the final layer
            bias = tf.Variable(tf.random_normal([self.hyperparameters["output_size"]]))
            # Log each of theses value in tensorboard
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("bias", bias)
            # The final layer is applied only on the last cell of the sequence.
            lstm_output = lstm_output[:, -1]
            # Compute the logits output
            linear_output = tf.matmul(lstm_output, weights) + bias
            # Compute the softmax
            softmax = tf.nn.softmax(linear_output)
            # Log the softmax in tensorboard
            tf.summary.histogram("softmax", softmax)

        return linear_output, softmax

    @staticmethod
    def build_errors(tf_y, tf_py, linear_py):
        """
            Build part of the graph used to measure the error
            **input: **
                *tf_y (tf.tensor)
                *tf_py (tf.tensor)
                *linear_py (tf.tensor)
            **return (Tuple (tf.tensor, tf.tensor, tf.tensor, tf.tensor))
        """
        with tf.name_scope("errors"):
            test_correct_prediction = tf.equal(tf.argmax(tf_py, 1), tf.argmax(tf_y, 1))
            correct_prediction = tf.equal(tf.argmax(linear_py, 1), tf.argmax(tf_y, 1))

            # There is in this method two cost and two accuracy (x, test_x)
            # The first one  compute errors metrics given the model output
            # The second one compute errors metrics given a user input (from the placeholder)

            test_cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=tf_py, labels=tf_y))
            test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=linear_py, labels=tf_y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Load the cost and the model accuracy in tensorboard
            tf.summary.scalar('cost', cost)
            tf.summary.scalar('accuracy', accuracy)

        return cost, accuracy, test_cost, test_accuracy

    def _hyperparameters_to_tensor(self):
        """
            This method is used to convert each parameter into a tensor value.
            This allows retrieving parameters previously used when a checkpoint is restored.
            Each tensor is saved into the graph which is saved into each checkpoint.
        """
        with tf.name_scope("hyperparameters"):
            tf.constant(
                self.hyperparameters["epoch"], dtype=tf.int32, name="epoch")
            tf.constant(
                self.hyperparameters["batch_size"], dtype=tf.int32, name="batch_size")
            tf.constant(
                self.hyperparameters["sequence_size"], dtype=tf.int32, name="sequence_size")
            tf.constant(
                self.hyperparameters["hidden_layer_size"], dtype=tf.int32, name="hidden_layer_size")
            tf.constant(
                self.hyperparameters["learning_rate"], dtype=tf.float32, name="learning_rate")
            tf.constant(
                self.hyperparameters["input_dropout"], dtype=tf.float32, name="input_dropout")
            tf.constant(
                self.hyperparameters["output_dropout"], dtype=tf.float32, name="output_dropout")
            tf.constant(
                self.hyperparameters["deep_size"], dtype=tf.int32, name="deep_size")
            tf.constant(
                self.hyperparameters["output_size"], dtype=tf.int32, name="output_size")

    def _checkpoint_to_hyperparameters(self, checkpoint):
        """
            This method is used to retrieve tensors (related to hyperparameters) previously saved.
            **input : **
                *checkpoint (String|None) Path to the checkpoint file to restore
        """
        parameters_to_retrieve = [
            "epoch", "batch_size", "sequence_size", "hidden_layer_size", "learning_rate",
            "input_dropout", "output_dropout", "deep_size", "output_size"
        ]
        scope_name = "hyperparameters/"

        def __retrive_parameters(parameters_to_retrieve, sess):
            """
                Retrive parameters from the graph
                **input : **
                    *parameters_to_retrieve (List) list of parameters
                    *sess (Tensorflow Session)
            """
            for param in parameters_to_retrieve:
                # Select only the tensor name
                graph_tensor_name = "%s%s" % (scope_name, param)
                log.info(
                    "Retrieving : %s from the checkpoint graph" % graph_tensor_name)
                try:
                    self.hyperparameters[param] = sess.run(
                        tf.get_default_graph().get_operation_by_name(
                            graph_tensor_name).outputs)[0]
                    log.info("%s = %s" % (param, self.hyperparameters[param]))
                except Exception as e:
                    log.warning("Warning : %s is not found.e = %s" % (graph_tensor_name, e))

        # Using CPU since we only want to retrieve tensors from the checkpoint
        with tf.device("/cpu:0"):
            restore_graph = tf.Graph()
            # We need to wrap this checkpoint into a sub graph to prevent conflict with
            # the main graph
            with restore_graph.as_default() as _:
                with tf.Session() as sess:
                    # Restore variables from disk.
                    loader = tf.train.import_meta_graph(checkpoint + '.meta')
                    loader.restore(sess, checkpoint)
                    __retrive_parameters(parameters_to_retrieve, sess)


    def init_model(self, checkpoint=None):
        """
            Create the tensorflow model
            **input : **
                *checkpoint (String|None) Path to the checkpoint file to restore
        """

        log.info("Init model ...")

        if checkpoint is None:
            self._hyperparameters_to_tensor()
        else:
            self._checkpoint_to_hyperparameters(checkpoint)

        if self.seed is not None:
            log.info("Tensorflow set random seed: %s" % self.seed)
            tf.set_random_seed(self.seed)

        # Inputs
        (self.tf_x, self.tf_y, self.tf_py,
         self.output_keep_prob, self.input_keep_prob) = self.build_inputs()
        # LSTM
        lstm_output, self.final_state, self.initial_state = self.build_lstm(
            tf_x=self.tf_x,
            output_keep_prob=self.output_keep_prob,
            input_keep_prob=self.input_keep_prob)
        # Outputs
        linear_output, self.softmax = self.build_outputs(lstm_output=lstm_output)
        # Errors
        self.cost, self.accuracy, self.test_cost, self.test_accuracy = Model.build_errors(
            tf_y=self.tf_y,
            tf_py=self.tf_py,
            linear_py=linear_output)
        # Optimization
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hyperparameters["learning_rate"]).minimize(self.cost)

        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

        # Tensorboard
        self.tensorboard = tf.summary.merge_all()
        train_log_name = os.path.join("tensorboard", self.mdeol_name, self.sub_train_log_name)
        test_log_name = os.path.join("tensorboard", self.model_name, self.sub_test_log_name)
        self.train_writer = tf.summary.FileWriter(train_log_name, self.session.graph)
        self.test_writer = tf.summary.FileWriter(test_log_name)
        self.train_writer_it = 0
        self.test_writer_it = 0

        if checkpoint is not None:
            log.info("Restore a previous model ...")
            self.saver.restore(self.session, checkpoint)
            log.info("Previous model Successfully loaded")

    def reset_model(self):
        """
            Reset all model variables
        """
        log.info("Restore model graph...")
        tf.reset_default_graph()
        self.session.close()
        self.session = None
        log.info("Graph successfully reset.")

    def predict(self, x):
        """
            Predict output from x values
            **input: **
                *x (Numpy array) of shape ((INPUT_SIZE,), (INPUT_SIZE,) ... (INPUT_SIZE,))
            **return (Numpy Array 2dim) **
        """
        return (self.session.run(self.softmax, feed_dict={
            self.tf_x: x,
            self.input_keep_prob: 1.,
            self.output_keep_prob: 1.}))

    def save_model(self):
        """
            Save the model
        """
        log.info("Saving model ...")

        if not os.path.exists(self.CHECKPOINTS_FOLDER):
            os.makedirs(self.CHECKPOINTS_FOLDER)

        save_path = os.path.join(self.CHECKPOINTS_FOLDER, "%s.ckpt" % self.model_name)
        save_path = self.saver.save(self.session, save_path)

        log.info("Model successfully saved here: %s" % save_path)

    def get_acc_loss_from_input(self, x_all, y_all):
        """
            Get Accuracy&Loss over all testing data
            This method compute error metrics from the user inputs and not from the model output.
            **input: **
                *x_all (Numpy Array)
                *y_all (Numpy Array 2-dim)
            **return : (Tuple (Float, Float) The accuracy and the loss of inputs **
        """
        py_list = []

        iteration = 0
        for x in x_all:
            py = self.session.run(self.softmax, feed_dict={
                self.tf_x: [x],
                self.input_keep_prob: 1.,
                self.output_keep_prob: 1.})[0]
            py_list.append(py)
            U.progress(i, len(x_all))  # Print current progression on the screen
            iteration = iteration + 1

        # Compute the loss over all predictions
        loss = self.session.run(self.test_cost, feed_dict={
            self.tf_py: py_list,
            self.tf_y: y_all,
            self.input_keep_prob: 1.,
            self.output_keep_prob: 1.})
        acc = self.session.run(self.test_accuracy, feed_dict={
            self.tf_py: py_list,
            self.tf_y: y_all,
            self.input_keep_prob: 1.,
            self.output_keep_prob: 1.})

        return acc, loss

    def get_acc_loss_from_model(self, x, y, training=False):
        """
            Get Acc&Loss.
            This method compute error metrics from the model output and not from the user input.
            **inpyt: **
                *x (Numpy array) of shape ((INPUT_SIZE,), (INPUT_SIZE,) ... (INPUT_SIZE,))
                *y (Numpy array) of shape ((OUTPUT_SIZE,), (OUTPUT_SIZE,) ... ).
                *training (Boolean) Used to indicate whether x and y are part of the training set.
                False by default
            **return : (Tuple (Float, Float) The accuracy and the loss of the batch **
        """
        # List of tensors to run
        tensors = [self.accuracy, self.cost, self.tensorboard]
        acc, loss, summary = self.session.run(tensors, feed_dict={
            self.tf_x: x,
            self.tf_y: y,
            self.input_keep_prob: 1.,
            self.output_keep_prob: 1.})

        # If this test is for the training set
        if not training:
            self.test_writer.add_summary(summary, self.test_writer_it)
            self.test_writer_it += 1
        else:
            # Otherwise, this is for the testing set
            self.train_writer.add_summary(summary, self.train_writer_it)
            self.train_writer_it += 1

        return acc, loss

    def get_acc_loss_from_result(self, all_py, all_y):
        """
            Get acc and loss from model result previously computed.
            **inpyt: **
                *x (Numpy array) of shape ((INPUT_SIZE,), (INPUT_SIZE,) ... (INPUT_SIZE,))
                *y (Numpy array) of shape ((OUTPUT_SIZE,), (OUTPUT_SIZE,) ... ).
                *training (Boolean) Used to indicate whether x and y are part of the training set.
                False by default
            **return : (Tuple (Float, Float) The accuracy and the loss of the result **
        """
        loss = self.session.run(self.test_cost, feed_dict={
            self.tf_py: all_py,
            self.tf_y: all_y,
            self.input_keep_prob: 1.,
            self.output_keep_prob: 1.})

        acc = self.session.run(self.test_accuracy, feed_dict={
            self.tf_py: all_py,
            self.tf_y: all_y,
            self.input_keep_prob: 1.,
            self.output_keep_prob: 1.})

        return acc, loss

    def optimize_model(self, x, y):
        """
            **input: **
                *x (Numpy array) of shape ((INPUT_SIZE,), (INPUT_SIZE,) ... (INPUT_SIZE,))
                *y (Numpy array) of shape ((OUTPUT_SIZE,), (OUTPUT_SIZE,) ... )
            **return (Tuple (Float, Float))
                Accuracy and Loss for this batch
        """
        initial_state = self.get_initial_state()

        # Tensors to run
        tensors = [self.optimizer, self.accuracy, self.cost, self.tensorboard]
        _, acc, cost, summary = self.session.run(tensors, feed_dict={
            self.tf_x: x,
            self.tf_y: y,
            self.input_keep_prob: self.hyperparameters["input_dropout"],
            self.output_keep_prob: self.hyperparameters["output_dropout"],
            self.initial_state: initial_state
        })

        # Log all scalars values in Tensorboard
        self.train_writer.add_summary(summary, self.train_writer_it)
        self.train_writer_it += 1

        return acc, cost

    def get_initial_state(self):
        """
            ** Return (Tensor result) **
                Initial state of all cells
        """
        return self.session.run(self.initial_state)
