"""杂项文件
"""
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
import collections
import os
import multiprocessing


# set random seed
# make session
# initialize variable
#-----------------------------------------------------------------#
def get_session(config = None,graph = None):
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess

def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0,
            log_device_placement=False)
        config.gpu_options.allow_growth = True
#         config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
#         config.gpu_options.per_process_gpu_memory_fraction = 0.9

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)

def set_global_seed(seed):
    tf.set_random_seed(seed)

ALREADY_INITIALIZED = set()

def initialize():
    """Initialize all the uninitialized variables in the global scope."""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def fc(x, scope, nh, init_scale=1, init_bias=0):
    """
    full connection，return x*W+b
    Parameters
    ----------
    x: tf.placeholder
    scope：str
    nh：int
    init_scale:int or float 
    init_bias：int or float 
    """
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin,nh], initializer=tf.truncated_normal_initializer(stddev=init_scale)) #有修改
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x,w) + b


def huber_loss(x, delta=1.0):
    """
    huber_loss
    """
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
        )

def encode_observation(ob_space, placeholder):
    if isinstance(ob_space, Discrete):
        return tf.to_float(tf.one_hot(placeholder,ob_space.n))
    elif isinstance(ob_space, Box):
        return tf.to_float(placeholder)
    else:
        raise NotImplementedError

def observation_placehoder(ob_space, batch_size=None, name="ob"):
    assert isinstance(ob_space,Discrete) or isinstance(ob_space,Box), "can only deal with Discrete or Box"
    
    return tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=ob_space.dtype, name=name) 

def observation_input(ob_space, batch_size=None, name="ob"):
    placeholder = observation_placehoder(ob_space, batch_size, name)
    return placeholder, encode_observation(ob_space, placeholder)

def adjust_shape(placeholder, data):
    if not isinstance(data, np.ndarray) or not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)
    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]

    return np.reshape(data, placeholder_shape)

class ObservationInput(object):
    def __init__(self, observation_space, name=None):
        input, self.processed_input = observation_input(observation_space, name=name)
        self._placeholder = input
        self.name = input.name

    def get(self):
        return self.processed_input

    def make_feed_dict(self, data):
        return {self._placeholder: adjust_shape(self._placeholder, data)}

def function(inputs, outputs, updates=None, givens=None):
    """
    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})
        with tf.Session() as sess:
            initialize()
            print(lin(2)) #output6
            print(lin(x=3)) #output9
            print(lin(2, 2)) #output10
            print(lin(x=2, y=3)) #output12
    ----------
    inputs: [tf.placeholder, tf.constant, or object with make_feed_dict method]
        "make_feed_dict"
    outputs: [tf.Variable] or tf.Variable
    updates: [tf.Operation] or tf.Operation
    gives：int,bool or float
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]

class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (type(inpt) is tf.Tensor and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def _feed_input(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = adjust_shape(inpt, value)

    def __call__(self, *args, **kwargs):
        assert len(args) + len(kwargs) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        for inpt in self.givens:
            feed_dict[inpt] = adjust_shape(inpt, feed_dict.get(inpt, self.givens[inpt]))
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        return results



class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """
        schedule_timesteps
        initial_p -> final_p
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)