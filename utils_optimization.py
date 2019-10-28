
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops

from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_random_ops import *
import tensorflow as tf

'''
https://www.tensorflow.org/api_docs/python/tf/disable_resource_variables
'''

class myAdamOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Adam algorithm.
  See [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
  ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
  """

  def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               use_locking=False,
               name="Adam"):
    r"""Construct a new Adam optimizer.
    Initialization:
    $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
    $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
    $$t := 0 \text{(Initialize timestep)}$$
    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section 2 of the paper:
    $$t := t + 1$$
    $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
    $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
    $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
    $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.
    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor. The exponential decay
        rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".  @compatibility(eager) When eager execution is
        enabled, `learning_rate`, `beta1`, `beta2`, and `epsilon` can each be a
        callable that takes no arguments and returns the actual value to use.
        This can be useful for changing these values across different
        invocations of optimizer functions. @end_compatibility
    """
    super(myAdamOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(
        initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    return training_ops.apply_adam(
        var,
        m,
        v,
        math_ops.cast(beta1_power, var.dtype.base_dtype),
        math_ops.cast(beta2_power, var.dtype.base_dtype),
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    return training_ops.resource_apply_adam(
        var.handle,
        m.handle,
        v.handle,
        math_ops.cast(beta1_power, grad.dtype.base_dtype),
        math_ops.cast(beta2_power, grad.dtype.base_dtype),
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)
    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(
        var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values,
        var,
        grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x,
            i,
            v,
            use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(grad, var, indices,
                                     self._resource_scatter_add)

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      beta1_power, beta2_power = self._get_beta_accumulators()
      with ops.colocate_with(beta1_power):
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta1_t, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta2_t, use_locking=self._use_locking)
    return control_flow_ops.group(
        *update_ops + [update_beta1, update_beta2], name=name_scope)

class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """
  A basic Adam optimizer that includes "correct" L2 weight decay.
  
  Ref.:
    [2018 ICLR] fixing weight decay regularization in Adam, https://openreview.net/pdf?id=rk6qdGgCZ
    
    https://github.com/google-research/bert/blob/master/optimization.py
  """
  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


def _ShapeTensor(shape):
  
  """Convert to an int32 or int64 tensor, defaulting to int32 if empty."""
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int32
  else:
    dtype = None
    
  return ops.convert_to_tensor(shape, dtype = dtype, name="shape")

class sg_mcmc_adam(optimizer.Optimizer):
    
  """
  
  """

  def __init__(self,
               learning_rate = 0.001,
               beta1 = 0.9,
               beta2 = 0.999,
               epsilon = 1e-8,
               use_locking = False,
               name = "sg_mcmc_adam"):
        
    r"""Construct a new Adam based optimizer.
    
    Initialization:
    $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
    $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
    $$t := 0 \text{(Initialize timestep)}$$
    
    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section 2 of the paper:
    
    $$t := t + 1$$
    $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
    $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
    $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
    $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
    
    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.
    
    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    
    Args:
      
      learning_rate: A Tensor or a floating point value.  The learning rate.
      
      beta1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      
      beta2: A float value or a constant float tensor. The exponential decay
        rate for the 2nd moment estimates.
      
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      
      use_locking: If True use locks for update operations.
      
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".  @compatibility(eager) When eager execution is
        enabled, `learning_rate`, `beta1`, `beta2`, and `epsilon` can each be a
        callable that takes no arguments and returns the actual value to use.
        This can be useful for changing these values across different
        invocations of optimizer functions. @end_compatibility
    """
    
    super(sg_mcmc_adam, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None
    

  def _get_beta_accumulators(self):
    
    with ops.init_scope():
        
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
        
      return (self._get_non_slot_variable("beta1_power", graph = graph),
              self._get_non_slot_variable("beta2_power", graph = graph))

  def _create_slots(self, 
                    var_list):
        
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key = lambda x: x.name)
    
    self._create_non_slot_variable(
        initial_value = self._beta1, name = "beta1_power", colocate_with = first_var)
    self._create_non_slot_variable(
        initial_value = self._beta2, name = "beta2_power", colocate_with = first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)
        
  def _prepare(self):
    
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name = "learning_rate")
    self._beta1_t = ops.convert_to_tensor(beta1, name = "beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name = "beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name = "epsilon")

  def _apply_dense(self, 
                   grad, 
                   var):
    
    # -----
     
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1.0 - beta2_power) / (1.0 - beta1_power))
    
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1.0 - beta1_t)
    m_t = state_ops.assign(m, m_scaled_g_values + m * beta1_t, use_locking = self._use_locking)
    
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1.0 - beta2_t)
    v_t = state_ops.assign(v, v_scaled_g_values + v * beta2_t, use_locking = self._use_locking)
    
    v_sqrt = math_ops.sqrt(v_t)
    #var_update = state_ops.assign_sub(var, 1.0*lr * m_t / (v_sqrt + epsilon_t), use_locking = self._use_locking)
    
    # ----- inject noise
    
    shape_tensor = _ShapeTensor(array_ops.shape(var))
    
    dtype = dtypes.float32
    
    seed = None
    seed1, seed2 = random_seed.get_seed(seed)
    
    rnd = gen_random_ops.random_standard_normal(shape_tensor, 
                                                dtype,
                                                seed = seed1, 
                                                seed2 = seed2)
    
    #inject_noise = rnd * math_ops.sqrt(2.0 * lr * (1.0-beta1_t)/(v_sqrt+epsilon_t))
    #inject_noise = rnd * math_ops.sqrt(1.0 * lr/(v_sqrt + epsilon_t))
    
    """ ? 1.0*lr leads to inferior performance ? """
    inject_noise = rnd * math_ops.sqrt(lr * (1.0 - beta1_t) * (1.0 - beta1_t)/(v_sqrt + epsilon_t))
    
    ''' beta1 on noise '''
    
    # -----
    """ ? 1.0 """
    var_update = state_ops.assign_sub(var, lr * m_t/(v_sqrt + epsilon_t) - inject_noise, use_locking = self._use_locking)
    
    return control_flow_ops.group(*[var_update, m_t, v_t]) 
    
    '''
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    
    return training_ops.apply_adam(
        var,
        m,
        v,
        math_ops.cast(beta1_power, var.dtype.base_dtype),
        math_ops.cast(beta2_power, var.dtype.base_dtype),
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op
    
    '''
    
  def _resource_apply_dense(self, 
                            grad, 
                            var):
    # -----
     
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1.0 - beta2_power) / (1.0 - beta1_power))
    
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1.0 - beta1_t)
    m_t = state_ops.assign(m, m_scaled_g_values + m * beta1_t, use_locking = self._use_locking)
    
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1.0 - beta2_t)
    v_t = state_ops.assign(v, v_scaled_g_values + v * beta2_t, use_locking = self._use_locking)
    
    v_sqrt = math_ops.sqrt(v_t)
    #var_update = state_ops.assign_sub(var, 1.0*lr * m_t / (v_sqrt + epsilon_t), use_locking = self._use_locking)
    
    # ----- inject noise
    
    shape_tensor = _ShapeTensor(array_ops.shape(var))
    
    dtype = dtypes.float32
    
    seed = None
    seed1, seed2 = random_seed.get_seed(seed)
    
    rnd = gen_random_ops.random_standard_normal(shape_tensor, 
                                                dtype,
                                                seed = seed1, 
                                                seed2 = seed2)
    
    #inject_noise = rnd * math_ops.sqrt(2.0 * lr * (1.0-beta1_t)/(v_sqrt+epsilon_t))
    #inject_noise = rnd * math_ops.sqrt(1.0 * lr/(v_sqrt + epsilon_t))
    
    """ ? 1.0*lr leads to inferior performance ? """
    inject_noise = rnd * math_ops.sqrt(lr * (1.0 - beta1_t) * (1.0 - beta1_t)/(v_sqrt + epsilon_t))
    
    ''' beta1 on noise '''
    
    # -----
    """ ? 1.0 """
    var_update = state_ops.assign_sub(var, lr * m_t/(v_sqrt + epsilon_t) - inject_noise, use_locking = self._use_locking)
    
    return control_flow_ops.group(*[var_update, m_t, v_t]) 

  def _finish(self, 
              update_ops, 
              name_scope):
    
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
        
        beta1_power, beta2_power = self._get_beta_accumulators()
      
        with ops.colocate_with(beta1_power):
            
            update_beta1 = beta1_power.assign(
                beta1_power * self._beta1_t, use_locking=self._use_locking)
            
            update_beta2 = beta2_power.assign(
                beta2_power * self._beta2_t, use_locking=self._use_locking)
    
    return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name = name_scope)

class sg_mcmc_RMSprop(optimizer.Optimizer):
    
  def __init__(self,
               learning_rate = 0.001,
               beta1 = 0.9,
               beta2 = 0.999,
               epsilon = 1e-8,
               use_locking = False,
               name = "Adam"):
        
    r"""Construct a new RMSprop optimizer.
    
    Initialization:
    $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
    $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
    $$t := 0 \text{(Initialize timestep)}$$
    
    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section 2 of the paper:
    
    $$t := t + 1$$
    $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
    $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
    $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
    $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
    
    Args:
      
      learning_rate: A Tensor or a floating point value.  The learning rate.
      
      beta1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      
      beta2: A float value or a constant float tensor. The exponential decay
        rate for the 2nd moment estimates.
      
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      
      use_locking: If True use locks for update operations.
      
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".  @compatibility(eager) When eager execution is
        enabled, `learning_rate`, `beta1`, `beta2`, and `epsilon` can each be a
        callable that takes no arguments and returns the actual value to use.
        This can be useful for changing these values across different
        invocations of optimizer functions. @end_compatibility
    """
    
    super(sg_mcmc_RMSprop, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

  def _get_beta_accumulators(self):
    
    with ops.init_scope():
        
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
        
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, 
                    var_list):
        
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key = lambda x: x.name)
    
    self._create_non_slot_variable(
        initial_value = self._beta1, name = "beta1_power", colocate_with = first_var)
    self._create_non_slot_variable(
        initial_value = self._beta2, name = "beta2_power", colocate_with = first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      #self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name = "learning_rate")
    self._beta1_t = ops.convert_to_tensor(beta1, name = "beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name = "beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name = "epsilon")

  def _apply_dense(self, 
                   grad, 
                   var):
    
    # -----
     
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1.0 - beta2_t)
    v_t = state_ops.assign(v, v_scaled_g_values + v * beta2_t, use_locking = self._use_locking)
    
    v_sqrt = math_ops.sqrt(v_t)
    
    #var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking = self._use_locking)
    
    # ----- inject noise
    
    shape_tensor = _ShapeTensor(array_ops.shape(var))
    
    dtype = dtypes.float32
    
    seed = None
    seed1, seed2 = random_seed.get_seed(seed)
    
    rnd = gen_random_ops.random_standard_normal(shape_tensor, 
                                                dtype,
                                                seed = seed1, 
                                                seed2 = seed2)
    
    inject_noise = rnd * math_ops.sqrt(1.0 * lr / (v_sqrt + epsilon_t))
    
    
    # ----- 
    
    var_update = state_ops.assign_sub(var, lr * grad / (v_sqrt + epsilon_t) - inject_noise, use_locking = self._use_locking)
    
    return control_flow_ops.group(*[var_update, v_t]) 

  def _resource_apply_dense(self,
                            grad, 
                            var):
    
    # -----
     
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1.0 - beta2_t)
    v_t = state_ops.assign(v, v_scaled_g_values + v * beta2_t, use_locking = self._use_locking)
    
    v_sqrt = math_ops.sqrt(v_t)
    
    #var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking = self._use_locking)
    
    # ----- inject noise
    
    shape_tensor = _ShapeTensor(array_ops.shape(var))
    
    dtype = dtypes.float32
    
    seed = None
    seed1, seed2 = random_seed.get_seed(seed)
    
    rnd = gen_random_ops.random_standard_normal(shape_tensor, 
                                                dtype,
                                                seed = seed1, 
                                                seed2 = seed2)
    
    inject_noise = rnd * math_ops.sqrt(1.0 * lr / (v_sqrt + epsilon_t))
    
    
    # ----- 
    
    var_update = state_ops.assign_sub(var, lr * grad / (v_sqrt + epsilon_t) - inject_noise, use_locking = self._use_locking)
    
    return control_flow_ops.group(*[var_update, v_t]) 
    

  def _finish(self, 
              update_ops, 
              name_scope):
    
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
        
        beta1_power, beta2_power = self._get_beta_accumulators()
      
        with ops.colocate_with(beta1_power):
            update_beta1 = beta1_power.assign(
                beta1_power * self._beta1_t, use_locking=self._use_locking)
            
            update_beta2 = beta2_power.assign(
                beta2_power * self._beta2_t, use_locking=self._use_locking)
    
    return control_flow_ops.group(
        *update_ops + [update_beta1, update_beta2], name=name_scope)

class myRMSprop(optimizer.Optimizer):
    
  def __init__(self,
               learning_rate = 0.001,
               beta1 = 0.9,
               beta2 = 0.999,
               epsilon = 1e-8,
               use_locking = False,
               name = "Adam"):
        
    r"""Construct a new RMSprop optimizer.
    
    Initialization:
    $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
    $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
    $$t := 0 \text{(Initialize timestep)}$$
    
    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section 2 of the paper:
    
    $$t := t + 1$$
    $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
    $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
    $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
    $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
    
    Args:
      
      learning_rate: A Tensor or a floating point value.  The learning rate.
      
      beta1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      
      beta2: A float value or a constant float tensor. The exponential decay
        rate for the 2nd moment estimates.
      
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      
      use_locking: If True use locks for update operations.
      
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".  @compatibility(eager) When eager execution is
        enabled, `learning_rate`, `beta1`, `beta2`, and `epsilon` can each be a
        callable that takes no arguments and returns the actual value to use.
        This can be useful for changing these values across different
        invocations of optimizer functions. @end_compatibility
    """
    
    super(myRMSprop, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

  def _get_beta_accumulators(self):
    
    with ops.init_scope():
        
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
        
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, 
                    var_list):
        
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key = lambda x: x.name)
    
    self._create_non_slot_variable(
        initial_value = self._beta1, name = "beta1_power", colocate_with = first_var)
    self._create_non_slot_variable(
        initial_value = self._beta2, name = "beta2_power", colocate_with = first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      #self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name = "learning_rate")
    self._beta1_t = ops.convert_to_tensor(beta1, name = "beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name = "beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name = "epsilon")

  def _apply_dense(self, 
                   grad, 
                   var):
    
    # -----
     
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    
    
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1.0 - beta2_t)
    v_t = state_ops.assign(v, v_scaled_g_values + v * beta2_t, use_locking = self._use_locking)
    
    v_sqrt = math_ops.sqrt(v_t)
    
    var_update = state_ops.assign_sub(var, lr * grad / (v_sqrt + epsilon_t), use_locking = self._use_locking)
    
    
    return control_flow_ops.group(*[var_update, v_t]) 

  def _finish(self, 
              update_ops, 
              name_scope):
    
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
        
        beta1_power, beta2_power = self._get_beta_accumulators()
      
        with ops.colocate_with(beta1_power):
            update_beta1 = beta1_power.assign(
                beta1_power * self._beta1_t, use_locking=self._use_locking)
            
            update_beta2 = beta2_power.assign(
                beta2_power * self._beta2_t, use_locking=self._use_locking)
    
    return control_flow_ops.group(
        *update_ops + [update_beta1, update_beta2], name=name_scope)

#import tensorflow.compat.v1 as tf1
#import tensorflow.compat.v2 as tf

#from tensorflow_probability.python.internal import distribution_util
#from tensorflow_probability.python.internal import dtype_util
#from tensorflow_probability.python.math import diag_jacobian

from tensorflow.python.training import training_ops

class StochasticGradientLangevinDynamics(optimizer.Optimizer):
    
  """
  An optimizer module for stochastic gradient Langevin dynamics.
  This implements the preconditioned Stochastic Gradient Langevin Dynamics
  optimizer [(Li et al., 2016)][1]. The optimization variable is regarded as a
  sample from the posterior under Stochastic Gradient Langevin Dynamics with
  noise rescaled in each dimension according to [RMSProp](
  http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
  Note: If a prior is included in the loss, it should be scaled by
  `1/data_size`, where `data_size` is the number of points in the data set.
  I.e., it should be divided by the `data_size` term described below.
  
  Args:
    learning_rate: Scalar `float`-like `Tensor`. The base learning rate for the
      optimizer. Must be tuned to the specific function being minimized.
    preconditioner_decay_rate: Scalar `float`-like `Tensor`. The exponential
      decay rate of the rescaling of the preconditioner (RMSprop). (This is
      "alpha" in Li et al. (2016)). Should be smaller than but nearly `1` to
      approximate sampling from the posterior. (Default: `0.95`)
    data_size: Scalar `int`-like `Tensor`. The effective number of
      points in the data set. Assumes that the loss is taken as the mean over a
      minibatch. Otherwise if the sum was taken, divide this number by the
      batch size. If a prior is included in the loss function, it should be
      normalized by `data_size`. Default value: `1`.
    burnin: Scalar `int`-like `Tensor`. The number of iterations to collect
      gradient statistics to update the preconditioner before starting to draw
      noisy samples. (Default: `25`)
    diagonal_bias: Scalar `float`-like `Tensor`. Term added to the diagonal of
      the preconditioner to prevent the preconditioner from degenerating.
      (Default: `1e-8`)
    name: Python `str` describing ops managed by this function.
      (Default: `"StochasticGradientLangevinDynamics"`)
    parallel_iterations: the number of coordinates for which the gradients of
        the preconditioning matrix can be computed in parallel. Must be a
        positive integer.
  Raises:
    InvalidArgumentError: If preconditioner_decay_rate is a `Tensor` not in
      `(0,1]`.
    NotImplementedError: If eager execution is enabled.
    
  #### References
  [1]: Chunyuan Li, Changyou Chen, David Carlson, and Lawrence Carin.
       Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural
       Networks. In _Association for the Advancement of Artificial
       Intelligence_, 2016. https://arxiv.org/abs/1512.07666
  """

  def __init__(self,
               learning_rate,
               preconditioner_decay_rate = 0.95,
               data_size = 1,
               burnin = 25,
               diagonal_bias = 1e-8,
               name = None,
               parallel_iterations = 10):
        
    default_name = 'StochasticGradientLangevinDynamics'
    with tf.name_scope(name, default_name, [
        learning_rate, preconditioner_decay_rate, data_size, burnin,
        diagonal_bias
    ]):
      if tf.executing_eagerly():
        raise NotImplementedError('Eager execution currently not supported for '
                                  ' SGLD optimizer.')

      self._preconditioner_decay_rate = tf.convert_to_tensor(
          value=preconditioner_decay_rate, name='preconditioner_decay_rate')
      self._data_size = tf.convert_to_tensor(value=data_size, name='data_size')
      
      self._burnin = tf.convert_to_tensor(value = burnin,
                                          name = 'burnin',
                                          dtype = tf.int64)
          #dtype = dtype_util.common_dtype([burnin], dtype_hint=tf.int64))
      
      self._diagonal_bias = tf.convert_to_tensor(value = diagonal_bias, 
                                                 name = 'diagonal_bias')
      # TODO(b/124800185): Consider migrating `learning_rate` to be a
      # hyperparameter handled by the base Optimizer class. This would allow
      # users to plug in a `tf.keras.optimizers.schedules.LearningRateSchedule`
      # object in addition to Tensors.
      self._learning_rate = tf.convert_to_tensor(
          value=learning_rate, name='learning_rate')
      self._parallel_iterations = parallel_iterations
      
      self._preconditioner_decay_rate = self._preconditioner_decay_rate
      self._data_size = self._data_size
      self._burnin = self._burnin
      self._diagonal_bias = self._diagonal_bias
        
      '''
      self._preconditioner_decay_rate = distribution_util.with_dependencies([
          tf1.assert_non_negative(
              self._preconditioner_decay_rate,
              message='`preconditioner_decay_rate` must be non-negative'),
          tf1.assert_less_equal(
              self._preconditioner_decay_rate,
              1.,
              message='`preconditioner_decay_rate` must be at most 1.'),
      ], self._preconditioner_decay_rate)
      
      self._data_size = distribution_util.with_dependencies([
          tf1.assert_greater(
              self._data_size,
              0,
              message='`data_size` must be greater than zero')
      ], self._data_size)

      self._burnin = distribution_util.with_dependencies([
          tf1.assert_non_negative(
              self._burnin, message='`burnin` must be non-negative'),
          tf1.assert_integer(
              self._burnin, message='`burnin` must be an integer')
      ], self._burnin)

      self._diagonal_bias = distribution_util.with_dependencies([
          tf1.assert_non_negative(
              self._diagonal_bias,
              message='`diagonal_bias` must be non-negative')
      ], self._diagonal_bias)
      
      '''

      super(StochasticGradientLangevinDynamics,
            self).__init__(name=name or default_name)

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'rms', 'ones')

  def get_config(self):
    # TODO(b/124800185): Consider making `learning_rate`, `data_size`, `burnin`,
    # `preconditioner_decay_rate` and `diagonal_bias` hyperparameters.
    pass

  def _prepare(self, var_list):
    # We need to put the conversion and check here because a user will likely
    # want to decay the learning rate dynamically.
    
    self._learning_rate_tensor = tf.convert_to_tensor(value=self._learning_rate, name='learning_rate_tensor')
    '''
    self._learning_rate_tensor = distribution_util.with_dependencies(
        [
            tf1.assert_non_negative(
                self._learning_rate,
                message='`learning_rate` must be non-negative')
        ],
        tf.convert_to_tensor(
            value=self._learning_rate, name='learning_rate_tensor'))
    '''        
    
    self._decay_tensor = tf.convert_to_tensor(
        value=self._preconditioner_decay_rate, name='preconditioner_decay_rate')

    super(StochasticGradientLangevinDynamics, self)._prepare(var_list)

  def _resource_apply_dense(self, grad, var):
    rms = self.get_slot(var, 'rms')
    new_grad = self._apply_noisy_update(rms, grad, var)
    return training_ops.resource_apply_gradient_descent(
        var.handle,
        tf.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        new_grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    rms = self.get_slot(var, 'rms')
    new_grad = self._apply_noisy_update(rms, grad, var, indices)
    return self._resource_scatter_add(
        var, indices,
        -new_grad * tf.cast(self._learning_rate_tensor, var.dtype.base_dtype))

  @property
  def variable_scope(self):
    """Variable scope of all calls to `tf.get_variable`."""
    return self._variable_scope

  def _apply_noisy_update(self, mom, grad, var, indices=None):
    # Compute and apply the gradient update following
    # preconditioned Langevin dynamics
    stddev = tf.where(
        tf.squeeze(self.iterations > tf.cast(self._burnin, tf.int64)),
        tf.cast(tf.math.rsqrt(self._learning_rate), grad.dtype),
        tf.zeros([], grad.dtype))
    # Keep an exponentially weighted moving average of squared gradients.
    # Not thread safe
    decay_tensor = tf.cast(self._decay_tensor, grad.dtype)
    new_mom = decay_tensor * mom + (1. - decay_tensor) * tf.square(grad)
    preconditioner = tf.math.rsqrt(new_mom +
                                   tf.cast(self._diagonal_bias, grad.dtype))

    # Compute gradients of the preconditioner.
    # Note: Since the preconditioner depends indirectly on `var` through `grad`,
    # in Eager mode, `diag_jacobian` would need access to the loss function.
    # This is the only blocker to supporting Eager mode for the SGLD optimizer.
    
    
    #_, preconditioner_grads = diag_jacobian(
    #     xs=var,
    #    ys=preconditioner,
    #    parallel_iterations=self._parallel_iterations)

    mean = 0.5 * (preconditioner * grad * tf.cast(self._data_size, grad.dtype) )
                 # - preconditioner_grads[0])
    
    stddev *= tf.sqrt(preconditioner)
    result_shape = tf.broadcast_dynamic_shape(
        tf.shape(input=mean), tf.shape(input=stddev))

    update_ops = []
    if indices is None:
      update_ops.append(mom.assign(new_mom))
    else:
      update_ops.append(self._resource_scatter_update(mom, indices, new_mom))

    with tf.control_dependencies(update_ops):
      return tf.random.normal(
          shape=result_shape, mean=mean, stddev=stddev, dtype=grad.dtype)