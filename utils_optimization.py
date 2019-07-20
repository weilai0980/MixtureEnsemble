
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
    inject_noise = rnd * math_ops.sqrt(lr * (1 - beta1_t)/(v_sqrt+epsilon_t))
    
    
    # -----
    
    var_update = state_ops.assign_sub(var, lr * m_t/(v_sqrt+epsilon_t) - inject_noise, use_locking = self._use_locking)
    
    
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
    
    '''
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1.0 - beta1_t)
    m_t = state_ops.assign(m, m_scaled_g_values + m * beta1_t, use_locking = self._use_locking)
    '''
    
    #with ops.control_dependencies([m_t]):
    #  m_t = scatter_add(m, indices, m_scaled_g_values)
    
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1.0 - beta2_t)
    v_t = state_ops.assign(v, v_scaled_g_values + v * beta2_t, use_locking = self._use_locking)
    
    #with ops.control_dependencies([v_t]):
    #  v_t = scatter_add(v, indices, v_scaled_g_values)
    
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
    
    inject_noise = rnd * math_ops.sqrt(lr / (v_sqrt + epsilon_t))
    
    var_update = state_ops.assign_sub(var, lr * grad / (v_sqrt + epsilon_t) - inject_noise, use_locking = self._use_locking)
    
    # ----- 
    
    
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


