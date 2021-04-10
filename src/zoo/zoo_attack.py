
import random
import sys
import os
import numpy as np
import scipy.misc
from numba import jit
import math
import time
from ..utils.get_modules import (
    get_classifier,
    get_frontend,
)
from ..models.combined import Combined
from ..utils.read_datasets import(
    cifar10
)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from ..utils.namers import attack_file_namer


BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # if we stop improving, abort gradient descent early
LEARNING_RATE = 0.01    # larger values converge faster to less accurate results
TARGETED = False          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_C = 0.01      # the initial constant c to pick as a first guess


@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    # indice = torch.tensor(range(0, 3*299*299), dtype = torch.int32)
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
    # true_grads = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
    # true_grads, losses, l2s, scores, nimgs = self.sess.run([self.grad_op, self.total_loss, self.l2dist, self.output, self.newimg], feed_dict={self.modifier: self.real_modifier})
    # grad = true_grads[0].reshape(-1)[indice]
    # print(grad, true_grads[0].reshape(-1)[indice])
    # self.real_modifier.reshape(-1)[indice] -= self.LEARNING_RATE * grad
    # self.real_modifier -= self.LEARNING_RATE * true_grads[0]
    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * corr * mt / (np.sqrt(vt) + 1e-8)
    # set it back to [-0.5, +0.5] region is tanh is used
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    # print(grad)
    # print(old_val - m[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1


def coordinate_ADAM_torch(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    # indice = torch.tensor(range(0, 3*299*299), dtype = torch.int64)
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
    # true_grads = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
    # true_grads, losses, l2s, scores, nimgs = self.sess.run([self.grad_op, self.total_loss, self.l2dist, self.output, self.newimg], feed_dict={self.modifier: self.real_modifier})
    # grad = true_grads[0].reshape(-1)[indice]
    # print(grad, true_grads[0].reshape(-1)[indice])
    # self.real_modifier.reshape(-1)[indice] -= self.LEARNING_RATE * grad
    # self.real_modifier -= self.LEARNING_RATE * true_grads[0]
    # ADAM update

    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (torch.sqrt(1 - torch.pow(beta2, epoch))) / \
        (1 - torch.pow(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * corr * mt / (torch.sqrt(vt) + 1e-8)
    # set it back to [-0.5, +0.5] region is tanh is used
    if proj:
        old_val = torch.maximum(torch.minimum(
            old_val, up[indice]), down[indice])
    # print(grad)
    # print(old_val - m[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1


@jit(nopython=True)
def coordinate_Newton(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    # def sign(x):
    #     return np.piecewise(x, [x < 0, x >= 0], [-1, 1])
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
        hess[i] = (losses[i*2+1] - 2 * cur_loss +
                   losses[i*2+2]) / (0.0001 * 0.0001)
    # print("New epoch:")
    # print('grad', grad)
    # print('hess', hess)
    # hess[hess < 0] = 1.0
    # hess[torch.abs(hess) < 0.1] = sign(hess[torch.abs(hess) < 0.1]) * 0.1
    # negative hessian cannot provide second order information, just do a gradient descent
    hess[hess < 0] = 1.0
    # hessian too small, could be numerical problems
    hess[hess < 0.1] = 0.1
    # print(hess)
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * grad / hess
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    # print('delta', old_val - m[indice])
    m[indice] = old_val
    # print(m[indice])


@jit(nopython=True)
def coordinate_Newton_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
        hess[i] = (losses[i*2+1] - 2 * cur_loss +
                   losses[i*2+2]) / (0.0001 * 0.0001)
    # print("New epoch:")
    # print(grad)
    # print(hess)
    # positive hessian, using newton's method
    hess_indice = (hess >= 0)
    # print(hess_indice)
    # negative hessian, using ADAM
    adam_indice = (hess < 0)
    # print(adam_indice)
    # print(sum(hess_indice), sum(adam_indice))
    hess[hess < 0] = 1.0
    hess[hess < 0.1] = 0.1
    # hess[torch.abs(hess) < 0.1] = sign(hess[torch.abs(hess) < 0.1]) * 0.1
    # print(adam_indice)
    # Newton's Method
    m = real_modifier.reshape(-1)
    old_val = m[indice[hess_indice]]
    old_val -= lr * grad[hess_indice] / hess[hess_indice]
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(
            old_val, up[indice[hess_indice]]), down[indice[hess_indice]])
    m[indice[hess_indice]] = old_val
    # ADMM
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,
                                 epoch[adam_indice]))) / (1 - np.power(beta1, epoch[adam_indice]))
    old_val = m[indice[adam_indice]]
    old_val -= lr * corr * mt[adam_indice] / (np.sqrt(vt[adam_indice]) + 1e-8)
    # old_val -= lr * grad[adam_indice]
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(
            old_val, up[indice[adam_indice]]), down[indice[adam_indice]])
    m[indice[adam_indice]] = old_val
    adam_epoch[indice] = epoch + 1
    # print(m[indice])


class BlackBoxL2:
    def __init__(self, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS, print_every=100, early_stop_iters=0,
                 abort_early=ABORT_EARLY,
                 initial_c=INITIAL_C,
                 use_log=True, use_tanh=True, use_resize=False, adam_beta1=0.9, adam_beta2=0.999, reset_adam_after_found=False,
                 solver="adam", save_ckpts="", load_checkpoint="", start_iter=0,
                 init_size=32, use_importance=False, device="cuda"):
        """
        The L_2 optimized attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of gradient evaluations to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_c: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """

        if solver != "fake_zero":
            torch.set_grad_enabled(False)

        self.image_size, self.num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.model = model
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations // 10
        print("early stop:", self.early_stop_iters)
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_c = initial_c
        self.start_iter = start_iter
        self.batch_size = batch_size
        self.num_channels = self.num_channels
        self.resize_init_size = init_size
        self.use_importance = use_importance
        if use_resize:
            self.small_x = self.resize_init_size
            self.small_y = self.resize_init_size
        else:
            self.small_x = self.image_size
            self.small_y = self.image_size

        self.use_tanh = use_tanh
        self.use_resize = use_resize
        self.save_ckpts = save_ckpts
        if save_ckpts:
            os.system("mkdir -p {}".format(save_ckpts))

        self.repeat = binary_search_steps >= 10
        self.device = device

        # each batch has a different modifier value (see below) to evaluate
        # small_shape = (None,self.small_x,self.small_y,num_channels)

        single_shape = (self.num_channels, self.image_size, self.image_size)
        small_single_shape = (self.num_channels, self.small_x, self.small_y)

        # the variable we're going to optimize over
        # support multiple batches
        # support any size image, will be resized to model native size

        # the real variable, initialized to 0
        self.load_checkpoint = load_checkpoint
        if load_checkpoint:
            # if checkpoint is incorrect reshape will fail
            print("Using checkpint", load_checkpoint)
            self.real_modifier = torch.load(load_checkpoint).reshape(
                (1,) + small_single_shape, map_location=torch.device(device))
        else:
            self.real_modifier = torch.zeros(
                (1,) + small_single_shape, dtype=torch.float32, device=self.device)

        if solver == "fake_zero":
            self.real_modifier.requires_grad = True
        # self.real_modifier = np.random.randn(image_size * image_size * num_channels).astype(torch.float32).reshape((1,) + single_shape)
        # self.real_modifier /= np.linalg.norm(self.real_modifier)
        # these are variables to be more efficient in sending data to tf
        # we only work on 1 image at once; the batch is for evaluation loss at different modifiers
        self.true_img = torch.zeros(single_shape, device=self.device)
        self.true_label_1hot = torch.zeros(num_labels, device=self.device)
        self.c = 0.0

        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = torch.tensor(
            range(0, self.use_var_len), dtype=torch.int64, device=self.device)
        self.used_var_list = torch.zeros(
            var_size, dtype=torch.int64, device=self.device)
        self.sample_prob = torch.ones(
            var_size, dtype=torch.float32, device=self.device) / var_size

        # upper and lower bounds for the modifier
        self.modifier_up = torch.zeros(
            var_size, dtype=torch.float32, device=self.device)
        self.modifier_down = torch.zeros(
            var_size, dtype=torch.float32, device=self.device)

        # random permutation for coordinate update
        self.perm = torch.randperm(var_size)
        self.perm_index = 0

        # ADAM status
        self.mt = torch.zeros(
            var_size, dtype=torch.float32, device=self.device)
        self.vt = torch.zeros(
            var_size, dtype=torch.float32, device=self.device)
        # self.beta1 = 0.8
        # self.beta2 = 0.99
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.reset_adam_after_found = reset_adam_after_found
        self.adam_epoch = torch.ones(
            var_size, dtype=torch.int64, device=self.device)
        self.stage = 0
        # variables used during optimization process
        self.grad = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device)
        self.hess = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device)
        # compile numba function
        # self.coordinate_ADAM_numba = jit(coordinate_ADAM, nopython = True)
        # self.coordinate_ADAM_numba.recompile()
        # print(self.coordinate_ADAM_numba.inspect_llvm())
        # np.set_printoptions(threshold=np.nan)
        # set solver
        solver = solver.lower()
        self.solver_name = solver
        if solver == "adam":
            self.solver = coordinate_ADAM
        if solver == "adam_torch":
            self.solver = coordinate_ADAM_torch
        elif solver == "newton":
            self.solver = coordinate_Newton
        elif solver == "adam_newton":
            self.solver = coordinate_Newton_ADAM
        elif solver != "fake_zero":
            print("unknown solver", solver)
            self.solver = coordinate_ADAM
        print("Using", solver, "solver")

    def get_new_prob(self, prev_modifier, gen_double=False):
        prev_modifier = torch.squeeze(prev_modifier)
        old_shape = prev_modifier.shape
        if gen_double:
            new_shape = (old_shape[0]*2, old_shape[1]*2, old_shape[2])
        else:
            new_shape = old_shape
        prob = torch.empty(shape=new_shape, dtype=torch.float32)
        for i in range(prev_modifier.shape[2]):
            image = torch.abs(prev_modifier[:, :, i])

            image_pool = torch.nn.functional.max_pool2d(
                image, old_shape[0] // 8)
            if gen_double:
                prob[:, :, i] = scipy.misc.imresize(
                    image_pool, 2.0, 'nearest', mode='F')
            else:
                prob[:, :, i] = image_pool
        prob /= torch.sum(prob)
        return prob

    def resize_img(self, small_x, small_y, reset_only=False):
        self.small_x = small_x
        self.small_y = small_y
        small_single_shape = (self.small_x, self.small_y, self.num_channels)
        if reset_only:
            self.real_modifier = torch.zeros(
                (1,) + small_single_shape, dtype=torch.float32, device=self.device)
        else:
            # run the resize_op once to get the scaled image
            prev_modifier = torch.clone(self.real_modifier)
            self.real_modifier = torch.nn.functional.interpolate(
                self.real_modifier, size=[self.small_x, self.small_y], mode='bilinear')
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = torch.tensor(
            range(0, self.use_var_len), dtype=torch.int64, device=self.device)
        # ADAM status
        self.mt = torch.zeros(
            var_size, dtype=torch.float32, device=self.device)
        self.vt = torch.zeros(
            var_size, dtype=torch.float32, device=self.device)
        self.adam_epoch = torch.ones(
            var_size, dtype=torch.int64, device=self.device)
        # update sample probability
        if reset_only:
            self.sample_prob = torch.ones(
                var_size, dtype=torch.float32, device=self.device) / var_size
        else:
            self.sample_prob = self.get_new_prob(prev_modifier, True)
            self.sample_prob = self.sample_prob.reshape(var_size)

    def fake_blackbox_optimizer(self):
        # for testing
        # self.real_modifier.requires_grad = True
        self.compute_loss(self.real_modifier)
        # self.total_loss.backward()
        # true_grads = self.real_modifier.grad.data
        true_grads = torch.autograd.grad(self.total_loss, self.real_modifier)

        losses, l2s, loss1, loss2, scores, nimgs = self.total_loss, self.l2dist, self.loss1, self.loss2, self.output, self.newimg

        # ADAM update
        grad = true_grads[0].reshape(-1).cpu().numpy()
        # print(true_grads[0])
        epoch = self.adam_epoch[0]
        mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        vt = self.beta2 * self.vt + (1 - self.beta2) * torch.square(grad)
        corr = (math.sqrt(1 - self.beta2 ** epoch)) / (1 - self.beta1 ** epoch)
        # print(grad.shape, mt.shape, vt.shape, self.real_modifier.shape)
        # m is a *view* of self.real_modifier
        m = self.real_modifier.reshape(-1)
        # this is in-place
        m -= self.LEARNING_RATE * corr * (torch.tensor(mt, device=self.device) / (
            torch.sqrt(torch.tensor(vt, device=self.device)) + 1e-8))
        self.mt = mt
        self.vt = vt
        # m -= self.LEARNING_RATE * grad
        if not self.use_tanh:
            m_proj = torch.maximum(torch.minimum(
                m, self.modifier_up), self.modifier_down)
            # np.copyto(m, m_proj)
            m = m_proj.clone()
        self.adam_epoch[0] = epoch + 1
        return losses[0], l2s[0], loss1[0], loss2[0], scores[0], nimgs[0]

    def blackbox_optimizer(self, iteration):
        # build new inputs, based on current variable value
        # var = np.repeat(self.real_modifier, self.batch_size * 2 + 1, axis=0)
        var = torch.repeat_interleave(
            self.real_modifier, self.batch_size * 2 + 1, axis=0)
        var_size = self.real_modifier.numel()
        # print(s, "variables remaining")
        # var_indice = np.random.randint(0, self.var_list.size, size=self.batch_size)
        if self.use_importance:
            # var_indice = np.random.choice(
            #     self.var_list.size, self.batch_size, replace=False, p=self.sample_prob)
            idx = self.sample_prob.multinomial(
                num_samples=self.batch_size, replacement=False)
            var_indice = torch.arange(
                self.var_list.numel(), device=self.device)[idx]
        else:
            # var_indice = np.random.choice(
            #     self.var_list.size, self.batch_size, replace=False)
            var_indice = torch.randperm(self.var_list.numel(), device=self.device)[
                :self.batch_size]

        indice = self.var_list[var_indice]
        # indice = self.var_list
        # regenerate the permutations if we run out
        # if self.perm_index + self.batch_size >= var_size:
        #     self.perm = torch.randperm(var_size)
        #     self.perm_index = 0
        # indice = self.perm[self.perm_index:self.perm_index + self.batch_size]
        # b[0] has the original modifier, b[1] has one index added 0.0001
        for i in range(self.batch_size):
            var[i * 2 + 1].reshape(-1)[indice[i]] += 0.0001
            var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.0001

        self.compute_loss(var)
        losses, l2s, loss1, loss2, scores, nimgs = self.total_loss, self.l2dist, self.loss1, self.loss2, self.output, self.newimg

        # losses = self.sess.run(self.total_loss, feed_dict={self.modifier: var})
        # t_grad = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
        # self.grad = t_grad[0].reshape(-1)
        # true_grads = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
        # self.coordinate_ADAM_numba(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        # coordinate_ADAM(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        # coordinate_ADAM(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh, true_grads)
        # coordinate_Newton(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        # coordinate_Newton_ADAM(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        self.solver(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier,
                    self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)

        # adjust sample probability, sample around the points with large gradient
        if self.save_ckpts:
            torch.save(self.real_modifier, '{}/iter{}'.format(self.save_ckpts,
                                                              iteration))

        if self.real_modifier.shape[0] > self.resize_init_size:
            self.sample_prob = self.get_new_prob(self.real_modifier)
            # self.sample_prob = self.get_new_prob(tmp_mt.reshape(self.real_modifier.shape))
            self.sample_prob = self.sample_prob.reshape(var_size)

        # if the gradient is too small, do not optimize on this variable
        # self.var_list = np.delete(self.var_list, indice[torch.abs(self.grad) < 5e-3])
        # reset the list every 10000 iterations
        # if iteration%200 == 0:
        #    print("{} variables remained at last stage".format(self.var_list.size))
        #    var_size = self.real_modifier.size
        #    self.var_list = torch.tensor(range(0, var_size))
        return losses[0], l2s[0], loss1[0], loss2[0], scores[0], nimgs[0]
        # return losses[0]

    def compute_loss(self, modifier):
        if self.use_resize:
            self.modifier = modifier
            # scaled up image
            self.scaled_modifier = torch.nn.functional.interpolate(
                self.modifier, size=[self.image_size, self.image_size], mode='bilinear')
            # operator used fo
        else:
            # no resize
            self.scaled_modifier = modifier

        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        # broadcast self.true_img to every dimension of modifier

        if self.use_tanh:
            self.newimg = torch.tanh(self.scaled_modifier + self.true_img)/2
        else:
            self.newimg = self.scaled_modifier + self.true_img

        # prediction BEFORE-SOFTMAX of the model
        # now we have output at #batch_size different modifiers
        # the output should have shape (batch_size, num_labels)

        self.output = model(self.newimg+0.5)
        if use_log:
            self.output = F.softmax(self.output, -1)

        # distance to the input data
        if self.use_tanh:
            self.l2dist = torch.sum(torch.square(
                self.newimg-torch.tanh(self.true_img)/2), [1, 2, 3])
        else:
            self.l2dist = torch.sum(torch.square(
                self.newimg - self.true_img), [1, 2, 3])

        # compute the probability of the label class versus the maximum other
        # self.true_label_1hot * self.output selects the Z value of real class
        # because self.true_label_1hot is an one-hot vector
        # the reduce_sum removes extra zeros, now get a vector of size #batch_size

        self.real = torch.sum((self.true_label_1hot)*self.output, 1)
        # (1-self.true_label_1hot)*self.output gets all Z values for other classes
        # Because soft Z values are negative, it is possible that all Z values are less than 0
        # and we mistakenly select the real class as the max. So we minus 10000 for real class
        self.other = torch.max(
            (1-self.true_label_1hot)*self.output - (self.true_label_1hot*10000), 1)[0]

        # If self.targeted is true, then the targets represents the target labels.
        # If self.targeted is false, then targets are the original class labels.
        if self.TARGETED:
            if use_log:
                # loss1 = - torch.log(self.real)
                # loss1 = torch.maximum(0.0, torch.log(
                #     self.other + 1e-30) - torch.log(self.real + 1e-30))
                loss1 = torch.clamp(
                    torch.log(self.other + 1e-30) - torch.log(self.real + 1e-30), min=-self.CONFIDENCE)
            else:
                # if targetted, optimize for making the other class (real) most likely
                # loss1 = torch.maximum(
                #     0.0, self.other-self.real+self.CONFIDENCE)

                loss1 = torch.clamp(self.other-self.real, min=-self.CONFIDENCE)
        else:
            if use_log:
                # loss1 = torch.log(self.real)
                # loss1 = torch.maximum(0.0, torch.log(
                #     self.real + 1e-30) - torch.log(self.other + 1e-30))

                loss1 = torch.clamp(torch.log(
                    self.real + 1e-30) - torch.log(self.other + 1e-30), min=-self.CONFIDENCE)
            else:
                # if untargeted, optimize for making this class least likely.
                # loss1 = torch.maximum(
                #     0.0, self.real-self.other+self.CONFIDENCE)

                loss1 = torch.clamp(self.real-self.other, min=-self.CONFIDENCE)

        # sum up the losses (output is a vector of #batch_size)
        self.loss2 = self.l2dist
        self.loss1 = self.c*loss1
        self.total_loss = self.loss1+self.loss2

    # only accepts 1 image at a time. Batch is used for gradient evaluations at different points

    def attack_batch(self, img, label_1hot):
        """
        Run the attack on a batch of images and labels.
        """
        def is_confidently_fooled(x, true_label):
            if not isinstance(x, (float, int, np.int64)) and not (isinstance(x, torch.Tensor) and x.numel() == 1):
                z = torch.clone(x)
                if self.TARGETED:
                    z[true_label] -= self.CONFIDENCE
                else:
                    z[true_label] += self.CONFIDENCE
                z = torch.argmax(z)
            else:
                z = x

            if self.TARGETED:
                return z == true_label
            else:
                return z != true_label

        # convert img to float32 to avoid numba error
        img = img.type(torch.float32)

        if torch.argmax(model(img+0.5)) != torch.argmax(label_1hot):
            print("Image is already misclassified.")
            return img, 0.0

        # remove the extra batch dimension
        if len(img.shape) == 4:
            img = img[0]
        if len(label_1hot.shape) == 2:
            label_1hot = label_1hot[0]
        # convert to tanh-space
        if self.use_tanh:
            img = torch.arctanh(img*1.999999)

        # set the lower and upper bounds accordingly
        c_lower_bound = 0.0
        c = self.initial_c
        c_upper_bound = 1e10

        # set the upper and lower bounds for the modifier
        if not self.use_tanh:
            self.modifier_up = 0.5 - img.reshape(-1)
            self.modifier_down = -0.5 - img.reshape(-1)

        # clear the modifier
        # if not self.load_checkpoint:
        #     if self.use_resize:
        #         self.resize_img(self.resize_init_size,
        #                         self.resize_init_size, True)
        #     else:
        #         self.real_modifier = torch.zeros(
        #             (1,) + (self.num_channels, self.small_x, self.small_y), dtype=torch.float32, device=self.device)
        #         if self.solver_name == "fake_zero":
        #             self.real_modifier.requires_grad = True

        # the best l2, score, and image attack
        outer_best_c = c
        outer_best_l2 = 1e10
        outer_best_score = -1
        if self.use_tanh:
            outer_best_adv = torch.tanh(img)/2
        else:
            outer_best_adv = img

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(outer_best_l2)

            best_l2 = 1e10
            best_score = -1

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                c = c_upper_bound

            # set the variables so that we don't have to send them over again
            # self.setup = []
            self.true_img = img.detach().clone()
            self.true_label_1hot = label_1hot.detach().clone()
            self.c = c
            # self.setup = [self.true_img, self.true_label_1hot, self.c]

            # use the current best model
            # np.copyto(self.real_modifier, outer_best_adv - img)
            # use the model left by last constant change

            prev_loss = 1e6
            train_timer = 0.0
            last_loss1 = 1.0
            if not self.load_checkpoint:
                if self.use_resize:
                    self.resize_img(self.resize_init_size,
                                    self.resize_init_size, True)
                else:
                    self.real_modifier = torch.zeros(
                        (1,) + (self.num_channels, self.small_x, self.small_y), dtype=torch.float32, device=self.device)
                    if self.solver_name == "fake_zero":
                        self.real_modifier.requires_grad = True

            # reset ADAM status
            self.mt.fill_(0.0)
            self.vt.fill_(0.0)
            self.adam_epoch.fill_(1)
            self.stage = 0
            multiplier = 1
            eval_costs = 0
            if self.solver_name != "fake_zero":
                multiplier = 24
            for iteration in range(self.start_iter, self.MAX_ITERATIONS):
                if self.use_resize:
                    if iteration == 2000:
                        # if iteration == 2000 // 24:
                        self.resize_img(64, 64)
                    if iteration == 10000:
                        # if iteration == 2000 // 24 + (10000 - 2000) // 96:
                        self.resize_img(128, 128)
                    # if iteration == 200*30:
                    # if iteration == 250 * multiplier:
                    #     self.resize_img(256,256)
                # print out the losses every 10%
                if iteration % (self.print_every) == 0:
                    # print(iteration,self.sess.run((self.total_loss,self.real,self.other,self.loss1,self.loss2), feed_dict={self.modifier: self.real_modifier}))

                    self.compute_loss(self.real_modifier)

                    total_loss, real, other, loss1, loss2 = self.total_loss, self.real, self.other, self.loss1, self.loss2
                    print("[STATS][L2] iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, real = {:.5g}, other = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(
                        iteration, eval_costs, train_timer, self.real_modifier.shape, total_loss[0], real[0], other[0], loss1[0], loss2[0]))
                    sys.stdout.flush()
                    # np.save('black_iter_{}'.format(iteration), self.real_modifier)

                attack_begin_time = time.time()
                # perform the attack
                if self.solver_name == "fake_zero":
                    total_loss, l2, loss1, loss2, score, nimg = self.fake_blackbox_optimizer()
                else:
                    total_loss, l2, loss1, loss2, score, nimg = self.blackbox_optimizer(
                        iteration)

                if self.solver_name == "fake_zero":
                    eval_costs += self.real_modifier.numel()
                else:
                    eval_costs += self.batch_size

                # reset ADAM states when a valid example has been found
                if loss1 == 0.0 and last_loss1 != 0.0 and self.stage == 0:
                    # we have reached the fine tunning point
                    # reset ADAM to avoid overshoot
                    if self.reset_adam_after_found:
                        self.mt.fill_(0.0)
                        self.vt.fill_(0.0)
                        self.adam_epoch.fill_(1)
                    self.stage = 1
                last_loss1 = loss1

                # check if we should abort search if we're getting nowhere.
                # if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                if self.ABORT_EARLY and iteration % self.early_stop_iters == 0:
                    if total_loss > prev_loss*.9999:
                        print("Early stopping because there is no improvement")
                        break
                    prev_loss = total_loss

                # adjust the best result found so far
                # the best attack should have the target class with the largest value,
                # and has smallest l2 distance

                if l2 < best_l2 and is_confidently_fooled(score, torch.argmax(label_1hot)):
                    best_l2 = l2
                    best_score = torch.argmax(score)
                if l2 < outer_best_l2 and is_confidently_fooled(score, torch.argmax(label_1hot)):
                    # print a message if it is the first attack found
                    if outer_best_l2 == 1e10:
                        print("[STATS][L3](First valid attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, l2 = {:.5g}".format(
                            iteration, eval_costs, train_timer, self.real_modifier.shape, total_loss, loss1, loss2, l2))
                        sys.stdout.flush()
                    outer_best_l2 = l2
                    outer_best_score = torch.argmax(score)
                    outer_best_adv = nimg
                    outer_best_c = c

                train_timer += time.time() - attack_begin_time

            # adjust the constant as needed

            if is_confidently_fooled(best_score, torch.argmax(label_1hot)) and best_score != -1:
                # success, divide const by two
                print('old c: ', c)
                c_upper_bound = min(c_upper_bound, c)
                if c_upper_bound < 1e9:
                    c = (c_lower_bound + c_upper_bound)/2
                print('new c: ', c)
            else:
                # failure, either multiply by 10 if no solution found yet
                #          or do binary search with the known upper bound
                print('old c: ', c)
                c_lower_bound = max(c_lower_bound, c)
                if c_upper_bound < 1e9:
                    c = (c_lower_bound + c_upper_bound)/2
                else:
                    c *= 10
                print('new c: ', c)

        if self.use_tanh:
            img = torch.tanh(img)/2

        # return the best solution found
        return outer_best_adv, outer_best_c

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        adv_images = torch.zeros_like(imgs)
        print('go up to', len(imgs))
        # we can only run 1 image at a time, minibatches are used for gradient evaluation
        for i in range(0, len(imgs)):
            print('tick', i)
            adv_images[i] = self.attack_batch(imgs[i], targets[i])[0]

        return adv_images


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets_1hot = []
    i = 0
    samples_sofar = 0
    while samples_sofar < samples:
        i += 1
        if torch.argmax(model(torch.tensor(data.test_data[start+i:start+i+1]+0.5, device="cuda", dtype=torch.float32).permute(0, 3, 1, 2))) != np.argmax(data.test_labels_1hot[start+i]):
            continue

        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels_1hot.shape[1])

            # print ('image label:', torch.argmax(data.test_labels[start+i]))
            for j in seq:
                # skip the original image label
                if (j == torch.argmax(data.test_labels_1hot[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets_1hot.append(
                    torch.eye(data.test_labels_1hot.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets_1hot.append(data.test_labels_1hot[start+i])

        samples_sofar += 1

    inputs = torch.tensor(inputs).permute(0, 3, 1, 2)
    targets_1hot = torch.tensor(targets_1hot)

    return inputs, targets_1hot


if __name__ == "__main__":
    from ..parameters import get_arguments

    args = get_arguments()
    args.adv_testing.box_type = "other"
    args.adv_testing.otherbox_method = "zoo"

    np.random.seed(42)
    torch.manual_seed(42)
    use_log = True
    use_tanh = True

    _, test_loader = cifar10(args)
    test_loader.test_data = test_loader.dataset.data/255-0.5
    test_loader.test_labels_1hot = np.eye(10)[test_loader.dataset.targets]

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    classifier = get_classifier(args)

    if args.neural_net.no_frontend:
        model = classifier

    else:
        frontend = get_frontend(args)

        model = Combined(frontend, classifier)

    model = model.to(device)
    model.eval()
    model.num_channels = 3
    model.image_size = 32
    model.num_labels = 10

    nb_samples = args.adv_testing.nb_imgs
    inputs, targets = generate_data(test_loader, samples=nb_samples, targeted=False,
                                    start=np.random.randint(0, 10000-nb_samples), inception=False)
    inputs, targets = inputs.to(device), targets.to(device)

    attack = BlackBoxL2(model, batch_size=128,
                        max_iterations=1000, confidence=0, use_log=use_log, device=device, solver="adam_torch", use_tanh=use_tanh)

    # inputs = inputs[1:2]
    # targets = targets[1:2]
    timestart = time.time()
    adv = attack.attack(inputs, targets)
    timeend = time.time()

    print("Took", timeend-timestart,
          "seconds to run", len(inputs), "samples.")

    adv += 0.5
    inputs += 0.5

    true_class = np.argmax(targets.cpu().numpy(), -1)
    clean_class = np.argmax(model(inputs.float()).cpu().numpy(), -1)
    adv_class = np.argmax(model(adv.float()).cpu().numpy(), -1)

    acc = ((true_class == adv_class).sum())/len(inputs)
    print("True label: ", true_class)
    print("Clean Classification: ", clean_class)
    print("Adversarial Classification: ", adv_class)
    print("Success Rate: ", (1.0-acc)*100.0)

    fooled_indices = (true_class != adv_class)

    attack = adv-inputs

    # plt.figure(figsize=(10, 10))
    # for i in range(3):
    #     plt.subplot(3, 3, 3*i+1)
    #     plt.imshow(inputs[fooled_indices]
    #                [i].detach().cpu().permute(1, 2, 0).numpy())
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(3, 3, 3*i+2)
    #     plt.imshow(adv[fooled_indices]
    #                [i].detach().cpu().permute(1, 2, 0).numpy())
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(3, 3, 3*i+3)
    #     plt.imshow(attack[fooled_indices]
    #                [i].detach().cpu().permute(1, 2, 0).numpy())
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.tight_layout()
    # plt.savefig('asd.pdf')

    print("Average distortion: ", torch.mean(
        torch.sum((adv[fooled_indices]-inputs[fooled_indices])**2, dim=(1, 2, 3))**.5).item())

    if args.adv_testing.save:
        attack_filepath = attack_file_namer(args)

        if not os.path.exists(os.path.dirname(attack_file_namer(args))):
            os.makedirs(os.path.dirname(attack_file_namer(args)))

        np.save(attack_filepath, adv.detach().cpu().numpy())
