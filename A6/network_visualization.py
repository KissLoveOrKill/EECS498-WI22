"""
Implements a network visualization in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from network_visualization.py!")


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # 中文提示：请参照对应 Notebook 的“中文任务提示”完成本处任务。
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    # Hint: X.grad.data stores the gradients                                     #
    ##############################################################################
    # Replace "pass" statement with your code
    # 1. 前向传播，得到每张图片的各类别分数：(N, 1000)
    scores = model(X)

    # 2. 取出每张图片正确类别的分数：(N, 1)
    correct_scores = scores.gather(1, y[:, None])

    # 3. 求这些分数对输入像素的梯度：(N, 3, H, W)
    grads = torch.autograd.grad(correct_scores.sum(), X)[0]

    # 4. 取梯度绝对值，再沿 RGB 通道取最大值：(N, H, W)
    saliency = grads.abs().amax(dim=1)
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the pogress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our adversarial attack to the input image, and make it require
    # gradient
    X_adv = X.detach().clone().requires_grad_(True)

    learning_rate = 1
    ##############################################################################
    # 中文提示：请参照对应 Notebook 的“中文任务提示”完成本处任务。
    # TODO: Generate an adversarial attack X_adv that the model will classify    #
    # as the class target_y. You should perform gradient ascent on the score     #
    # of the target class, stopping when the model is fooled.                    #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate an adversarial     #
    # attack in fewer than 100 iterations of gradient ascent.                    #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    for iteration in range(max_iter):
        # 计算目标类别分数，并检查当前图像是否已经被分到目标类别。
        scores = model(X_adv)
        target_score = scores[0, target_y]
        max_score, predicted_y = scores[0].max(dim=0)
        if verbose:
            print(
                "Iteration %d: target score %.3f, max score %.3f"
                % (iteration, target_score.item(), max_score.item())
            )
        if predicted_y.item() == target_y:
            break

        # 只计算目标分数对图像的梯度，不累积模型参数的梯度。
        grad = torch.autograd.grad(target_score, X_adv)[0]
        grad_norm = grad.norm()
        if grad_norm.item() == 0:
            break

        # 梯度上升：沿 L2 归一化后的梯度修改图像，保持每步步长为 1。
        with torch.no_grad():
            X_adv.add_(learning_rate * grad / grad_norm)
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    ########################################################################
    # 中文提示：请参照对应 Notebook 的“中文任务提示”完成本处任务。
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    scores = model(img)
    # 最大化：目标类别分数 - L2 正则项
    objective = scores[0, target_y] - l2_reg * img.square().sum()
    # 计算这个目标对图像的梯度
    grad = torch.autograd.grad(objective, img)[0]
    # 梯度上升，原地更新图像
    with torch.no_grad():
        img.add_(learning_rate * grad)
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
