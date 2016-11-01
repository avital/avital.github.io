---
layout: post
title: Domain Adaptation by Backpropagation
date: 2016-10-24 18:27:31
image:
  feature: domain-adaptation-by-backprop.png
title_image: images/domain-adaptation-by-backprop-title.png
excerpt_separator: <!--more-->
---


In supervised learning, we train neural networks on a large set of
labelled examples. We test the accuracy of a trained model on a
held-out test set. A common choice for the test is a small random
selection of training examples. *This is fine when the deployed model
sees the same distribution as the training set. Often, though, this
isn't the case.*<!--more-->

Let's say we're trying to build a system that predicts whether an image is a cat or a dog. And there are many examples out there for images with a label for "cat" or "dog". We can train a neural network on this dataset and get very high accuracy on a held-out portion of the same dataset from which we trained the network. From a statistical perspective, we say that the images were sampled from a distribution (which we don't know). And testing on a held-out portion of data means that the model that we trained on a portion of our data has is accurate when predicting on a held-out portion sampled from the *same distribution*.

But what if we want our model to predict well on images from a different distribution?

For example, maybe we want to guess "cat" or "dog" from images that came from an old kind of camera, where the colors come out very different. Or maybe all the images we have are blurry in a certain kind of way.

If we just trained a model on images from modern cameras (the "source" distribution) and try to predict on older cameras (the "target" distribution), our model will perform much more poorly than we'd expect by looking at the test accuracy alone.

How do we solve this problem? The first observation is that there definitely is a model that would correctly predict on images from both types of cameras. If we had access to an equally large dataset of labelled images taken from the old camera, we could train the model from the combined distribution of images taken from both kinds of cameras. Our model then could learn to predict well for images from both distributions.

Here's the kicker though. We don't have labels for images taken from the old camera. In fact, we may not even have any images of dogs or cats taken with the old camera. We just have a huge pile of images. With no labels whatsoever.

Here comes domain adaptation, or more specifically, "covariate shift".

We want to learn from the distribution of target images (without labels), in addition to the distribution of source images (with labels). And our output should be a model that can predict correctly for images from both distributions.

I recently read a cool paper about this topic, that presents a very clean approach for adding this kind of ability to any feed-forward neural network. This includes convolutional neural networks commonly used for image classification.

I am personally working on audio applications but using SFTF spectrograms one can (sort of) convert audio to images, so I'm hoping the same approach might work.

## Unsupervised Domain Adaptation by Backpropagation

Here's the paper: [https://arxiv.org/abs/1409.7495]

In short, what they do is take a neural network architecture designed for image classification. They split the network into two portions. The first portion is the "feature extractor", where high-level features of the images are discovered (e.g. "circle", "nose", ...). The second portion is the "label predictor", where these features are combined to predict "cat" or "dog". The split between the two section isn't well-defined and probably some trial and error is needed to choose a good split. Maybe the common split is between the convolution layers and the fully-connected layers commons used at the end of a CNN.

Then, they add another set of parallel layers on top of the feature extractor, called the domain classifier. This section of the network tries to learn to distinguish whether an input was sampled from the source or target domains. In our case, it learns to distinguish whether an image was taken from an new camera or an old camera.

The kicker is how we train this network. We train it so that the combination of (feature extractor, label predictor) is as good as we can make it at predicting the label (in our case, "cat" or "dog") for images taken with the new camera. We also train it so that the combination of (feature extractor, domain classifier) is as *bad* as possible at predicting which domain the image came from. But!! We want the domain classifier itself, if we don't change the feature extractor to be as *good* as it can at predicting the domain. The domain classifier is fighting the feature extractor -- the feature extractor changes as to make the domain classifier do a poor job at classifying, while the domain classifier changes as to make that part of the feature extractor's job fail. All of this is happening *while* the label predictor is learning to get better at emitting "cat" or "dog.

Over time, the feature extractor layers end up finding the common features that aren't distinguishable between the source and target distributions, while the label predictor learns to predict "dog" or "cat" for images from the source distribution (remember, that's the only distribution on which we have labels). So, if this game works well, then at the end we can drop the domain classifier portion and we've trained a neural network to predict "dog" or "cat" for images that come from either the source or target distributions!

## Training

How can we train this model? We can’t apply backpropagation directly, because backpropagation will make all layers of the network try to minimize the same loss function. Let’s see what we want the three portions of the network to do:

1. The label predictor is trying to get better at predicting “dog” or “cat” for `G_f(input)` where `input` is sampled from the source distribution
2. The domain classifier is trying to get better at predicting “source” or “target” for `G_f(input)` where `input` is sampled from either distribution
3. The feature extractor is trying to make the label predictor right
4. The feature extractor is also trying to make the domain classifier *wrong*

The reason back propagation can’t work as-is, is that we can’t devise a loss function for the network that gets both:
a. domain classifier getting worse (see 3 above)
b. domain classifier getting better (see 2 above)

How do we fix this?

Let’s start with our loss function. The loss function we’ll use is:

(insert loss function)

So, minimizing this loss function would correctly achieve (1), (2) and (3). But it would lead to the opposite of (4).

The fix is (drumroll…): When back-propagating gradients from the domain classifier back to the feature extractor, we *reverse* the gradient by multiplying it by `-\lambda` where `\lambda` is a hyperparameter. We do this by adding a layer called a “gradient reversal layer” that breaks the normal rules of differentiation by passing as a no-op on the forward prop, while  multiplying by `-\lambda` on the backprop. This layer can be easily implemented in any deep learning library.
