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

Let's say we're predicting whether an
image is a cat or a dog. We have access to tons of cat and dog images. These images were all taken from the same camera. We train our neural network on this dataset and get high accuracy on a held-out validation set. But when we deploy this model, it mispredicts images uploaded by users. Those images all came from a phone camera.

We can explain this with statistical language. Our training set was sampled from the distribution of cat and dog images from one camera. We call this the "source distribution". The distribution of test examples, the "target distribution", is different. This problem is formally known as "domain adaptation"/"domain transfer", or specifically, "covariate shift".

How do we fix our model to perform well on the target distribution?

We could collect tons of labelled images from the target distribution. Then we could train a new model from a combination of the source and target distributions. If we can do that, we'll end up with a model that predicts well for both distributions.

But collecting those labels could be expensive or even impossible. Maybe the images we can get from the target distribution aren't even dogs or cats. What if all we have is a ton of images from the target distribution? With no labels. Can we use those to improve our model?

## Unsupervised Domain Adaptation by Backpropagation

Ganin and Lempitsky wrote [a great paper](https://arxiv.org/abs/1409.7495) on solving this problem for neural networks. Their solution is simple. It works for any feed-forward neural network, such as ConvNets. It is easy to implement it in any deep learning framework. And it beats state of the art on standard [domain adaptation datasets](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code).

Here's what they do. Start with a neural network architecture proven to work for your classification problem. The paper uses LeNet-5 for MNIST and AlexNet for ImageNet. Then, split the network into two
parts. The first part, the "feature extractor", consists of the lower layers. The feature extractor detects high-level features of the images. The second part, the "label predictor", consists of the upper layers. The label predictor is where features are combined to predict "cat" or "dog". In the paper, they always split at the point where fully-connected layers start.

(image from top of page 10)

XCXC stop here

Then, they add another set of parallel layers on top of the feature
extractor, called the domain classifier. This section of the network
tries to learn to distinguish whether an input was sampled from the
source or target domains. In our case, it learns to distinguish
whether an image was taken from an new camera or an old camera.

The kicker is how we train this network. We train it so that the
combination of (feature extractor, label predictor) is as good as we
can make it at predicting the label (in our case, "cat" or "dog") for
images taken with the new camera. We also train it so that the
combination of (feature extractor, domain classifier) is as *bad* as
possible at predicting which domain the image came from. But!! We want
the domain classifier itself, if we don't change the feature extractor
to be as *good* as it can at predicting the domain. The domain
classifier is fighting the feature extractor -- the feature extractor
changes as to make the domain classifier do a poor job at classifying,
while the domain classifier changes as to make that part of the
feature extractor's job fail. All of this is happening *while* the
label predictor is learning to get better at emitting "cat" or "dog.

Over time, the feature extractor layers end up finding the common
features that aren't distinguishable between the source and target
distributions, while the label predictor learns to predict "dog" or
"cat" for images from the source distribution (remember, that's the
only distribution on which we have labels). So, if this game works
well, then at the end we can drop the domain classifier portion and
we've trained a neural network to predict "dog" or "cat" for images
that come from either the source or target distributions!

## Training

How can we train this model? We can’t apply backpropagation directly,
because backpropagation will make all layers of the network try to
minimize the same loss function. Let’s see what we want the three
portions of the network to do:

1. The label predictor is trying to get better at predicting “dog” or
“cat” for `G_f(input)` where `input` is sampled from the source
distribution

2. The domain classifier is trying to get better at predicting
“source” or “target” for `G_f(input)` where `input` is sampled from
either distribution

3. The feature extractor is trying to make the label predictor right

4. The feature extractor is also trying to make the domain classifier
*wrong*

The reason back propagation can’t work as-is, is that we can’t devise
a loss function for the network that gets both:

a. domain classifier getting worse (see 3 above)

b. domain classifier getting better (see 2 above)

How do we fix this?

Let’s start with our loss function. The loss function we’ll use is:

(insert loss function)

So, minimizing this loss function would correctly achieve (1), (2) and
(3). But it would lead to the opposite of (4).

The fix is (drumroll...): When back-propagating gradients from the
domain classifier back to the feature extractor, we *reverse* the
gradient by multiplying it by `-\lambda` where `\lambda` is a
hyperparameter. We do this by adding a layer called a “gradient
reversal layer” that breaks the normal rules of differentiation by
passing as a no-op on the forward prop, while multiplying by
`-\lambda` on the backprop. This layer can be easily implemented in
any deep learning library.