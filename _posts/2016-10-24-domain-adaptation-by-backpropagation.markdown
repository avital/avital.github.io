---
layout: post
title: Domain Adaptation by Backpropagation
date: 2016-10-24 18:27:31
image:
  feature: domain-adaptation-by-backprop.png
title_image: images/domain-adaptation-by-backprop-title.png
excerpt_separator: <!--more-->
---

In supervised learning, we train neural networks on a ton of
labelled examples. We test the accuracy of a trained model on a
held-out test set. A common choice for the test set is a random
selection of 20% of the training examples. *This is fine when the deployed model
sees the same distribution as the training set. In practice, this often
isn't the case*. We can solve this with **domain adaptation**. All we need is an
**unlabelled data set** from the distribution the model tests on.
I'll describe a simple technique by Ganin and Lempitsky to do that.<!--more-->

This post explains the approach proposed in [this great
paper](https://arxiv.org/abs/1409.7495). It's a simple solution that
works for any feed-forward network, such as ConvNets. It is easy to
implement in any deep learning framework. And it beats state of the
art on standard [domain adaptation
datasets](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code).

## Background: Covariate Shift

Covariate shift and domain adaptation are relevant for any machine
learning problem. I'll illustrate the problem and this solution using
the following toy machine learning problem.

Let's say we're predicting whether an image is a cat or a dog. We have
access to tons of cat and dog images. These images were all taken from
a few different camera types. We train our neural network on this
dataset and get 90% accuracy on a held-out validation set. Woohoo! Success!

But then...

When we deploy our model, it mispredicts many images uploaded by
certain users. Turns out, our model is inaccurate when photos come
from certain phone cameras. We didn't have images from that camera
in our training set. Images taken from those cameras look different
enough, so that our model just doesn't work well on them. Here's an example:

<img src="images/domain-adaptation-example.png" />

This might seem like an overfitting problem. The #1 way to solve
overfitting is to collect more data. But that only works if the new
data comes from the distribution you'll be testing on. It might be
expensive to get labelled photos of dogs and cats from this other
camera.

Another case where domain adaptation can help is when training on
synthetic data. It can be extremely cheap to generate a synthetic data
set. But a model trained on synthetic data might not predict well on
real world examples. How can we adapt the learning from the synthetic
dataset, so that it works well on the real dataset? (I wonder if
this comes up when training reinforcement learning models via
simulators such as MuJoCo. Does anyone know?)

Statistically, we say that our training set is sampled from a **source
distribution**. And our test examples are sampled from a **target
distribution**. We don't know either of these distributions directly. But
we approximate them with datasets. The fact that these
distributions differ is called
*covariate shift*. Techniques to solve this problem are called *domain
adaptation*. (I think I'm a little wrong about the exact terminology here --
I'd appreciate a clarification.)

To solve this, we need to know something about the target
distribution. Here's the key insight. Collecting labelled data from
the target distribution might be expensive. But collecting **unlabelled
data** might be easy. In the example described above,
*we get free data from users
taking photos with our app*.

To summarize, we have:
1. A labelled dataset from the source distribution
2. An unlabelled dataset from the target distribution

How can we train a model to predict labels well on both datasets?

## Unsupervised Domain Adaptation by Backpropagation

Here's Ganin and Lempitsky's approach, which they call _Unsupervised
Domain Adaptation by Backpropagation_.

They start with a neural network architecture designed for a
particular classification problem. For example, LeNet-5 for MNIST and
AlexNet for ImageNet. Then, they give names to two parts of the
network. The first part, the "feature extractor", consists of the
lower layers. The feature extractor detects high-level features of the
images. The second part, the "label predictor", consists of the upper
layers. The label predictor combines features to predict "cat" or
"dog". In the paper, the split is at the point where fully-connected
layers start.

<img src="images/domain-adaptation-mnist.png" />

Then, they add parallel layers on top of the feature extractor, called
the domain classifier. In the diagram, these appear on the bottom in
pink. This new part of the network will learn to detect whether an
image came from the source or target domains. What's this "GRL" circle
you ask? Wait for it...

The kicker (and the insight behing GRL) is how we train this
network. We want [feature extractor → label predictor] to be good at
predicting "cat" or "dog". We only have labels for images from the
source distribution, so this only makes sense for those.

We also want
[feature extractor → domain classifier] to be **bad** at predicting
"source" or "target". But!! We want the domain classifier itself to be
**good** at predicting the domain. How can we get both of these goals?

## Desired training goal

Let's clarify the training goal. We train the feature extractor
and label predictor as usual. Then we add something: the domain
classifier is fighting the feature extractor. (In part, this resembles
the training regimen of Generative Adversarial Networks. Can someone
help compare and contrast?)

This is what we want to happen during training:

1. The domain classifier should get better.
2. The label predictor should get better.
3. The feature extractor should change such that [feature extractor → label predictor] gets better.
4. The feature extractor should change such that [feature extractor → domain classifier] gets **worse**.

(1) and (4) are what make the feature extractor learn features common
to both source and target distributions.

(2) and (3) are what make [feature extractor → label predictor]
predict "dog" or "cat".

If this training regimen works well, we then abandon the domain
classifier. The remaining network predicts "dog" or "cat" for images
from either source or target distributions! That would be great!

But there's a problem.

(1) and (4) are in conflict about what they're optimizing. We need to find a
clever way around that.

## Gradient Reversal Layer

<img src="images/domain-adaptation-by-backprop-title.png" />

Let’s start by defining a loss function. This combines the loss
from both the domain classifier and the label predictor:

<p> <!-- needed for correct font size in MathJax -->
$$
E(\theta_f, \theta_y, \theta_d) =
\sum_{i\in\text{source domain}} L_y^i(\theta_f, \theta_y) -
\lambda \sum_{i} L_d^i(\theta_f, \theta_d)
$$
</p>

Where:
* $$\theta_f$$, $$\theta_y$$, $$\theta_d$$ are the respective parameters
of the feature extrator, label predictor and domain classifier
* $$L_y^i$$ is the loss from the
label predictor for $$i$$th image. These values are only defined for
images from the source distribution (only those have labels.)
* $$L_d^i$$ is the loss from the domain classifier (defined
for examples from both domains.)
for training example $$i$$.
* $$\lambda>0$$ is a hyperparameter we can tweak. Setting $$\lambda$$
too low would lead to poor transfer from the source domain to the
target domain. Setting $$\lambda$$
too high would reduce classification accuracy on the source domain (and thus
on the problem at large).

We don't quite want to minimize $$E$$ -- let's see why. Let's revisit
points (1) to (4) above. What would happen during training if we try to minimize $$E$$?

1. The domain classifier should get getter. **<span style="color:red">✗</span>**
2. The label predictor should get better. **<span style="color:green">✓</span>**
3. The feature extractor should change such that [feature extractor → label predictor] gets better. **<span style="color:green">✓</span>**
4. The feature extractor should change such that [feature extractor → domain classifier] gets worse. **<span style="color:green">✓</span>**

...we... ALMOST have it. If we're optimizing $$E$$ as defined above,
the domain classifier will get worse.

Here's how we fix it: *Break the rules of
differentiation!*. Specifically, we introduce a new kind of layer. The
GRL, or *Gradient Reversal Layer*, is a psuedo-function
$$R_\lambda(x)$$ such that:
1. <!-- force inline equation -->$$R_\lambda(x) = x$$
2. <!-- force inline equation -->$$\frac{dR\lambda(x)}{dx} = -\lambda \bf{I}$$

Of course, such a function doesn't *actually* exist. But remember
how backpropagation works. Layers are defined by what they do
during feedforward and separately what they do during backprop.
The backprop function is *normally* the derivative of the feedforward function.
But we can definitely define $R_\lambda(x)$ in deep learning libraries
such as TensorFlow or Theano.

xcxc revisit here. do we need to define E~?

xcxc explain saddle point?

xcxc domain-invariant
xcxc discriminative

xcxc find a nice way to close the article

</p>
