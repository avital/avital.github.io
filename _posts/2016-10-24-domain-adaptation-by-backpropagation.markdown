---
layout: post
title: "Unsupervised Domain Adaptation by Backpropagation"
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
isn't the case*. Solutions to this problem are called **"domain adaptation"**.
I'll explain a domain adaptation
[technique proposed by Ganin and Lempitsky](https://arxiv.org/abs/1409.7495){:target="_blank"}.
For this technique, all we need is an
**unlabelled data set** from the distribution the model tests on.<!--more-->

Ganin and Lempitsky propose an awesome method for domain adaptation. Here's
what's cool about it:
1. It's simple to understand and implement.
2. It works for any feed-forward network architcture, such as ConvNets.
3. It beats state-of-the art on standard [domain adaptation
datasets](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code).

## An example

Let's say we're classifying photos as cat or dog. We send a team to
collect labelled photos. The team tries their best to capture a
diverse dataset. They come back with millions of images. Great!  We go
ahead and train a convolutional neural network on those labelled
images. Piece of cake, right? When we're done training our model, we
even get 90% accuracy on a held-out test set! We pop open a bottle of
champagne.

XXX show graph of loss and val loss

But then... we deploy our model in our fancy new app...

...and get tons of user reports about wrong predictions. Turns out,
our model is inaccurate for certain camera types. And for some kinds
of dogs and cats. And under some lighting conditions.

<img src="images/domain-adaptation-example.png" />

This might seem like an overfitting problem, and in a way it is.
But our validation loss graph seemed to show that we're not overfitting.
That's because the validation data was sampled from the training data.
What we really want is to validate on examples taken from the wild.

We go ahead and label some of the images our users took. That's
our new "real world validation set". Now, let's graph validation
accuracy on that set during training

XXX show graph

The #1 way to solve overfitting in this case would be to collect more labelled data.
But it has to be from the actual distribution seen in the wild. That might
be expensive to do.

Or, think of a case where we train our model on cheap synthetic data.
For example, using 3D scenes from a game.  A model trained this way
may not predict well on real world examples.

This is our question today: *How can we adapt the learning from one
labelled dataset (the "source" dataset) dataset, so that it works well
on an unlabelled real world dataset (the "target" dataset)?*

(Question: Does anyone know if this comes up when training
reinforcement learning models via simulators such as MuJoCo?)

## Background: A bit of statistics

First, a quick riddle. What's the difference between machine learning
and statistics?

Answer:

(Yes, that's the answer.) Machine learning is just what computer
scientists call statistics. Also, machine learning typically focuses
more on applications. Here's how a statistician would explain a classification problem
problem:

*You are given a dataset of inputs and outputs $$(x_i, y_i)$$. Find a
model that estimates the distribution $$\mathcal{P}(\bf{x}, \bf{y})$$
from which the dataset was drawn.*

(They'd probably say "covariate" and "label" instead of "input" and
"output", but you get the gist.)

What's a distribution? It's just a table of probabilities for every
possible pair of $$(x, y)$$. (I'm cheating a little here -- that's
only accurate for discrete input and output spaces, but the intuition
is the same for continuous spaces like the space of images)

XXX show a distribution

What do we mean by "estimate"? That's not completely defined.  Imagine
that our dataset has images of dogs paired with their breed.  What
should a "good estimate" predict when shown an image of a cat?
(Side note: Bayesian techniques solve this elegantly by predicting
**distributions rather than values**, so when shown a cat a model
might essentially predict a formal version of "I don't know")

One more important statistical concept is the *marginal distribution*
$$\mathcal{P}(\bf{x})$$. This is the distribution on inputs if we
forget the outputs. So, we say that an input is drawn from
$$\mathcal{P}(\bf{x})$$. (Why the name "marginal"? Because it's what
you write in the margins if you sum each rows in a table of
probabilities)

XXX show marginal distribution

(Another side note: In practice, many classification problems are
solved by minimizing "cross-entropy", also known as "logistic loss".
These measures are derived from a theory of discrepancy between
a model and a dataset called "KL divergence".)

## Covariate Shift

Imagine that we assemble two groups of people. Each group is asked to
each collect a large dataset of dog images paired with their breeds.
We find a good model for each dataset, meaning we estimate two
distributions $$\mathcal{S}(\bf{x}, \bf{y})$$ and
$$(\mathcal{T}(\bf{x}, \bf{y})$$.

How do the marginal distributions $$\mathcal{S}(\bf{x})$$ and $$\mathcal{T}(\bf{x})$$
compare? There must have **something** in common since they both contain images
of dogs. But they could be very different. For example, each group could
have used different cameras, angle positions, backgrounds, etc.

XXX show example of two datasets and how they differ

But there's one thing that's safe to assume. If we hide the labels,
and give all of the images to both groups, the groups will
agree on labels even for photos they didn't take.

We call this the *"covariate shift assumption"*. In fancy statisical
terms, the conditional distributions are equal:
$$\mathcal{S}(\bf{Y}|\bf{X}=x) = \mathcal{T}(\bf{Y}|\bf{X}=x)$$ even though the marginal
distributions differ: $$\mathcal{S}(\bf{X}) \neq \mathcal{S}(\bf{Y})$$

Statistically, we say that our training set is drawn from a **source
distribution**. And our test examples are drawn from a **target
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
Domain Adaptation by Backpropagation_. Their key insight is an augmented
neural network architecture. This archicture uses a Gradient Reversal Layer, or
"GRL". GRL is a "function" that breaks the rules
of differentiation.

They start with a neural network architecture designed for a
particular classification problem. For example, LeNet-5 for MNIST or
AlexNet for ImageNet.

XXX show LeNet-5

Then, split the network at some point and add a parallel
branch. This new branch is called the "domain classifier" and will
attempt to predict whether an image comes from the source
dataset or the target dataset.

<img src="images/domain-adaptation-mnist.png" />

This "GRL" is the Gradient Reversal Layet we alluded to above. Wait
for it...

Here's a general diagram of the approach, with helpful names
given to different parts of the network.

<img src="images/domain-adaptation-by-backprop.png" />

We train the label predictor as usual.  The kicker (and the insight
behing GRL) is how we train the domain classifier. We want the domain
classifier to try its best to distinguish the source and target
datasets. In parallel, we want the feature extractor to fool the
domain classifier into not succeeding. (In part, this resembles
the training regimen of Generative Adversarial Networks. Can someone
help compare and contrast?)

This can be described in terms of gradient descent and
backpropagataion. When applying gradient descent to this network,
gradients flow from the domain classifier to the feature extractor.
We want those gradients to encourage the feature extractor to find
representations that are *domain invariant*.

If this training regimen works well, we then abandon the domain
classifier. The remaining network predicts "dog" or "cat" for images
from either source or target distributions! That's what we want!!

## In search of an optimization objective

Let’s try to define a loss function that we'll minimize via gradient
descent.  This should combine the loss from both the domain classifier and
the label predictor:

<p> <!-- needed for correct font size in MathJax -->
$$
E(\theta_f, \theta_y, \theta_d) =
\sum_{i\in\text{source dataset}} L_y^i(\theta_f, \theta_y) -
\lambda \sum_{i\in\text{both datasets}} L_d^i(\theta_f, \theta_d)
$$
</p>

Where:
* $$\theta_f$$, $$\theta_y$$, $$\theta_d$$ are the respective parameters
of the feature extrator, label predictor and domain classifier
* $$L_y^i(\theta_f, \theta_y) = L_y(G_y(G_f(x; \theta_f); \theta_y), y_i)$$ is the loss from the
label predictor for $$i$$th image. These values are only defined for
images from the source dataset (only those have labels.)
* $$L_d^i(\theta_f, \theta_d) = L_d(G_d(G_f(x_i; \theta_f); \theta_d), y_i)$$ is the loss from the domain classifier (defined
for examples from both datasets.)
for training example $$i$$.
* $$\lambda>0$$ is a hyperparameter we can tweak. Setting $$\lambda$$
too low would lead to poor transfer from the source domain to the
target domain. Setting $$\lambda$$
too high would reduce classification accuracy on the source domain (and thus
on the problem at large). In a way, $\lambda$ acts as a regularization
parameters, controlling how much to encourage the learned model to be
domain invariant.

We don't quite want to minimize $$E$$. As defined, minimizing $$E$$
would correctly enourage the feature extractor to fool the domain
classifier. But it would not encourage the domain classifier
to get better to begin with. So we aren't playing our desired game, one that
leads to a domain invariant feature extractor.

...we... ALMOST have it.

Instead of minimizing $$E$$, what we really want to find is a
**saddle point** of $$E$$. Specifically, we want to find
$$\hat\theta_f, \hat\theta_y, \hat\theta_d$$ such that:

<p> <!-- needed for correct font size in MathJax -->
$$
\DeclareMathOperator*{\argmin}{arg\,min}
(\hat\theta_f, \hat\theta_y) = \argmin_{\theta_f, \theta_y} E(\theta_f, \theta_y, \hat\theta_d)
$$
</p>
and
<p> <!-- needed for correct font size in MathJax -->
$$
\DeclareMathOperator*{\argmax}{arg\,max}
\hat\theta_d = \argmax_{\theta_d} E(\hat\theta_f, \hat\theta_y, \theta_d)
$$
</p>

## Gradient Reversal Layer

Can we approximate a saddle point with stochastic gradient descent and
backpropagation? If we could we'd be able to easily implement this
technique in TensorFlow, Theano or any other deep learning library.

Here's how we fix it: *Break the rules of
differentiation!*. Specifically, we introduce a new kind of layer. The
GRL, or *Gradient Reversal Layer*, is a psuedo-function $$R_\lambda(x)$$ such that:

1. <!-- force inline equation -->$$R_\lambda(x) = x$$
2. <!-- force inline equation -->$$\frac{dR_\lambda(x)}{dx} = -\lambda \bf{I}$$

Of course, such a function doesn't *actually* exist. But remember
how backpropagation works. Layers can be defined by what they do
during feedforward and separately what they do during backprop.
The backprop function is *normally* the derivative of the feedforward function.
But from an implementation perspective, that's not necessary. In deep learning
libraries such as TensorFlow or Theano, layers are defined as two separate
functions.

We use the GRL "pseudo-function", $$R_\lambda(\bf{x})$$, to define a new
"loss pseudo-function":

<p> <!-- needed for correct font size in MathJax -->
$$
\tilde{E}(\theta_f, \theta_y, \theta_d) = \\
\sum_{i\in\text{source dataset}}
L_y(G_y(G_f(x_i; \theta_f); \theta_y), y_i) + \\
\sum_{i\in\text{both datasets}}  
L_d(G_d(R_\lambda(G_f(x_i; \theta_f)); \theta_d), y_i)
$$
</p>

$$\tilde{E}$$ is just like $$E$$ with one difference. When $$\theta_d$$ is
being optimized, it acts as if the loss function is just $$L_d$$. But when
$$\theta_f$$ is being optimized, it acts as if the loss function is the same
$$E$$ we defined above.

In short, when minimizing this new loss psuedo-function $$\tilde{E}$$,
1. The domain classifier gets better. **<span style="color:green">✓</span>**
2. The label predictor gets better. **<span style="color:green">✓</span>**
3. The feature extractor changes so that label predictor gets better. **<span style="color:green">✓</span>**
4. The feature extractor changes so that domain classifier gets worse. **<span style="color:green">✓</span>**

**We found the saddle point we were looking for!**

This optimization process would yield
a classifier that's *discriminative* and *domain invariant*. This solves
our domain adaptation problem. All we needed was the GRL layer to generate
the right optimization objective. And we get best-in-class results
on standard domain adaptation datasets. I think that's pretty cool.

## On the choice of target dataset

Is it really true that we can use any unlabelled dataset as our target
dataset? What if we're predicting dog or cat, and our target
dataset contains tons of examples that are neither dog or cat?

Here's what training would look like. The domain classifier would
quickly learn to predict "target dataset" for images of neither dog
nor cat. In response, the feature extractor would change to make
those images have the same features as a dog or cat image.

That could work fine, but it has a cost. The feature extractor
now has to have more capacity than it needed to for the original
classification problem. The more types of images the feature
extractor needs to learn to mask as cat or dog, the wider
and/or deeper we need the feature extractor to be. That, in turn,
would make training more costly and potentially make it harder to
avoid overfitting.

And what in the extreme example where our target dataset contains *no*
dogs or cats? This could still work -- we'd need the features learned
by the feature extractor to be enough for the label classifier to do
its job, while not being enough for the domain classifier to do *its*
job. We can make this happen by ensuring the label classifier has more
capacity than the domain classifier. Indeed, that's true in the MNIST
and SVHN architectures described in the paper. Though, curiously not
in the GTSRB architecture. I wonder why. It depends on the target
dataset used, though.
