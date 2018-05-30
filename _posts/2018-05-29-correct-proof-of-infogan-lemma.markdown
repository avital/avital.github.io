---
layout: post
comments: true
title: "A correct proof of a lemma from the InfoGAN paper"
date: 2018-05-29 13:53:58
excerpt_separator: <!--more-->
---


The [InfoGAN paper](https://arxiv.org/pdf/1606.03657.pdf) has the following lemma:

**Lemma 5.1**
For random variables $$X, Y$$ and function $$f(x, y)$$ under suitable regularity conditions:
$$\mathbb{E}_{x \sim X, y \sim Y|x}[f(x, y)] = 
 \mathbb{E}_{x \sim X, y \sim Y|x, x' \sim X|y}[f(x, y)]$$.

The proof in the paper seems wrong -- here's a step where $$x$$ mysteriously becomes $$x'$$:

<!--more-->

$$
\begin{align*}
& \int_x \int_y P(x,y) {\color{red}{f(x,y)}} \int_{x'} P(x' | y)dx'dydx \\
= & \int_x P(x) \int_y P(y|x) \int_{x'} P(x'|y) {\color{red}{f(x',y)}} dx' dy dx
\end{align*}
$$

After consulting with others, we couldn't fix this proof. Instead, [Nic Ford](http://nicf.net) found the following proof:

$$
\begin{align*}
   & \mathbb{E}_{x \sim X,y \sim Y|x}[f(x, y)] = & \mbox{make expectations explicit...} \\
   & \mathbb{E}_{x \sim P(X)}\big[\mathbb{E}_{y \sim P(Y|X=x)}[f(x, y)]\big] = & \mbox{by definition of $P(Y|X=x)$...} \\
   & \mathbb{E}_{x,y \sim P(X,Y)}[f(x, y)] = & \mbox{by definition of $P(X|Y=y)$... ...} \\
   & \mathbb{E}_{y \sim P(Y)}\big[\mathbb{E}_{x \sim P(X|Y=y)}[f(x, y)]\big] = & \mbox{rename $x$ to $x'$...} \\
   & \mathbb{E}_{y \sim P(Y)}\big[\mathbb{E}_{x' \sim P(X|Y=y)}[f(x', y)]\big] = & \mbox{by the law of total expectation...} \\
   & \mathbb{E}_{x \sim P(X)}\Big[\mathbb{E}_{y \sim P(Y|X=x)}\big[\mathbb{E}_{x' \sim P(X|Y=y)}[f(x', y)]\big]\Big] = &  \mbox{make expectations implicit...} \\
   & \mathbb{E}_{x \sim X,y \sim Y|x,x' \sim X|y}[f(x', y)] & \\
\end{align*}
$$


(This note is also available as a [PDF](/assets/correct-proof-of-infogan-lemma.pdf).)






