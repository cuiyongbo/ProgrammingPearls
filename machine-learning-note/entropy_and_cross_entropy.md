Entropy
=======

Definition: In information theory, entropy is the average rate
of information produced from a certain stochastic process.

.. math::

    H(X) = -\sum_{i=1}^{n} {P(x_i) \log P(x_i)}


KL divergence
=============

The KL divergence between two distributions has many different interpretations
from an information theoretic perspective. It is also, in simplified terms,
an expression of “surprise” – under the assumption that P and Q are close,
it is surprising if it turns out that they are not, hence in those cases
the KL divergence will be high. If they are close together,
then the KL divergence will be low.

Here we interpret the KL divergence to be something like the following –
if P is the “true” distribution, then the KL divergence is the amount of
information “lost” when expressing it via Q.

.. math::

    D_{KL}(P || Q) = \sum_{i}^{n}{P(x_i) \log P(x_i)} - \sum_{i}^{n}{P(x_i) \log Q(x_i)}


Cross Entropy
=============

Definition: a widely used metric of similarity between two probability distributions.

Cross entropy is, at its core, a way of measuring the “distance” between two probability
distributions P and Q. As you observed, entropy on its own is just a measure of a single
probability distribution. As such, if we are trying to find a way to model a true probability
distribution, P, using, say, a neural network to produce an approximate probability distribution Q,
then there is the need for some sort of distance or difference measure which can be minimized.

.. math::

    H(p, q) = H(p) + D_{KL}(p || q)


.. rubric:: Footnotes

.. [#] `Introduction to cross entropy <https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/>`_
.. [#] `Introduction to cross entropy 2 <https://adventuresinmachinelearning.com/cross-entropy-kl-divergence/>`_
