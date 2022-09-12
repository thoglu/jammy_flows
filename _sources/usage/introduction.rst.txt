**************************
Introduction
**************************

Jammy Flows (**J** oint **A** utoregressive **M** anifold ( **MY** ) Normalizing **Flows**) implements normalizing flow PDFs that can be defined to describe joint distributions over an arbitrary number of multiple manifolds and grew out of work described in .


The manifolds are connected via (inverse) autoregressive connections, where the connectivity is akin to the one described in the paper *Improving Variational Inference with Inverse Autoregressive Flow* (*https://arxiv.org/abs/1606.04934*). There are two major differences to the IAF implementation. The first difference is that the autoregressive connections in *jammy_flows* are explicitly linking different manifold flows, not each individual dimension. This allows to describe distributions defined jointly over multiple manifolds, e.g. a Euclidean manifold and a sphere. The second difference is that the coupling layers are not affine (i.e. linear couplings that describe Gaussian distributions), but general nonlinear couplings that can describe any normalizing flow distribution. As such, the autoregressive connections map to any desired normalizing flow implemented in the package.

The autoregressive routing is automatically handled in the background, and the user can get started to define a PDF with a simple syntax.

----------------
Initialization
----------------

For example, to describe a PDF over a four dimensional space, consisting of 2 Euclidean dimensions and a 2-sphere, one could write

..  code-block:: python

    import jammy_flows
     
    flow_pdf=jammy_flows.pdf("e2+s2", "ggg+n")


The first argument ``"e2+s2"`` defines the manifold structure, e.g. a 2-d Euclidean PDF (``e2``), autoregressively linked to a 2-sphere PDF (``s2``). The second argument defines the individual flow layers for each manifold. In this case there are three Gaussianization flow layers (``ggg``), which means the flow function for the Euclidean part is a composite function of three Gaussianization Flow layers as :math:`f_{tot}(x)=f_{g,1}(f_{g,2}(f_{g,3}(x)))`. The spherical part is a segmented-sphere flow, which is applied once and abbreviated with ``n``.
A list of supported manifolds and respective flow abbreviations for each manifold is given in the API.

Since Autoregressive structure imposes an ordering, the above example is slightly different than

..  code-block:: python

    import jammy_flows
     
    flow_pdf=jammy_flows.pdf("s2+e2", "n+ggg")


----------------
Evaluation
----------------
Once a PDF is defined, one can evaluate it. Let's look at the first example again:

To evaluate, one can just call it like

..  code-block:: python

    import torch
    import jammy_flows

    flow_pdf=jammy_flows.pdf("e2+s2", "ggg+n")

    # by default the pdf is in double precision, so we feed it a double tensor
    # the last two coordinates are spherical coordinates between (0,pi) and (0,2pi), respectively
    target=torch.DoubleTensor([2.4, 2.0, 0.0, 2*pi])


    # evaluating at target gives 3 return values
    log_prob_target, log_prob_base, value_base=flow_pdf(target)

The first return value is the log-probability at the target position. The second is the log-probability at the base distribution.
The third is the position in the base space.

This evaluation will actually treat the spherical dimension as not being a manifold, i.e. not embedded in some higher dimensional space. Normally, it is desired to evaluate it as a true spherical PDF.

..  code-block:: python
    
    # the same target expressed with the spherical coordinates now in embedding space
    target=torch.DoubleTensor([2.4, 2.0, 0.0, 0.0, 1.0])

    # evaluating at embedding coordinates
    log_prob_target, log_prob_base, value_base=flow_pdf(target, force_embedding_coordinates=True)

The log-probability obtained will now contain the extra log-det factor to properly account for the curvature of the sphere.

* Evaluation
----------------
Sampling
----------------





All layers inherit from the base class 
