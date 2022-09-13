**************************
Introduction
**************************

Jammy Flows (**J** oint **A** utoregressive **M** anifold ( **MY** ) Normalizing **Flows**) implements normalizing flow PDFs that can be defined to describe joint distributions over an arbitrary number of multiple manifolds and grew out of work described in *Unifying supervised learning and VAEs - automating statistical inference in (astro-)particle physics with amortized conditional normalizing flows* (*https://arxiv.org/abs/2008.05825*).
.
The manifolds are connected via (inverse) autoregressive connections, where the connectivity is akin to the one described in the paper *Improving Variational Inference with Inverse Autoregressive Flow* (*https://arxiv.org/abs/1606.04934*). There are two major differences to the IAF implementation. The first difference is that the autoregressive connections in *jammy_flows* are explicitly linking different manifold flows, not each individual dimension. This allows to describe distributions defined jointly over multiple manifolds, e.g. a Euclidean manifold and a sphere. The second difference is that the coupling layers are not affine (i.e. linear couplings that describe Gaussian distributions), but general nonlinear couplings that can describe any normalizing flow distribution. As such, the autoregressive connections map to any desired normalizing flow implemented in the package.

The autoregressive routing is automatically handled in the background, and the user can get started to define a PDF with a simple syntax.

There are currently 4 different manifolds supported by *jammy_flows*. 

    * **Euclidean ("e")**
    * **spherical ("s")**
    * **interval ("i")**
    * **simplex ("a")**

**The abbreviations are important, because they are used by *jammy_flows* to construct a tensor product of manifolds on which the PDF will live**. For each manifold, there are manifold-specific normalizing flows defined on them. Each of those flows also is abbreviated by its own letter. For a list of flows and respective letter abbreviations, have a look in the API documentation.

Now lets see how to use this abbreviation logic in practice.

----------------
Initialization
----------------


For example, to describe a PDF over a four dimensional space, consisting of 2 Euclidean dimensions and a 2-sphere, one could write

..  code-block:: python

    import jammy_flows
     
    flow_pdf=jammy_flows.pdf("e2+s2", "ggg+n")


The first argument ``"e2+s2"`` defines the manifold structure, e.g. a 2-d Euclidean PDF (``e2``), autoregressively linked to a 2-sphere PDF (``s2``). The manifolds are separated by a ``"+"``. The second argument defines the individual flow layers for each manifold. In this case there are three Gaussianization flow layers (``ggg``), which means the flow function for the Euclidean part is a composite function of three Gaussianization Flow layers as :math:`f_{tot}(x)=f_{g,1}(f_{g,2}(f_{g,3}(x)))`. The spherical part is a segmented-sphere flow, which is applied once and abbreviated with ``n``. Again, the flow layer definitions for each manifold are separated by a ``"+"``.
A list of supported manifolds and respective flow abbreviations for each manifold is given in the API.

Since autoregressive structure imposes an ordering, the above example is different than

..  code-block:: python

    import jammy_flows
     
    flow_pdf=jammy_flows.pdf("s2+e2", "n+ggg")

and you have to feed the PDF input tensors that follow the ordering.

The definition of interval PDFs is slightly different, in that the interval is directly defined within the PDF definition by appending the bounds within the first argument. If no bounds are given, the range of the interval PDF will be from 0 to 1.

..  code-block:: python
    :caption: Interval flows can use slightly different definition in the first argument to directly set the interval bounds.

    import jammy_flows
     
    flow_pdf=jammy_flows.pdf("i1_-3.0_3.0+i1", "r+rrr")


The PDF above will be a 2-d PDF defined over the interval -3 to 3 on the x-axis and 0 to 1 on the y-axis. The first interval on -3 to 3 uses one rational-quadratic spline transformation, the second flow on the interval 0 to 1 (default) uses three rational-quadratic spline transformations.

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
    log_prob_target, log_prob_base, position_base=flow_pdf(target)

The first return value is the log-probability at the target position. The second is the log-probability at the base distribution.
The third is the position in the base space.

This evaluation will actually treat the spherical dimension as not being a manifold, i.e. not embedded in some higher dimensional space. Normally, it is desired to evaluate it as a true spherical PDF.

..  code-block:: python
    
    # the same target expressed with the spherical coordinates now in embedding space
    target=torch.DoubleTensor([2.4, 2.0, 0.0, 0.0, 1.0])

    # evaluating at embedding coordinates
    log_prob_target, log_prob_base, position_base=flow_pdf(target, force_embedding_coordinates=True)

The log-probability obtained will now contain the extra log-det factor to properly account for the curvature of the sphere.

----------------
Sampling
----------------
