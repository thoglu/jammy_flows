import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pylab

import numpy
import scipy
import scipy.special
import torch

import jammy_flows

flow_kwargs=dict()
flow_kwargs["g"]=dict()
flow_kwargs["g"]["center_mean"]=1

p_single_nocenter=jammy_flows.pdf("e1", "g")
single_samples_nocenter,_,_,_=p_single_nocenter.sample(samplesize=1000000)

p_single=jammy_flows.pdf("e1", "g", options_overwrite=flow_kwargs)
single_samples,_,_,_=p_single.sample(samplesize=1000000)

p_multiple_nocenter=jammy_flows.pdf("e1", "gg")
multi_samples_nocenter,_,_,_=p_multiple_nocenter.sample(samplesize=1000000)

p_multiple=jammy_flows.pdf("e1", "gg", options_overwrite=flow_kwargs)
multi_samples,_,_,_=p_multiple.sample(samplesize=1000000)

single_samples_nocenter=single_samples_nocenter[:,0].detach().numpy()
single_samples=single_samples[:,0].detach().numpy()

multi_samples_nocenter=multi_samples_nocenter[:,0].detach().numpy()
multi_samples=multi_samples[:,0].detach().numpy()

fig=pylab.figure()

ax=pylab.gca()

ax.hist(single_samples_nocenter, bins=50, histtype="step",label="single noc %.3e" % (numpy.mean(single_samples_nocenter)))

ax.hist(single_samples, bins=50,histtype="step",label="single %.3e" % (numpy.mean(single_samples)))
ax.hist(multi_samples, bins=50,histtype="step",label="multi %.3e" % (numpy.mean(multi_samples)))
ax.hist(multi_samples_nocenter, bins=50,histtype="step",label="multi nocenter %.3e" % (numpy.mean(multi_samples)))

ax.legend()

fig.tight_layout()
pylab.savefig("center_mean_test.png")