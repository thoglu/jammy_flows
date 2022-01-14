import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import numpy as np
import numpy
import scipy.stats
import torch
import torch.optim as optim

import jammy_flows
from jammy_flows import helper_fns
import pylab
from matplotlib import rc
import random

def seed_everything(seed_no):
    random.seed(seed_no)
    numpy.random.seed(seed_no)
    torch.manual_seed(seed_no)

## Generate data that follows letter shapes using some TTF template
###################################################################

def sample_character(char, path='OpenSans-Bold.ttf', fontsize=60, width_per_cell=0.5, num_samples=1000, center_coords=(0,0), manifold_type="e"):

    """
    Based on https://stackoverflow.com/a/27753869/190597 (jsheperd)
    """

    font = ImageFont.truetype(path, fontsize) 
    w, h = font.getsize(char)  
    h *= 2
    image = Image.new('L', (w, h), 1)  
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), char, font=font) 
    arr = np.asarray(image)
    arr = np.where(arr, 0, 1)
    arr = arr[(arr != 0).any(axis=1)]

    one_mask=arr.T==1
    num_x_cells=one_mask.shape[0]
    num_y_cells=one_mask.shape[1]

 
    ## discretized random sampling that follows letter shape
    xvals, yvals=np.meshgrid(np.arange(one_mask.shape[0]), np.arange(one_mask.shape[1]))

    xvals=xvals.T.astype('float64')
    yvals=yvals.T.astype('float64')

    xvals-=num_x_cells//2
    yvals-=num_y_cells//2

    # add some extra noise
    xvals+=np.random.normal(size=xvals.shape)
    yvals+=np.random.normal(size=yvals.shape)

    xvals*=width_per_cell
    yvals*=width_per_cell*(-1.0) ## have to flip y 

    one_coords=np.hstack([xvals[one_mask][:,None], yvals[one_mask][:,None]])
    
    sample_indices=np.random.choice(len(one_coords), num_samples)
    samples=one_coords[sample_indices]

    samples[:,0]+=center_coords[0]
    samples[:,1]+=center_coords[1]

    ## scale azimuth to make it similar to zenith
    if(manifold_type=="s"):
        azi_diff=(samples[:,1]-numpy.pi)
        samples[:,1]=numpy.pi+azi_diff*2
      

    return samples


## this function generates train and test data
def sample_data(pdf_def, sentence, num_samples=10000):

    words=sentence.split(" ")

    num_words=len(words)

    last_len=len(words[0])
    for w in words:
        if(len(w)!=last_len):
            raise Exception("All words in sentence must be of same length")

    ## every char takes 2 dimensions
    manifold_str=""
    len_per_word=0
    pdf_dim=0

    for pdf in pdf_def.split("+"):
        
        if(int(pdf[1:])%2!=0):
            raise Exception("Characters take 2 dimensions, so string is visualized with 2*len(str) dims. Every PDF must have a dimension divisible by 2 for simplicity.")

        len_per_word=int(pdf[1:])//2
        pdf_dim+=int(pdf[1:])

        if("e" in pdf):
            manifold_str+=len_per_word*"e"
        elif("s" in pdf):
            manifold_str+=len_per_word*"s"

    word_indices=np.random.choice(num_words, num_samples)
  
    _, class_occurences = np.unique(word_indices, return_counts=True)

    labels=torch.randn( (num_samples, pdf_dim)).type(torch.float64)
  
    ## loop words
    for w_index, w in enumerate(words):

        this_w_sample=[]
        ## loop char per word
        for c_index, c in enumerate(w):
           
            center=(0,0)
            stretch=0.5

            ## if sphere, center character at equator
            if(manifold_str[c_index]=="s"):
                center=(np.pi/2.0, np.pi)
                stretch=0.05
            res=sample_character(c, num_samples=class_occurences[w_index], width_per_cell=stretch, center_coords=center, manifold_type=manifold_str[c_index])
            
            if(manifold_str[c_index]=="s"):
                assert( ((res[:,0]<0) | (res[:,0]>np.pi)).sum()==0)
                assert( ((res[:,1]<0) | (res[:,1]>2*np.pi)).sum()==0)
            this_w_sample.append(torch.from_numpy(res))

        tot_sample=torch.cat(this_w_sample, dim=1)
        labels[word_indices==w_index]=tot_sample


    onehot_input = torch.nn.functional.one_hot(torch.from_numpy(word_indices), num_words).type(torch.float64)

    return onehot_input, labels

#######################################################################

## plot the model during training
def plot_test(test_data, test_labels, model, words, fname="figs/test.png"):

    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    num_words=len(torch.unique(test_data, dim=0))

    fig=pylab.figure(figsize=((num_words+1)*4, 4)) 

    gridspec=fig.add_gridspec(1, num_words+1)

    word_ids=torch.nn.functional.one_hot(torch.arange(num_words), num_words).type(torch.float64)

    ## 2 * log_pdf differences
    pdf_res, base_pdf_res, _=model(test_labels)#, conditional_input=test_data)

    dim=test_labels.shape[1]

    glob_dim_index=0
    bounds=[]
    bmin=9999
    bmax=-9999
    mask=[]
    for pdf_str in model.pdf_defs_list:
        this_dim=int(pdf_str[1:])
        this_type=pdf_str[0]
        if(this_type=="e"):

            for ind in range(this_dim):
                this_min=test_labels.detach().numpy()[:,glob_dim_index].min()
                this_max=test_labels.detach().numpy()[:,glob_dim_index].max()

                if(this_min<bmin):
                    bmin=this_min

                if(this_max>bmax):
                    bmax=this_max

                glob_dim_index+=1

        else:
            glob_dim_index+=2
            continue
    
   
    sphere_plot_type="standard"
    for pdf_str in model.pdf_defs_list:
        this_dim=int(pdf_str[1:])
        this_type=pdf_str[0]

        if(this_type=="s"):

            if(sphere_plot_type=="standard"):
                bounds.append([0,np.pi])
                bounds.append([0,2*np.pi])
            else:
                bounds.append([-2,2])
                bounds.append([-2,2])

            glob_dim_index+=2

        else:

            for ind in range(this_dim):
                bounds.append([bmin,bmax])
    
    logpz_max= scipy.stats.multivariate_normal.logpdf( dim*[0], mean=dim*[0])
    twice_pdf_diff=2*(logpz_max - base_pdf_res)

    coverage_probs=np.linspace(0.01,0.99,100)
    true_twice_llhs=scipy.stats.chi2.ppf(coverage_probs, df=dim)
    ## plot PDF for individual "word input data"

    colors=pylab.cm.tab10.colors

    cov_ax=fig.add_subplot(gridspec[0,num_words])

    for word_index, wid in enumerate(word_ids):

        helper_fns.visualize_pdf(model, fig, gridspec=gridspec[0,word_index], conditional_input=None, total_pdf_eval_pts=2000, nsamples=10000, contour_probs=[], hide_labels=True,bounds=bounds,s2_norm=sphere_plot_type)
    
        ## plot coverage
        this_coverage=twice_pdf_diff[(wid[word_index]==test_data[:,word_index])]
        
        act_cov=[]
        for ind,true_cov in enumerate(coverage_probs):
            act_cov.append(sum(this_coverage<true_twice_llhs[ind])/float(len(this_coverage)))

        cov_ax.plot(coverage_probs, act_cov, label=r"$p(x|'%s')$" % words[word_index], color=colors[word_index])

    cov_ax.plot([0.0,1.0],[0.0,1.0], color="k", lw=2.0, ls="--")
    cov_ax.set_xlim(0,1)
    cov_ax.set_ylim(0,1)
    cov_ax.grid(True)
    cov_ax.legend(loc="upper right")
    cov_ax.set_title("Coverage")

    fig.suptitle("pdf structure: %s" % "+".join(model.pdf_defs_list))
    fig.tight_layout()

    fig.savefig(fname)

    pylab.close(fig)

    #test_evals, standard_normal_base_evals, _=model(test_labels, conditional_input=test_data)


############################

if __name__ == "__main__":

    parser = argparse.ArgumentParser('train_example')

    parser.add_argument("-sentence", type=str, default="JAMMY FLOWS")
    parser.add_argument("-pdf_def", type=str, default="e4+s2+e4")
    parser.add_argument("-layer_def", type=str, default="gggg+n+gggg") 
    parser.add_argument("-train_size", type=int, default=200000)
    parser.add_argument("-batch_size", type=int, default=20)
    parser.add_argument("-test_size", type=int, default=1000)
    parser.add_argument("-lr", type=float, default=0.001)

    args=parser.parse_args()

    seed_everything(1)

    assert(args.train_size % args.batch_size==0)

    ## train data used for training
    train_data, train_labels=sample_data(args.pdf_def, args.sentence, num_samples=args.train_size)

    ## test used to calculate coverage
    test_data, test_labels=sample_data(args.pdf_def, args.sentence, num_samples=args.test_size)

    extra_flow_defs=dict()
    extra_flow_defs["n"]=dict()
    extra_flow_defs["n"]["kwargs"]=dict()
    extra_flow_defs["n"]["kwargs"]["zenith_type_layers"]="g"
    extra_flow_defs["n"]["kwargs"]["use_extra_householder"]=0

    extra_flow_defs["g"]=dict()
    extra_flow_defs["g"]["kwargs"]=dict()
    extra_flow_defs["g"]["kwargs"]["regulate_normalization"]=1
    extra_flow_defs["g"]["kwargs"]["fit_normalization"]=1
    extra_flow_defs["g"]["kwargs"]["add_skewness"]=0

    word_pdf=jammy_flows.pdf(args.pdf_def, args.layer_def, conditional_input_dim=None, hidden_mlp_dims_sub_pdfs="128",flow_defs_detail=extra_flow_defs, use_custom_low_rank_mlps=False,
        custom_mlp_highway_mode=4)

    word_pdf.count_parameters(verbose=True)
    ## initalize params with test sample (only advantage gains for Gaussianization flows)
    word_pdf.init_params(data=test_labels)

    ## start training loop
    num_batches=args.train_size//args.batch_size
    num_epochs=300
    plot_every_n=200
    glob_counter=0

    cur_lr=args.lr
    for ep_id in range(num_epochs):

        optimizer = optim.Adam(word_pdf.parameters(), lr=cur_lr)

        for batch_id in range(num_batches):

            ## get new batch
            batch_data, batch_labels=train_data[batch_id*args.batch_size:batch_id*args.batch_size+args.batch_size], train_labels[batch_id*args.batch_size:batch_id*args.batch_size+args.batch_size]

            ## reset accumulated grad
            optimizer.zero_grad()

            ## evaluate PDF
            log_pdf, _,_=word_pdf(batch_labels)#, conditional_input=batch_data)
           
            ## neg log-loss
            loss=-log_pdf.mean()

            print("loss ", loss)

            ## backprop
            loss.backward()

            ## take a gradient step
            optimizer.step()

            ## plot test data
            if(glob_counter%plot_every_n==0):


                with torch.no_grad():
                    print("VALIDATION EVAL")    
                    val_log_pdf, _, _=word_pdf(test_labels)#, conditional_input=test_data)
                    val_loss=-val_log_pdf.mean()
                    print("ep: %d / batch_id: %d / val-loss %.3f" % (ep_id, batch_id, val_loss))
                    print("before plotting")
                    print("----------------------------->")
                    plot_test(test_data, test_labels, word_pdf, args.sentence.split(" "), fname="./figs/%.6d.png" % glob_counter)

            glob_counter+=1

        cur_lr*=0.9