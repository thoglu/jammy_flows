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

def sample_character(char, path='OpenSans-Bold.ttf', fontsize=60, width_per_cell=(0.5,0.5), num_samples=1000, center_coords=(0,0), manifold_type="e"):

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

    xvals*=width_per_cell[0]
    yvals*=width_per_cell[1]*(-1.0) ## have to flip y 


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


def get_center_and_stretch(manifold_str, c_index):

    

    ## if sphere, center character at equator
    if(manifold_str[c_index]=="e"):
        center=0.0
        stretch=0.5
    elif(manifold_str[c_index]=="s"):
        center=np.pi/2.0
        stretch=0.05
    elif(manifold_str[c_index]=="i"):
        center = 0.5
        stretch = 0.015
    elif(manifold_str[c_index]=="c"): 
        center=0.25
        stretch=0.005
    else:
        raise Exception("Unsupported manifold", manifold_str[c_index])

    return center, stretch

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
        
        #if(int(pdf[1:])%2!=0):
        #    raise Exception("Characters take 2 dimensions, so string is visualized with 2*len(str) dims. Every PDF must have a dimension divisible by 2 for simplicity.")

        #len_per_word=int(pdf[1:])//2
        cur_pdf_dim=int(pdf[1:])
        pdf_dim+=cur_pdf_dim

        if("e" in pdf):
            manifold_str+=cur_pdf_dim*"e"
        elif("s" in pdf):
            manifold_str+=cur_pdf_dim*"s"
        elif("i" in pdf):
            manifold_str+="i"
        elif("c" in pdf):
            manifold_str+="c"*cur_pdf_dim
        else:
            raise Exception("Unsupported manifold ", pdf)

    assert(len(manifold_str)%2==0)
    
    word_indices=np.random.choice(num_words, num_samples)
  
    _, class_occurences = np.unique(word_indices, return_counts=True)

    labels=torch.randn( (num_samples, pdf_dim)).type(torch.float64)
  
    ## loop words
    for w_index, w in enumerate(words):

        this_w_sample=[]
        ## loop char per word
        for c_index, c in enumerate(w):
            
            first_center, first_stretch=get_center_and_stretch(manifold_str, c_index*2)
            sec_center, sec_stretch=get_center_and_stretch(manifold_str, c_index*2+1)

            ## fix second coordinate on sphere (azimuth) to be centered around pi instead of pi/2
            if(manifold_str[c_index*2+1]=="s"):
                sec_center+=numpy.pi/2.0
            
            res=sample_character(c, num_samples=class_occurences[w_index], width_per_cell=(first_stretch, sec_stretch), center_coords=(first_center, sec_center), manifold_type=manifold_str[c_index*2+1])
            
            if(manifold_str[c_index*2+1]=="s"):
                assert( ((res[:,0]<0) | (res[:,0]>np.pi)).sum()==0), res
                assert( ((res[:,1]<0) | (res[:,1]>2*np.pi)).sum()==0), ("min ", res[:,1].min(), "max ", res[:,1].max())
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

    cinput=test_data
    if(model.conditional_input_dim is None):
        cinput=None
    pdf_res, base_pdf_res, _=model(test_labels, conditional_input=cinput)

    dim=test_labels.shape[1]

    glob_dim_index=0
    bounds=[]
    bmin=9999
    bmax=-9999
    mask=[]
 
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

        elif(this_type=="e"):
            for ind in range(this_dim):
                bounds.append([-12.0,12.0])
          
        elif(this_type=="i"):
            bounds.append([0.0,1.0])
        elif(this_type=="c"):
            for ind in range(this_dim):
                bounds.append([0.0001,0.9999])
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

        cinput=wid.unsqueeze(0)

        if(model.conditional_input_dim is None):
            cinput=None

        _,_,pdf_integral=helper_fns.visualize_pdf(model, fig, gridspec=gridspec[0,word_index], conditional_input=cinput.to(test_data), total_pdf_eval_pts=10000, nsamples=10000, contour_probs=[], hide_labels=True,bounds=bounds,s2_norm=sphere_plot_type)
    
        ## plot coverage
        this_coverage=twice_pdf_diff[(wid[word_index]==test_data[:,word_index])]
     
        act_cov=[]
        for ind,true_cov in enumerate(coverage_probs):
            
            act_cov.append(float(sum(this_coverage<true_twice_llhs[ind]).cpu())/float(len(this_coverage)))
           
        cov_ax.plot(coverage_probs, act_cov, label=r"$p(x|'%s')$ (integral: %.2f)" % (words[word_index],pdf_integral), color=colors[word_index])

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


############################

if __name__ == "__main__":

    parser = argparse.ArgumentParser('train_example')

    parser.add_argument("-sentence", type=str, default="JAMMY FLOWS")
    parser.add_argument("-pdf_def", type=str, default="e10")
    parser.add_argument("-layer_def", type=str, default="ggggg") 
    parser.add_argument("-use_conditional_pdf", type=int, default=1) 
    parser.add_argument("-train_size", type=int, default=200000)
    parser.add_argument("-batch_size", type=int, default=20)
    parser.add_argument("-test_size", type=int, default=500)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-plot_every_n", type=int, default=200)
    parser.add_argument("-fully_amortized_pdf", type=int, default=0)
    parser.add_argument("-gpu", type=int, default=0)

    args=parser.parse_args()

    if(args.gpu):
        assert(torch.cuda.is_available)
    gpu_device=torch.device("cuda:0")

    seed_everything(1)

    assert(args.train_size % args.batch_size==0)

    if(args.use_conditional_pdf==0):
        assert(len(args.sentence.split(" "))==1), "Using no conditional pdf means we can only have one word, but more than one word in sentence!"

    ## train data used for training
    train_data, train_labels=sample_data(args.pdf_def, args.sentence, num_samples=args.train_size)

    ## test used to calculate coverage
    test_data, test_labels=sample_data(args.pdf_def, args.sentence, num_samples=args.test_size)
    if(args.gpu):
        test_data=test_data.to(gpu_device)
        test_labels=test_labels.to(gpu_device)

    extra_flow_defs=dict()
    extra_flow_defs["n"]=dict()

    extra_flow_defs["g"]=dict()
    extra_flow_defs["g"]["nonlinear_stretch_type"]="classic" 



    cinput=len(args.sentence.split(" "))
    if(args.use_conditional_pdf==0):
        cinput=None

    if(args.fully_amortized_pdf):
        word_pdf=jammy_flows.fully_amortized_pdf(args.pdf_def, 
                                                 args.layer_def, 
                                                 conditional_input_dim=cinput, 
                                                 options_overwrite=extra_flow_defs, 
                                                 amortization_mlp_use_costom_mode=False,
                                                 amortization_mlp_dims="128",
                                                 amortization_mlp_highway_mode=0)
    else:
        word_pdf=jammy_flows.pdf(args.pdf_def, 
                                 args.layer_def, 
                                 conditional_input_dim=cinput, 
                                 hidden_mlp_dims_sub_pdfs="128",
                                 options_overwrite=extra_flow_defs, 
                                 use_custom_low_rank_mlps=True,
                                 custom_mlp_highway_mode=0)

    word_pdf.count_parameters(verbose=True)
    ## initalize params with test sample (only advantage gains for Gaussianization flows)
    word_pdf.init_params(data=test_labels)

    
    if(args.gpu):
        word_pdf.to(gpu_device)

    ## start training loop
    num_batches=args.train_size//args.batch_size
    num_epochs=300
    #plot_every_n=200
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
            cinput=batch_data
            if(args.use_conditional_pdf==0):
                cinput=None

            if(args.gpu):
                batch_labels=batch_labels.to(gpu_device)
                if(cinput is not None):
                    cinput=cinput.to(gpu_device)

            log_pdf, _,_=word_pdf(batch_labels, conditional_input=cinput)
           
            ## neg log-loss
            loss=-log_pdf.mean()

            print("loss ", loss)

            ## backprop
            loss.backward()

            ## take a gradient step
            optimizer.step()

            ## plot test data
            if(glob_counter%args.plot_every_n==0):


                with torch.no_grad():
                    print("VALIDATION EVAL")
                    
                    test_cinput=test_data
                    if(args.use_conditional_pdf==0):
                        test_cinput=None



                    val_log_pdf, _, _=word_pdf(test_labels, conditional_input=test_cinput)
                    val_loss=-val_log_pdf.mean()
                    print("ep: %d / batch_id: %d / val-loss %.3f" % (ep_id, batch_id, val_loss))
                    print("before plotting")
                    print("----------------------------->")
                    plot_test(test_data, test_labels, word_pdf, args.sentence.split(" "), fname="./figs/%.6d.png" % glob_counter)

            glob_counter+=1

        cur_lr*=0.9