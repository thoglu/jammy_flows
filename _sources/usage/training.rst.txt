**************************
Training
**************************


--------------------------------
non-conditional PDF
--------------------------------

Lets say we have Euclidean 3-d samples "x" from a 3-d target distribution we want to approximate with a PDF p(x).
First we initalize a 3-d PDF:

..  code-block:: python

    import jammy_flows

    pdf=jammy_flows.pdf("e3", "gggg")

We chose to do so with 4 Gaussianization flow layers. Next we just loop through the batches and minimize negative log-probability, assuming we have a torch optimizer:

..  code-block:: python

    # target_samples = array of 3-d samples from the target distribution
    # batch_size: the given batch size
    # num_batches_per_epoch: number of batches in the epoch
    # optimizer: the given pytorch optimizer

    batch_size=10

    for ind in range(num_batches_per_epoch):
        this_label_batch=target_samples[ind:ind+batch_size]

        optimizer.zero_grad()

        log_pdf,_,_=pdf(this_label_batch)

        neg_log_loss=-log_pdf.mean()

        neg_log_loss.backward()

        optimizer.step()

    ### after training ###

    ## evaluation
    # target_point = point "x" to evaluate the PDF at
    # log_pdf, base_log_pdf, base_point = pdf(target_point)

    ## sampling
    # target_sample, base_sample, target_log_pdf, base_log_pdf = pdf.sample(samplesize=1000)


--------------------------------
conditional PDF
--------------------------------

Let's now say we want to describe a 3-d conditional PDF p(x;y) that depends on some 2-dimensional input 'y'.

..  code-block:: python

    import jammy_flows

    pdf=jammy_flows.pdf("e3", "gggg", conditional_input_dim=2)


Here we have to specify the conditional input dimension, and then in the training loop add the input.
The rest is very similar to the non-conditional PDF:

..  code-block:: python

    # target_samples = array of 3-d samples from the target distribution
    # input_data = array of 2-d input data of same length as target_samples
    # batch_size: the given batch size
    # num_batches_per_epoch: number of batches in the epoch
    # optimizer: the given pytorch optimizer

    for ind in range(num_batches_per_epoch):
        this_label_batch=target_samples[ind:ind+batch_size]
        this_data_batch=input_data[ind:ind+batch_size]

        optimizer.zero_grad()

        log_pdf,_,_=pdf(this_label_batch, conditional_input=this_data_batch)

        neg_log_loss=-log_pdf.mean()

        neg_log_loss.backward()

        optimizer.step()

    ### after training ###

    ## evaluation
    # target_point = point "x" to evaluate the PDF at
    # log_pdf, base_log_pdf, base_point = pdf(target_point, conditional_input=some_input)

    ## sampling, shape of 'some_input' defines number of samples in conditional pdf
    # target_sample, base_sample, target_log_pdf, base_log_pdf = pdf.sample(conditional_input=some_input) 
