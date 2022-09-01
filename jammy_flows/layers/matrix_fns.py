import torch
import numpy

def obtain_lower_triangular_matrix_and_logdet(dimension, single_log_diagonal_entry=None, log_diagonal_entries=None, lower_triangular_entries=None, cov_type="full", upper_triangular=False):
    """
    Obtain lower trinagular (cholesky) matrix. Diagonal entries have to be positive
    """
    ## all variances are the same
    if(cov_type=="identity"):

        return torch.eye(dimension, dtype=torch.float64).unsqueeze(0), 0.0

    elif(cov_type=="diagonal_symmetric"):
        assert(single_log_diagonal_entry is not None)
        assert(single_log_diagonal_entry.shape[1]==1)

        full_diagonal=single_log_diagonal_entry.repeat(1, dimension)

        return torch.diag_embed(full_diagonal.exp()), dimension*single_log_diagonal_entry.sum(axis=-1)

    elif(cov_type=="diagonal"):
        assert(log_diagonal_entries is not None)
        assert(log_diagonal_entries.shape[1]==dimension)

        return torch.diag_embed(torch.exp(log_diagonal_entries)), log_diagonal_entries.sum(axis=-1)

    elif(cov_type=="full"):

        assert(log_diagonal_entries is not None), "Require log-diagonal entries for full triangular description."

        assert(lower_triangular_entries.shape[1]==int(dimension*(dimension-1)/2) )

        cum_indices=numpy.cumsum(numpy.arange(dimension)+1)
        tot_output=0.0

        for ind in range(dimension):

            offset=-dimension+ind+1

            if(ind==(dimension-1)):
                ## diagonal
                tot_output=tot_output+torch.diag_embed(torch.exp(log_diagonal_entries))
            else:

                if(ind==0):
                    ## add 
                    tot_output=tot_output+torch.diag_embed(lower_triangular_entries[:,:cum_indices[ind]], offset=offset)
                else:   
                    tot_output=tot_output+torch.diag_embed(lower_triangular_entries[:, cum_indices[ind-1]:cum_indices[ind]], offset=offset)

        if(upper_triangular):
            # upper triangular matrix is just transpose of lower triangular
            tot_output=tot_output.permute(0, 2,1)

        return tot_output, log_diagonal_entries.sum(axis=-1)

    else:
        raise Exception("Unknown cov type", cov_type)

def obtain_inverse_lower_triangular_matrix_and_logdet(dimension, single_log_diagonal_entry=None, log_diagonal_entries=None, lower_triangular_entries=None, cov_type="full", upper_triangular=False):

    if(cov_type=="identity"):

        return torch.eye(dimension, dtype=torch.float64).unsqueeze(0), 0.0

    elif(cov_type=="diagonal_symmetric"):
        assert(single_log_diagonal_entry is not None)
        assert(single_log_diagonal_entry.shape[1]==1)

        full_diagonal=(torch.exp(-single_log_diagonal_entry)).repeat(1, dimension)

        return torch.diag_embed(full_diagonal), -dimension*single_log_diagonal_entry.sum(axis=-1)

    elif(cov_type=="diagonal"):
        assert(log_diagonal_entries is not None)
        assert(log_diagonal_entries.shape[1]==dimension)

        return torch.diag_embed(torch.exp(-log_diagonal_entries)), -log_diagonal_entries.sum(axis=-1)

    elif(cov_type=="full"):

        assert(log_diagonal_entries is not None)
        assert(lower_triangular_entries is not None)

        cum_indices=numpy.cumsum(numpy.arange(dimension)+1)
        tot_output=0.0

        ## obtain regular matrix .. determinantes of sub matrices are used for inverse
        regular_matrix,_=obtain_lower_triangular_matrix_and_logdet(dimension, log_diagonal_entries=log_diagonal_entries, lower_triangular_entries=lower_triangular_entries, cov_type="full")
        

        largest_submatrix=regular_matrix[:, 1:,:-1]

        for ind in range(dimension):

            # ind= 0 -> offset = dim-1
            # ind =1 -> offset = dim-2 ...
            # ind = dim-1 -> offset = 0

            offset=-dimension+ind+1

            if(ind==(dimension-1)):
                ## diagonal
                tot_output=tot_output+torch.diag_embed(torch.exp(-log_diagonal_entries))
            else:

                use_sign_flip=int(numpy.fabs(offset))%2

                sign_factor=1.0
                if(use_sign_flip==1):
                    sign_factor=-1.0

                ## inverse off diagonal entries are sub-determinants divided by comibnations of diagonal
                if(ind==0):
                    ## the first 
                    #print("IND 0")
                   
                    
                    entry=(torch.det(largest_submatrix).unsqueeze(-1))/torch.exp(torch.sum(log_diagonal_entries, dim=1, keepdims=True))

                    tot_output=tot_output+torch.diag_embed(entry*sign_factor, offset=offset)

                else:   

                    off_diagonals=[]
                    num_off_diagonals=ind+1
                    largest_submatrix_subdim=dimension-1

                    individual_matrices_subdim=largest_submatrix_subdim-ind

                    for sub_ind in range(num_off_diagonals):
                        
                        off_diagonals.append(sign_factor*(torch.det(largest_submatrix[:, sub_ind:sub_ind+individual_matrices_subdim, sub_ind:sub_ind+individual_matrices_subdim]).unsqueeze(-1))/torch.exp(torch.sum(log_diagonal_entries[:, sub_ind:sub_ind+individual_matrices_subdim+1], dim=1, keepdims=True) ) )
                        
                    off_diagonals=torch.cat(off_diagonals, dim=1)
                    tot_output=tot_output+torch.diag_embed(off_diagonals, offset=offset)


        if(upper_triangular):
            tot_output=tot_output.permute(0,2,1)

     
        return tot_output, -log_diagonal_entries.sum(axis=-1)

    else:
        raise Exception("Unknown cov type", cov_type)