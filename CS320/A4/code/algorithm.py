# CSC320 Winter 2017
# Assignment 4
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):

    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################
    # Get dimention of image
    N = source_patches.shape[0]
    M = source_patches.shape[1]
    C = source_patches.shape[2]
    P = source_patches.shape[3]
    num = (np.log(1./w)/np.log(alpha)).astype(int)+1
    window = (w*alpha**np.arange(num)).reshape(-1,1)
    # Define the direction of loop, if it is odd iteration, we start from top to
    # bottom; if it is even iteration, we start from bottem to top
    if(odd_iteration):
        istart,istop,istep, jstart,jstop,jstep= (0,N,1,0,M,1)
    else:
        istart,istop,istep, jstart,jstop,jstep= (N-1,-1,-1,M-1,-1,-1)
    for i in range(istart,istop,istep):
        for j in range(jstart,jstop,jstep):
            current = source_patches[i,j].reshape((-1))
            offsetdict = f_coord_dictionary[i][j][0]
            desfunc = lambda m: [i,j]+ m[2]
            # Implement propagation algorithm
            if(propagation_enabled):
                for x in ["v","h"]:
                    ok = False                  
                    if(x == 'v' and 0<=i-istep<N):
                        des = np.array(map(desfunc, f_heap[i-istep][j]))
                        ok = True
                    if(x == 'h' and 0<=j-jstep<M):
                        des = np.array(map(desfunc, f_heap[i][j-jstep]))
                        ok = True
                    if(ok):                   
                        des = np.clip(des, [0,0], [N-1,M-1])
                        valid = []
                        for v in des - [i,j]:
                            if((v[0],v[1]) not in offsetdict):
                                offsetdict[(v[0],v[1])] = None
                                valid.append(v)
                        if(len(valid)!=0):
                            valid = np.array(valid)
                            des = [i,j] + valid
                            ktargets = target_patches[des[:,0],des[:,1]].reshape((-1,C*P))                        
                            different = current - ktargets                         
                            weight = np.count_nonzero(~np.isnan(different),axis=1)        
                            different[np.isnan(different)] = 0
                            similarity = -np.einsum('ij, ij->i', different, different)/weight
                            map(lambda x,y:heappushpop(f_heap[i][j],(x,_tiebreaker.next(),y)), similarity,valid)
            # Implement random algorithm
            if(random_enabled):
                coffsets = f_heap[i][j]
                ui = np.array(map(lambda o: o[2]+ window*np.random.uniform(-1,1,(num,2)),coffsets))
                tl = ([i,j]+ui).astype(int)
                tl = np.clip(tl, [0,0], [N-1,M-1]).reshape(-1,2)
                ui = []
                for v in tl - [i,j]:
                    if((v[0],v[1]) not in offsetdict):
                        offsetdict[(v[0],v[1])] = None
                        ui.append(v)
                if(len(ui)!=0):
                    ui = np.array(ui)
                    tl = [i,j]+ ui
                    targets = target_patches[tl[:,0],tl[:,1]].reshape((-1,C*P))
                    diff = current - targets
                    Weight = np.count_nonzero(~np.isnan(diff),axis=1)
                    diff[np.isnan(diff)] = 0
                    random_scores = -np.einsum('ij, ij->i', diff, diff)/Weight
                    map(lambda x,y:heappushpop(f_heap[i][j],(x,_tiebreaker.next(),y)), random_scores,ui)
    #############################################

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    f_heap = None
    f_coord_dictionary = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    N = source_patches.shape[0]
    M = source_patches.shape[1]
    K = f_k.shape[0]
    g = make_coordinates_matrix(source_patches.shape[:3])
    t = (g + f_k).reshape(-1,2)
    source = source_patches.reshape(N,M,-1)
    match = target_patches[t[:,0],t[:,1],:,:].reshape(K,N,M,-1)
    difference = source-match
    weight = np.count_nonzero(~np.isnan(difference),axis=3)
    difference[np.isnan(difference)] = 0
    priority = -np.einsum('ijkw, ijkw->ijk', difference, difference)/weight
    f_heap = []
    f_coord_dictionary = []
    for i in range(N):
        row = []
        drow = []
        f_heap.append(row)
        f_coord_dictionary.append(drow)
        for j in range(M):
            col = []
            d = {}
            dcol = [d]
            row.append(col)
            drow.append(dcol)
            for k in range(K):
                tup = (priority[k][i][j],_tiebreaker.next(),f_k[k][i][j])
                heappush(col,tup)
                d[(f_k[k][i][j][0],f_k[k][i][j][1])] = None
    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    N = len(f_heap)
    M = len(f_heap[0])
    K = len(f_heap[0][0])
    f_kshape = K,N,M,2
    f_k = np.zeros(f_kshape)*np.nan
    D_kshape = K,N,M
    D_k = np.zeros(D_kshape)*np.nan
    for i in range(N):
        for j in range(M):
            items = nlargest(K,f_heap[i][j])
            for k in range(len(items)):
                f_k[k,i,j] = items[k][2]
                D_k[k,i,j] = -items[k][0]
    f_k = f_k.astype(int)
    #############################################

    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    #denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    #calculate weight
    N = target.shape[0]
    M = target.shape[1]
    K = len(f_heap[0][0])
    g = make_coordinates_matrix(target.shape)
    f_k, D_k = NNF_heap_to_NNF_matrix(f_heap)       
    tlocation = (g+f_k).reshape((-1,2))    
    kim = target[tlocation[:,0],tlocation[:,1]].reshape((-1,N,M,3))
    epower = np.exp(-(D_k**.5/h**2))
    Z = np.sum(epower, axis=0)
    w = epower/Z
    denoised = np.zeros(target.shape)
    for n in range(N):
        for m in range(M):
            for k in range(K):
                denoised[n,m] += kim[k,n,m]*w[k,n,m]
        
    #############################################

    return denoised




#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################

#############################################



# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################
    tlocation = (make_coordinates_matrix(target.shape)+f).reshape(-1,2)
    rec_source = target[tlocation[:,0],tlocation[:,1]].reshape(-1,f.shape[1],3)    

    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
