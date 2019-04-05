import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.time import Time
import pylab
import glob
import time
import scipy.signal
import matplotlib.cm as cm
from astropy.stats import sigma_clip
import scipy.ndimage.interpolation as interp
from scipy.ndimage.interpolation import shift
import astropy
import os
from collections import Counter
import pandas as pd
from astropy import units as u
from astropy.stats import sigma_clipping
import numpy.ma as ma

def read_raw_fits(im, newimx, newimy):
    """
    [add your own docstring and add comments to describe the purpose of this function, filling out templates below]
    Purpose: 
    Reading in the raw file
    Input:
    Raw Image, x and y of total combined image
    Returns: Final combined image
    Example Usage:
    Read in quadrant science frame from HDI and combine into full image.
    """
    # What is taking place in the following steps?
    # Answer: creating a zeroes matrix of the new image dimensions, reading in the images, finding the header of the
    # original image
    im_padded = np.zeros([newimx, newimy])
    im_all = fits.open(im)
    im_header = im_all[0].header
    # Some useful info:
    # im_padded[newimx/2:newimx, 0:newimy/2] places data in the upper left quadrant
    # im_all[4].data corresponds to the northeast quadrant of the image; needs to be inverted in Y
    # [(newimx/2):(newimx), newimy/2:(newimy)] places data in the upper right quadrant
    # im_all[3].data corresponds to the northwest quadrant of the image; needs to be inverted in X and Y
    # [0:newimx/2, 0:newimy/2] places data in lower left quadrant
    # im_all[2].data corresponds to the southeast quadrant of the image; needs no inversion
    # [0:newimx/2, newimy/2:newimy] places data in the lower right quadrant
    # im_all[1].data corresponds to the southwest quadrant of the image; needs to be inverted in X
    # What is taking place in the following four commands?
    # Answer: Taking the appropriate quadrants, inverting them in the appropriate ways and then placing them in the 
    # appropriate quadrant in the final image.
    # northeast
    im_padded[newimx//2:newimx, 0:newimy//2] = np.flipud(im_all[4].data) # can also write im_all[4].data[::-1]
    # northwest
    im_padded[(newimx//2):(newimx), newimy//2:(newimy)] = np.fliplr(np.flipud(im_all[3].data)) # can also write im_all[3].data[::-1,::-1]
    # southeast
    im_padded[0:newimx//2, 0:newimy//2] = im_all[2].data
    # southwest
    im_padded[0:newimx//2, newimy//2:newimy] = np.fliplr(im_all[1].data)
    return im_padded, im_header 

def overscan_only_bias_correct(fits_im, newimx, newimy):
    """
    Purpose: 
    For a given FITS image (fits_im), read in imframe and measure the mean overscan values for each of its quadrants.
    Then, apply the offsets to each quadrant: subtract the appropriate bias level from each quadrant in the FITS 
    image of interest.
        
    Input: 
    fits_im - raw science (or flat) frame to be corrected
    newimx - dimensions in x-axis 
    newimy - dimensions in y-axis 
    
    Returns: 
    corr_image - a bias-corrected version of the input science (or flat) image
    im_hdr - header of the input science (or flat) image
                
    Example Usage: 
    new_corrected_im, corrected_im_header = overscan_only_bias_correct('image.fits', x_image_size, y_image_size)
    """
    # First, make your input science (or flat) image into a four-quadrant mosaic. 
    # Finish the following line of code:
    imframe, imframe_hdr = read_raw_fits(fits_im, newimx, newimy)
    # Measure the overscan levels for each quadrant: 
    # Reminder: [newimx/2:newimx, 0:newimy/2] => upper left quadrant
    overscan_ul = np.median(imframe[newimx//2:newimx, (newimy//2 - 108):newimy//2]) 
    # Reminder: [(newimx/2):(newimx), newimy/2:(newimy)] => upper right quadrant
    overscan_ur = np.median(imframe[(newimx//2):(newimx), newimy//2:(newimy//2 + 108)])
    # Reminder: [0:newimx/2, 0:newimy/2] => lower left quadrant
    overscan_ll = np.median(imframe[0:newimx//2, (newimy//2-108):newimy//2])
    # Reminder: [0:newimx/2, newimy/2:newimy] => lower right quadrant
    overscan_lr = np.median(imframe[0:newimx//2, (newimy//2):(newimy//2)+108])
    # Now make a new empty image to place each subtracted quadrant
    corr_im = np.zeros([newimx, newimy])
    # ul
    corr_im[newimx//2:newimx, 0:newimy//2] = imframe[newimx//2:newimx, 0:newimy//2]- overscan_ul
    # ur
    corr_im[(newimx//2):(newimx), newimy//2:(newimy)] = imframe[(newimx//2):(newimx), newimy//2:(newimy)] - overscan_ur
    # ll
    corr_im[0:newimx//2, 0:newimy//2] = imframe[0:newimx//2, 0:newimy//2] - overscan_ll
    # lr
    corr_im[0:newimx//2, newimy//2:newimy] = imframe[0:newimx//2, newimy//2:newimy] - overscan_lr
    return corr_im, imframe_hdr

def osandframe_b_oneamp(file, bia_dir):
    """
    Purpose: 
    For a given FITS image (file), read in image and measure the mean overscan values for the entire image.
    Then, apply the offsets to the image: using Master Bias subtract the appropriate bias level from the FITS 
    image of interest. 
    WARNING: Must have created Master Bias for night before running this code.
        
    Input: 
    fits_im - raw science (or flat) frame to be corrected
    bia_dir - the directory to look for bias frames (including Master Bias)
    
    Returns: 
    corr_image - a bias-corrected version of the input science (or flat) image
    im_hdr - header of the input science (or flat) image
                
    Example Usage: 
    new_corrected_im, corrected_im_header = osandframe_b_oneamp('image.fits', bia_dir)
    """
    
    os_col_mean = []
    os_row_mean = []
    file_im = fits.getdata(file) #getting image data
    file_head = fits.getheader(file) #getting image header
    
    for col in range(4100,4140,1): #range of columns to use in overscan mean
        os_col_mean.append(np.mean(file_im[:,col]))
    for row in range(4120,4140,1): #range of rows to use in overscan mean
        os_row_mean.append(np.mean(file_im[row,:]))
    t_rmean = np.mean(os_row_mean) #mean of column means
    t_cmean = np.mean(os_col_mean) #mean of row means
    if abs(t_rmean-t_cmean)<5: #as long as the means are close
        os_mean = (t_rmean+t_cmean)//2 #average them
    else:
        print('There may be something wrong with the overscan.')
        print('the row is '+str(t_rmean)+' the col is '+str(t_cmean))
        print('   Passing '+file)
    master_bias = fits.getdata(bia_dir+'Master_bias.fits') #getting master bias
    s_m_b = master_bias*os_mean
    
    corr_im = file_im-s_m_b
    
    return corr_im, file_head

def trim_overscan(fitsimage):
    """
    Purpose: 
    Trim the overscan region from a mosaiced HDI 4-amplifier image. This code assumes that the overscan region 
    runs down the middle of the image (i.e., image has been rotated North up, East left, with quadrants in the correct positions), and that the input FITS 
    file has a header to extract the size of the overscan region. 
    Input: 
    fitsimage - A mosaiced FITS image including all four quadrants. Must include a header with keywords providing 
                the size of the overscan region (OVRSCAN1).
    Returns: 
    trimmed_image - image array with the overscan region removed.
    im_hdr - header of the input FITS image.         
    Example Usage: 
    im_no_overscan, im_head = trim_overscan('test_image.fits')
    """
    # Get the data and header from the image
    im, im_hdr = fits.getdata(fitsimage, header = True)
    # Determine the image dimensions 
    y_imsize = int(im.shape[0])
    x_imsize = int(im.shape[1])
    # Pull the size of the overscan region out of the header
    overscan_size = int(im_hdr['OVRSCAN1'])
    # Define the columns we'd like to remove from the image array (where to remove along x-axis)
    cols_to_remove = np.s_[(int(x_imsize/2) - overscan_size):(int(x_imsize/2) + overscan_size)]
    # Use numpy's delete function to remove those columns (axis=1 removes columns; axis=0 removes rows)
    trimmed_image = np.delete(im, cols_to_remove, axis = 1)
    return trimmed_image, im_hdr

def oneamp_trim_os(fitsimage):
    """
    Purpose: 
    Trim the overscan region from an HDI 1-amplifier image. This code assumes that the overscan region 
    runs in specific regions on the top and right side of the image, and that the input FITS 
    file has a header to extract the size of the overscan region. 
    Input: 
    fitsimage - A mosaiced FITS image including all four quadrants. Must include a header with keywords providing 
                the size of the overscan region (OVRSCAN1).
    Returns: 
    trimmed_image - image array with the overscan region removed.
    im_hdr - header of the input FITS image.         
    Example Usage: 
    im_no_overscan, im_head = trim_overscan('test_image.fits')
    """
    # Get the data and header from the image
    im, im_hdr = fits.getdata(fitsimage, header = True)
    # Determine the image dimensions 
    y_imsize = int(im.shape[0])
    x_imsize = int(im.shape[1])
    # Pull the size of the overscan region out of the header
    overscan_size = int(im_hdr['OVRSCAN1'])
    # Define the columns we'd like to remove from the image array (where to remove along x-axis)
    cols_to_remove = np.s_[4096:len(im)]
    rows_to_remove = np.s_[4112:len(im)]
    # Use numpy's delete function to remove those columns (axis=1 removes columns; axis=0 removes rows)
    c_trimmed_image = np.delete(im, cols_to_remove, axis = 1)
    trimmed_image = np.delete(c_trimmed_image, rows_to_remove, axis = 0)
    return trimmed_image, im_hdr

def median_combine (filelist,normal=False,sigma_clip=False):
    '''
    Purpose: Creates a median combine image from a file list.
    
    Syntax: med_image = median_combine(filelist)
    
    Inputs:
    filelist = a list of files names as a string
    
    Outputs: 
    an array of float, to be written into a .fits file
    '''
    n = len(filelist)
    #This is the number of images with which to take the median of
    first_frame_data = fits.getdata(filelist[0])
    #This gets the dimensions of the first image
    imsize_y, imsize_x = first_frame_data.shape
    #This creates a new array of the same dimensions as the first image
    fits_stack = np.zeros((n, imsize_y, imsize_x), dtype = np.float32) 
    #Establishes a blank/zeros array
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        if normal:
            im_normal = im/np.median(im)
            fits_stack[ii] = im_normal
        else:
            fits_stack[ii] = im
        #It iterates through both dimensions and puts all the data from the raw images into the blank/zero array
    if sigma_clip:
        clipped_arr = sigma_clipping.sigma_clip(fits_stack,sigma=3,axis=0)
        median_clipped_arr = np.ma.median(clipped_arr,axis=0)
        med_frame = median_clipped_arr.filled(0.)
        
    else:
        # median and normalize the single-value frame
        med_frame = np.median(flat,axis=0)
    #This creates the median frame
    return med_frame

def flat_type_sorter(filelist):
    '''
    Purpose: Sorts flats into type based on DEC position. Will only work for WIYN 0.9 m.
    
    Syntax: twi_list, dome_list = flat_type_sorter(filelist)
    
    Inputs:
    filelist = a list of flat files names as a string
    
    Outputs: 
    twi_list = a list of twilight flat names
    dome_list = a list of dome flat names
    '''
    twi_list = []
    dome_list = []
    exp_twi = []
    exp_dome = []
    obs_t_twi = []
    obs_t_dome = []
    for ind,filelist in enumerate(filelist):
        header = fits.getheader(filelist)
        obs_t = Time(header['DATE-OBS'],format='iso',scale='utc')
        if header['DECSTRNG']=='+30:00:00': #checking the declination for zenith positioning
            twi_list.append(filelist) #if at zenith it is a twilight flat
            obs_t_twi.append(obs_t.jd)
            exp_twi.append(header['EXPTIME'])
        else:
            dome_list.append(filelist) #if its anywhere else its a dome flat
            obs_t_dome.append(obs_t.jd)
            exp_dome.append(header['EXPTIME'])
    #checking the exposure times and time of exposure, which could have slipped through other checks  
    av_t_twi = np.median(obs_t_twi) #average time that the flats were taken (Julian Date)
    av_t_dome = np.median(obs_t_dome) 
    for ind,filelist in enumerate(twi_list):
        head = fits.getheader(filelist) 
        exp = head['EXPTIME']
        t_diff = abs(obs_t_twi[ind]-av_t_twi) #difference between the time of the specific flat and the average time
        if t_diff>0.02: #if that difference is too large
            del twi_list[ind] #delete that image from the list
        if exp!=max(exp_twi, key=Counter(exp_twi).get): #if the exposure time is not the most common expusure
            del twi_list[ind] #delete that image from the list
    #for ind,filelist in enumerate(dome_list):
        #head = fits.getheader(filelist)
        #exp = head['EXPTIME']
        #t_diff = abs(obs_t_dome[ind]-av_t_dome) #difference between the time of the specific flat and the average time
        #if t_diff>0.04: #if that difference is too large
            #del dome_list[ind] #delete that image from the list
        #if exp!=max(exp_dome, key=Counter(exp_dome).get):
            #del dome_list[ind]
            
    return twi_list, dome_list

def master_flat_creator (filelist,filt,cut):
    '''
    Purpose: Creates a master flat from a given flat list.
    
    Syntax: master_flat_creator(filelist,filt,cut)
    
    Inputs:
    filelist = a list of flat files names as a string
    filt = a string with the filter given to create a name with filter (ie Master_V_flat.fits)
    cut = the last folder before the files as a string
    
    Outputs: 
    writes a fits file w/ name Master_filt_flat.fits
    '''
    
    if filelist!=[]: #checks if the filelist is empty
        header = fits.getheader(filelist[0]) 
        #med = median_combine(filelist,normal=True,sigma_clip=True) #median combine the list
        #master = med/np.median(med) #normalize the median frame
        master = median_combine(filelist,normal=True,sigma_clip=True) #median combine the list
        shortname = filelist[0].split(cut)
        fits.writeto(shortname[0]+cut+'Master_'+filt+'_flat.fits', master, header, overwrite=True)
        print('Master Flat for ' + filt + ' created in ' + shortname[0] + cut)
    else:
        print('The '+filt+' list was empty.')
    return
        
def master_bias_creator (filelist,cut):
    '''
    Purpose: Creates a master bias from a given bias list.
    
    Syntax: master_bias_creator(filelist,cut)
    
    Inputs:
    filelist = a list of flat files names as a string
    cut = the last folder before the files as a string
    
    Outputs: 
    writes a fits file w/ name Master_bias.fits
    '''
    
    if filelist!=[]: #checks if the filelist is empty
        header = fits.getheader(filelist[0]) 
        med = median_combine(filelist,sigma_clip=True) #median combine the list
        master = med/np.median(med) #normalize the median frame
        shortname = filelist[0].split(cut)
        fits.writeto(shortname[0]+cut+'Master_bias.fits', master, header, overwrite=True)
        print('Master Bias created in ' + shortname[0] + cut)
    else:
        print('The bias list was empty.')
    return
    
def cross_image(im1_name, im2_name, **kwargs):
    '''
    cross_image
    ---------------
    calcuate cross-correlation of two images in order to find shifts
    inputs
    ---------------
    im1                      : (matrix of floats)  first input image
    im2                      : (matrix of floats) second input image
    boxsize                  : (integer, optional) subregion of image to cross-correlate
    xmin/xmax                : (integer, optional) x dimen of a subregion of image to cross-correlate
    ymin/ymax                : (integer, optional) y dimen of a subregion of image to cross-correlate
    
    returns
    ---------------
    xshift                   : (float) x-shift in pixels
    yshift                   : (float) y-shift in pixels
    dependencies
    ---------------
    scipy.signal.fftconvolve : two-dimensional fourier convolution
    centroid                 : a centroiding algorithm of your choosing or defintion
    numpy                    : imported as np
    todo
    ---------------
    -add more **kwargs capabilities for centroid argument
    
    '''
    # The type cast into 'float' is to avoid overflows:
    im1 = fits.getdata(im1_name)
    im2 = fits.getdata(im2_name)
    
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')
    # Enable a trimming capability using keyword argument option.
    if 'boxsize' in kwargs:
        im1_gray = im1_gray[0:kwargs['boxsize'],0:kwargs['boxsize']]
        im2_gray = im2_gray[0:kwargs['boxsize'],0:kwargs['boxsize']]
    #Enable a trimming capability using keyword argument option.
    if 'xmin' in kwargs:
        im1_gray = im1_gray[kwargs['ymin']:kwargs['ymax'],kwargs['xmin']:kwargs['xmax']]
        im2_gray = im2_gray[kwargs['ymin']:kwargs['ymax'],kwargs['xmin']:kwargs['xmax']]

    # Subtract the averages (means) of im1_gray and im2_gray from their respective arrays     
    im1_gray -= np.nanmean(im1_gray)
    im2_gray -= np.nanmean(im2_gray)
    # guard against extra nan values
    im1_gray[np.isnan(im1_gray)] = np.nanmedian(im1_gray)
    im2_gray[np.isnan(im2_gray)] = np.nanmedian(im2_gray)
    # Calculate the correlation image using fast Fourrier Transform (FFT)
    # Note the flipping of one of the images (the [::-1]) to act as a high-pass filter
    corr_image = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')
    # Find the peak signal position in the cross-correlation, which gives the shift between the images
    corr_tuple = np.unravel_index(np.nanargmax(corr_image), corr_image.shape)
    try: # try to use a centroiding algoritm to find a better peak
        xcenc,ycenc = centroid(corr_image.T,corr_tuple[0],corr_tuple[1],nhalf=10,derivshift=1.)
    except: # if centroiding algorithm fails, use just peak pixel
        xcenc,ycenc = corr_tuple   
    # Calculate shifts (distance from central pixel of cross-correlated image)
    xshift = xcenc - corr_image.shape[0]/2.
    yshift = ycenc - corr_image.shape[1]/2.
    return xshift,yshift

def shift_image(image,xshift,yshift):
    '''
    shift_image
    -------------
    wrapper for scipy's implementation that shifts images according to values from cross_image
    
    inputs
    ------------
    image           : (matrix of floats) image to be shifted
    xshift          : (float) x-shift in pixels
    yshift          : (float) y-shift in pixels
    
    outputs
    ------------
    shifted image   : shifted, interpolated image. 
                      same shape as input image, with zeros filled where the image is rolled over
    
    
    '''
    return scipy.ndimage.interpolation.shift(image,(xshift,yshift))

def flat_divide(filename, path_to_flat, cut):
    '''
    This divides an already bias and dark subtracted file by the master flat frame    
    '''
    data = fits.getdata(filename)
    header = fits.getheader(filename)
    
    #testing for zeros in flat (impact flat divide)
    t_flat = fits.getdata(path_to_flat)
    zeros = np.argwhere(t_flat==0)
    try:
        if not zeros:
            masterFlat = t_flat
    except(ValueError):
        print('The Flat had ZEROs. Replacing them with the median of the image.')
        t_flat[zeros]=np.median(t_flat)
        masterFlat = t_flat
    
    superDivide = data/masterFlat
    shortname = filename.split(cut)
    fits.writeto(shortname[0]+cut+'f' + shortname[1], superDivide, header, overwrite=True)
    print('   Wrote FITS file '+shortname[0]+cut+'f' + shortname[1])
    return

def combined_im_creator (filelist,cut):
    if filelist!=[]: #checks if the filelist is empty
        header = fits.getheader(filelist[0]) 
        med = median_combine(filelist) #median combine the list
        shortname = filelist[0].split(cut)
        fits.writeto(shortname[0]+cut+'c_' + shortname[1], med, header, overwrite=True)
        print('Combined image c_'+shortname[1]+' created in ' + shortname[0] + cut)
    else:
        print('The given list was empty.')
    return

def centroid(data_arr,xcen,ycen,nhalf=5,derivshift=1.):
    '''
    centroid
    -----------------
    based on dimension-indepdendent line minimization algorithms implemented in IDL cntrd.pro 
    
    inputs
    ----------------
    data_arr      : (matrix of floats) input image
    xcen          : (int) input x-center guess
    ycen          : (int) input y-center guess
    nhalf         : (int, default=5) the excised box of pixels to use
                    recommended to be ~(2/3) FWHM (e.g. only include star pixels).
    derivshift    : (int, default=1) degree of shift used to calculate derivative. 
                     larger values can find shallower slopes more efficiently
    
    outputs
    ---------------
    xcenf         : the centroided x value
    ycenf         : the centroided y value
    
    dependencies
    ---------------
    numpy         : imported as np
    
    also see another implementation here:
    https://github.com/djones1040/PythonPhot/blob/master/PythonPhot/cntrd.py
    
    '''
    # input image requires the transpose to 
    #
    # find the maximum value near the given point
    data = data_arr[int(ycen-nhalf):int(ycen+nhalf+1),int(xcen-nhalf):int(xcen+nhalf+1)]
    yadjust = nhalf - np.where(data == np.max(data))[0][0]
    xadjust = nhalf - np.where(data == np.max(data))[1][0]
    xcen -= xadjust
    ycen -= yadjust
    #
    # now use the adjusted centers to find a better square
    data = data_arr[int(ycen-nhalf):int(ycen+nhalf+1),int(xcen-nhalf):int(xcen+nhalf+1)]
    #
    # make a weighting function
    ir = (nhalf-1) > 1 
    # sampling abscissa: centers of bins along each of X and Y axes
    nbox = 2*nhalf + 1
    dd = np.arange(nbox-1).astype(int) + 0.5 - nhalf
    #Weighting factor W unity in center, 0.5 at end, and linear in between 
    w = 1. - 0.5*(np.abs(dd)-0.5)/(nhalf-0.5) 
    sumc   = np.sum(w)
    #
    # find X centroid
    # shift in Y and subtract to get derivative
    deriv = np.roll(data,-1,axis=1) - data.astype(float)
    deriv = deriv[nhalf-ir:nhalf+ir+1,0:nbox-1]
    deriv = np.sum( deriv, 0 )                    #    ;Sum X derivatives over Y direction
    sumd   = np.sum( w*deriv )
    sumxd  = np.sum( w*dd*deriv )
    sumxsq = np.sum( w*dd**2 )
    dx = sumxsq*sumd/(sumc*sumxd)
    
    xcenf = xcen - dx
    #
    # find Y centroid
    # shift in X and subtract to get derivative
    deriv = np.roll(data,-1,axis=0) - data.astype(float)    # Shift in X & subtract to get derivative
    deriv = deriv[0:nbox-1,nhalf-ir:nhalf+ir+1]
    deriv = np.sum( deriv,1 )               #    ;Sum X derivatives over Y direction
    sumd   = np.sum( w*deriv )
    sumxd  = np.sum( w*dd*deriv )
    sumxsq = np.sum( w*dd**2 )  
    dy = sumxsq*sumd/(sumc*sumxd)
    ycenf = ycen - dy
    return xcenf,ycenf

def close_im (filelist,cut,night,filt,**kwargs):
    '''
    Purpose: Creates a combined image of all the images that are within a specified timeframe.
    
    Syntax: close_im(filelist,cut,filt,**kwargs)
    
    Inputs:
    filelist = a list of flat files names as a string
    cut = the last folder before the files as a string
    night = the night the data was taken as a string
    filt = the filter of the filelist as a string
    spec_time_diff = a specified maximum time difference between images in JD (optional)
    
    Outputs: 
    writes a fits file w/ name c_NightFilterNumber for every combination in list
    '''
    if 'spec_time_diff' in kwargs:
        time_difference = kwargs['spec_time_diff'] #make the time difference the specified time difference
    else:
        time_difference = 0.004 #or approx five minutes in JD
    loop = 0
    
    #cross correlation box dimensions
    ymin = 3100
    ymax = 3900
    xmin = 2820
    xmax = 3820
    
    while filelist != []:
        little_list = []
        s_lil_list = []
        header_1 = fits.getheader(filelist[0]) #getting header of first image in list
        time_1 = Time(header_1['DATE-OBS'],format='iso',scale='utc') #time of observation of first image
        for ind, filename in enumerate(filelist):
            header = fits.getheader(filename) #getting header of image
            obs_t = Time(header['DATE-OBS'],format='iso',scale='utc')#time of observation of image
            T_dif = obs_t.jd-time_1.jd #time difference b/w the first image and image
            if T_dif<time_difference: #if the time difference is less than the specified time difference
                little_list.append(filename) #add the image to the little list
        for filename in little_list:
            w = np.where(filename==np.array(filelist))[0][0] #finding the images that little list found
            del filelist[w] #delete the found image from the list
        imsize_y, imsize_x = fits.getdata(little_list[0]).shape
        #This creates a new array of the same dimensions as the first image
        s_lil_list = np.zeros((imsize_y, imsize_x , len(little_list)), dtype = np.float32) 
        for ind, filename in enumerate (little_list):
            
            x_shift, y_shift = cross_image(little_list[0], little_list[ind], ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax) #find the shifts
            image = fits.getdata(filename)
            #print(str(np.median(image))+'is the median of the indiv image'+str(x_shift)+' '+str(y_shift))
            s_lil_list[:,:,ind] = shift_image(image,x_shift,y_shift) #make the shifted image
            #print('Masked? '+str(ma.is_masked(image)))
        loop = loop+1
        med_frame = np.median(s_lil_list, axis=2)
        print(str(np.median(med_frame))+'is the median of the combined image')
        header = fits.getheader(little_list[0])
        shortname = little_list[0].split(cut)
        fits.writeto(shortname[0]+cut+'c_'+night+filt+str(loop)+'.fits', med_frame, header, overwrite=True)
        print('Combined image c_'+night+filt+str(loop)+'.fits'+' created in ' + shortname[0] + cut)
    return

def combined_im_sort(filelist,filt,verbose):
    '''
    Purpose: Sorts combined images into their appropriate combined folders
    
    Syntax: combined_im_sort(filelist,filt,verbose)
    
    Inputs:
    filelist = a pre-sorted filelist with the combined files
    filt = the filter of the filelist
    verbose = when verbose=True, prints things, when verbose=False, doesn't print things
    
    Outputs: 
    sorts files into appropriate folders
    '''
    direct = os.getcwd() #get directory
    datadir = direct+'/Data' #the data file in the directory
    destination = datadir + '/' + filt +'_Combined' #where the files should end up
    if os.path.exists(destination): #if the desitination already exists
        pass
    else: #make a new directory
        if verbose == True:
            print("Making new directory: " + destination) 
        os.mkdir(destination)
    for ind, filename in enumerate(filelist): #over all the files in the given list
        shortname = filename.split('Science/') #get the filename
        if verbose == True:
            print("Moving " + filename + " to: " + destination + '/' + shortname[1])
        os.rename(filename, destination + '/' + shortname[1])  #move it to the destination

def jd_diff_calc(number, unit):
    '''
    Purpose: Quick Julian Date time difference calculator.
    
    Syntax: jd_diff_calc(number, unit)
    
    Inputs:
    number = the number of what ever time unit you want in JD
    unit = the unit of what ever time difference you want (Must be either Hour, Minute, or Second)
    
    Outputs: 
    prints the Julian Date time difference for whatever time inputed
    '''
    hour_jd = 1./24
    min_jd = hour_jd/60
    sec_jd = min_jd/60
    if 'Hour' == unit:
        time_diff = number*hour_jd
    elif 'Minute' == unit:
        time_diff = number*min_jd
    elif 'Second' == unit:
        time_diff = number*sec_jd
        
    print(str(number)+' '+unit+' in Julian date is: '+str(time_diff))
    return
