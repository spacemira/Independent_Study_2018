# I am a module!

import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.io import fits
import matplotlib.cm as cm
import scipy.signal
import glob
import os
from astropy.stats import sigma_clip
import time
import scipy.ndimage.interpolation as interp
from astropy.io import fits 

def func(procdir,filt,coord,background,padding,targname):
    #func(band,coord,background,procdir):
    def mediancombine(filelist):
        '''
        This makes a median of a series of images
        '''
        n = len(filelist)
        #This is the number of images with which to take the median of
        first_frame_data = fits.getdata(filelist[0])
        #This gets the dimensions of the first image
        imsize_y, imsize_x = first_frame_data.shape
        #This creates a new array of the same dimensions as the first image
        fits_stack = np.zeros((imsize_y, imsize_x , n), dtype = np.float32) 
        #Establishes a blank/zeros array
        for ii in range(0, n):
            im = fits.getdata(filelist[ii])
            fits_stack[:,:,ii] = im
            #It iterates through both dimensions and puts all the data from the raw images into the blank/zero array
        med_frame = np.median(fits_stack, axis=2)
        #This creates the median frame
        return med_frame
    
    def bias_subtract(filename, path_to_bias, cut):
        '''
        This reads in the data, the master bias file and subtracts the bias from the data
        '''
        data = fits.getdata(filename)
        header = fits.getheader(filename)
        
        masterBias = fits.getdata(path_to_bias)
        
        superSubtract = data-masterBias
        shortname = filename.split(cut)
        fits.writeto(shortname[0]+cut+'b_' + shortname[1], superSubtract, header, clobber=True)
        print('   Wrote FITS file ',shortname[0]+'b_' + shortname[1])
        return 
    
    def dark_subtract(filename, path_to_dark, cut):
        '''
        This subtracts the dark file from an already bias-corrected file    
        '''
        data = fits.getdata(filename)
        header = fits.getheader(filename)
    
        masterDark = fits.getdata(path_to_dark)
    
        superSubtract = data-masterDark
        shortname = filename.split(cut)
        fits.writeto(shortname[0]+cut+'d' + shortname[1], superSubtract, header, clobber=True)
        print('   Wrote FITS file ', shortname[0]+'d' + shortname[1])
        return   
    
    def flat_divide(filename, path_to_flat, cut):
        '''
        This divides an already bias and dark subtracted file by the master flat frame    
        '''
        data = fits.getdata(filename)
        header = fits.getheader(filename)
        
        masterFlat = fits.getdata(path_to_flat)
        
        superDivide = data/masterFlat
        shortname = filename.split(cut)
        fits.writeto(shortname[0]+cut+'f' + shortname[1], superDivide, header, clobber=True)
        print('   Wrote FITS file ','f' + filename)
        return    
    
    def norm_combine_flats(filelist):
        '''
        normalizes and combines flats
        '''
        n = len(filelist)
        first_frame_data = fits.getdata(filelist[0])
        imsize_y, imsize_x = first_frame_data.shape
        fits_stack = np.zeros((imsize_y, imsize_x , n), dtype = np.float32) 
        for ii in range(0, n):
            im = fits.getdata(filelist[ii])
            norm_im =  im/np.median(im)
            fits_stack[:,:,ii] = norm_im
        med_frame = np.median(fits_stack, axis=2)
        return med_frame
    
    def dark_rescale(inttime, masterdarkpath, masterdarkexptime, cut):
        '''
        Purpose: Re-scale a master dark frame to a given exposure time.
    
        Required inputs:
        inttime:           Desired integration time to rescale to (this can be a single integer or 
                        a list of exposure times, e.g., [1.0, 10.0, 90.0]). Units are in seconds.
        masterdarkpath:    File path to the master dark FITS file (e.g., /calibration/darks/Master_Dark_60s.fits)
        masterdarkexptime: Exposure time of the master dark frame
    
        Output:
        One or more re-scaled dark frames with the naming convention "Master_Dark_#s.fits"
    
        '''
        # Check if the desired integration time is an integer or list:
        if type(inttime) is not list: inttime = [inttime]

        # Read in the master dark data:
        masterdark = fits.getdata(masterdarkpath)
        header = fits.getheader(masterdarkpath)

        # Normalize the master dark frame to a 1-second dark by dividing by its exposure time:
        dark_1s = masterdark/masterdarkexptime
        shortname = masterdarkpath.split(cut)

        # Rescale for all the new exposure times, and write out new scaled dark files:
        for newtime in inttime:
            scaled_dark_data = dark_1s * newtime
            header['EXPTIME'] = newtime
            header['EXPOSURE'] = newtime
            newdarkname = "Master_Dark_" + str(newtime) + "s.fit"
            print("Writing ", newdarkname)
            fits.writeto(shortname[0]+cut+newdarkname, scaled_dark_data, header, clobber=True)
            new_master_dark_path = shortname[0]+cut+newdarkname

        print("Finished rescaling.")
        return new_master_dark_path

    
    #----------------------------------------------------------------
    
    #need bias files
    biasfiles = glob.glob(procdir+'calibration\\biasframes\\*_bias.fit.gz')
    median_bias = mediancombine(biasfiles)
    biasheader = fits.getheader(biasfiles[0])
    fits.writeto(procdir+'calibration\\biasframes\\'+"MasterBias.fit", median_bias, biasheader, clobber=True)
    print('   Wrote FITS file MasterBias.fit', 'in ',procdir+'\\calibration\\biasframes\\')
    master_bias_path = procdir + 'calibration\\biasframes\\MasterBias.fit'
    
    #need dark files
    darks = glob.glob(procdir+'calibration\\darks\\60sec\\*b_*.fit.gz')
    if (len(darks)==0):
        darkfiles = glob.glob(procdir+'calibration\\darks\\60sec\\*_dark*.fit.gz')
        cut = '60sec\\'
        darkheader = fits.getheader(darkfiles[0])
        #need master_bias_path
        for filename in darkfiles:
            bias_subtract(filename, master_bias_path,cut)
        darks = glob.glob(procdir+'calibration\\darks\\60sec\\*b_*.fit.gz')
        mediandarks=mediancombine(darks)
        fits.writeto(procdir+'calibration\\darks\\60sec\\'+"Master_Dark_60s.fit", mediandarks, darkheader, clobber=True)
        print('   Wrote FITS file Master_Dark_60s.fit', 'in ',procdir)
    
    darkheader = fits.getheader(darks[0])
    cut = '60sec\\'
    
    master_dark_path = procdir + 'calibration\\darks\\60sec\\Master_Dark_60s.fit'
    masterdarkexptime = darkheader['EXPTIME']
    
    iflatlist = glob.glob(procdir+'calibration\\flats\\Iflat\\*_Iflat.fit.gz')
    vflatlist = glob.glob(procdir+'calibration\\flats\\Vflat\\*_Vflat.fit.gz')
    
    if (filt=='I'):
            flatlist = iflatlist
            flat_cut = 'Iflat\\'
            #path to r flat here
    else:
        if (filt=='V'):
            flatlist = vflatlist
            flat_cut = 'Vflat\\'
                #path to v flat here
    
    flatheader = fits.getheader(flatlist[0])
    
    inttime = flatheader['EXPTIME']
    
    #dark_rescale(inttime, master_dark_path, masterdarkexptime,cut)
    
    new_master_dark_path = dark_rescale(inttime, master_dark_path, masterdarkexptime, cut)
    
    
    for filename in flatlist:
        bias_subtract(filename, master_bias_path, flat_cut)
    

    i_bias_flat = glob.glob(procdir+'calibration\\flats\\Iflat\\b_*_Iflat.fit.gz')
    v_bias_flat = glob.glob(procdir+'calibration\\flats\\Vflat\\b_*_Vflat.fit.gz')
    if (filt=='I'):
            bias_flat = i_bias_flat
            flat_cut = 'Iflat\\'
            #path to r flat here
    else:
        if (filt=='V'):
            bias_flat = v_bias_flat
            flat_cut = 'Vflat\\'
                #path to v flat here
    
    for filename in bias_flat:
        dark_subtract(filename, new_master_dark_path, flat_cut)
    
    norm_flat = norm_combine_flats(bias_flat)
    
    fits.writeto(procdir+'calibration\\flats\\'+flat_cut+'Master_Flat_'+filt+'.fit', norm_flat, flatheader, clobber=True)
    print('   Wrote FITS file Master_Flat', 'in ',procdir)
    master_flat_path = procdir+'calibration\\flats\\'+flat_cut+'Master_Flat_'+filt+'.fit'

    #-------------------------------------------------------------------------
    
    def cross_image_ME(coord, background, imlist):
        """
        Takes a list of images, takes a given region in each image and finds the centroid of the star in that region then calculates
        the offsets of each image.
        """
        x_cent = []
        y_cent = []
        for ind,image in enumerate(imlist):
            data = fits.getdata(imlist[ind]) #values for each of the images
            guide = data[coord[0]:coord[1],coord[2]:coord[3]] #taking the user inputed guide star
            bckgrnd_d = data[background[0]:background[1],background[2]:background[3]] #taking in the user inputed background
            med_bckgrnd = np.median(bckgrnd_d) #taking the median of the background
            std_bckgrnd = np.std(bckgrnd_d) #taking the standard dev of the background
            comp = med_bckgrnd+(3*std_bckgrnd) #what the values are being compared to
            dim = guide.shape #the dimensions of the guide image
            sub_data = guide-med_bckgrnd #the data without the median background
            for x in range(0,dim[0]): 
                for y in range(0,dim[1]):
                    if guide[x,y]<comp: #comparing the pixels to the discriminating value
                        sub_data[x,y]=0

            topx = np.zeros(dim[0])
            topy = np.zeros(dim[1])
            for x in range(0,dim[0]):
                sumxax = np.sum(sub_data,axis=1)
                topx[x] = x*sumxax[x]
                botx = sum(sumxax)
    
            for y in range(0,dim[1]):
                sumyax = np.sum(sub_data,axis=0)
                topy[y] = y*sumyax[y]
                boty = sum(sumyax)
            xc = sum(topx)/botx #individual x centroid for the image
            yc = sum(topy)/boty #individual y centroid for the image
            x_cent.append(xc) #storing the centroid coordinates
            y_cent.append(yc)
        
        m_x_cent = x_cent[0] #'master' coordinates as the first image
        m_y_cent = y_cent[0]
    
        offset_x = x_cent-m_x_cent #offset of each image from the master
        offset_y = y_cent-m_y_cent

        return offset_x,offset_y
    
    def shift_function_ME(offset_x, offset_y, padding, imlist,cut):
        """
        Takes a list of images, takes the corresponding offset of that image in the x and y directions, and output registered
        images to disk generated from these offsets and a median combined image.
        """
        datadir = os.getcwd()
        for ind,image in enumerate(imlist):
            data = fits.getdata(imlist[ind]) #values for each of the images
            filename = imlist[ind]
            off_x = offset_x[ind]
            off_y = offset_y[ind]
            pad_image = np.pad(data,padding,'constant', constant_values=-0.001)
            shift_image = interp.shift(pad_image,(off_y,off_x), cval=-0.001)
            shift_image[shift_image <= 1] = np.nan
            header = fits.getheader(filename)
            shortname = filename.split(cut)
            fits.writeto(shortname[0]+cut+'shift_' + shortname[1], shift_image, header, clobber=True)
            print('   Wrote FITS file ',shortname[0]+'shift_'+ shortname[1])
        
        shortname = imlist[0].split(cut)
        i_data = glob.glob(procdir+targname+'\\Iband\\shift_*I.fit')
        v_data = glob.glob(procdir+targname+'\\Vband\\shift_*V.fit')

        if (filt=='I'):
                filelist = i_data
                data_cut = 'Iband\\'
                #path to r flat here
        else:
            if (filt=='V'):
                filelist = v_data
                data_cut = 'Vband\\'
                #path to v flat here
        n = len(filelist)
        #This is the number of images with which to take the median of
        first_frame_data = fits.getdata(filelist[0])
        #This gets the dimensions of the first image
        imsize_y, imsize_x = first_frame_data.shape
        #This creates a new array of the same dimensions as the first image
        fits_stack = np.zeros((imsize_y, imsize_x , n), dtype = np.float32) 
        #Establishes a blank/zeros array
        for ii in range(0, n):
            im = fits.getdata(filelist[ii])
            fits_stack[:,:,ii] = im
            #It iterates through both dimensions and puts all the data from the raw images into the blank/zero array
        med_frame = np.nanmedian(fits_stack, axis=2)
        fits.writeto(shortname[0]+cut+'Median_Combined_'+filt+'.fit', med_frame, header, clobber=True)
        

        return 
    
    #-----------------------------------------------------------
    
    i_data = glob.glob(procdir+targname+'\\Iband\\*I.fit')
    v_data = glob.glob(procdir+targname+'\\Vband\\*V.fit')

    if (filt=='I'):
            imlist = i_data
            data_cut = 'Iband\\'
            #path to r flat here
    else:
        if (filt=='V'):
            imlist = v_data
            data_cut = 'Vband\\'
                #path to v flat here
                
    for filename in imlist:
        bias_subtract(filename, master_bias_path, data_cut)
        
    i_bias_data = glob.glob(procdir+targname+'\\Iband\\b_*_I.fit')
    v_bias_data = glob.glob(procdir+targname+'\\Vband\\b_*_V.fit')
    
    if (filt=='I'):
            bias_data = i_bias_data
            data_cut = 'Iband\\'
            #path to r flat here
    else:
        if (filt=='V'):
            bias_data = v_bias_data
            data_cut = 'Vband\\'
                #path to v flat here
    
    for filename in bias_data:
        dark_subtract(filename, new_master_dark_path, data_cut)
    
    i_db_data = glob.glob(procdir+targname+'\\Iband\\db_*_I.fit')
    v_db_data = glob.glob(procdir+targname+'\\Vband\\db_*_V.fit')
    
    if (filt=='I'):
            db_data = i_db_data
            data_cut = 'Iband\\'
            #path to r flat here
    else:
        if (filt=='V'):
            db_data = v_db_data
            data_cut = 'Vband\\'
                #path to v flat here
    
    for filename in db_data:
        flat_divide(filename, master_flat_path, data_cut)
        
    i_fdb_data = glob.glob(procdir+targname+'\\Iband\\fdb_*_I.fit')
    v_fdb_data = glob.glob(procdir+targname+'\\Vband\\fdb_*_V.fit')
    
    if (filt=='I'):
            fdb_data = i_fdb_data
            data_cut = 'Iband\\'
            #path to r flat here
    else:
        if (filt=='V'):
            fdb_data = v_fdb_data
            data_cut = 'Vband\\'
                #path to v flat here

    offsetx, offsety = cross_image_ME(coord, background, fdb_data)
    print(offsetx)
    print(offsety)
    shift_function_ME(offsetx, offsety, padding, fdb_data,data_cut)
    
    
    
    
    
    