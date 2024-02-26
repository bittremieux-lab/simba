import numpy as np
import random

class SelfSupervision:
    @staticmethod
    def modify_peaks(data_sample, prob_peaks=0.15):


        batch_size= data_sample['mz_0'].shape[0]

        # select the flips per row
        number_peaks_sampled_per_row= ((prob_peaks)*data_sample['number_peaks']).astype(int)



        print(number_peaks_sampled_per_row)
        print(data_sample['number_peaks'])
        peaks_sampled = [np.random.choice(total_peaks, size=n) for n,total_peaks in zip(number_peaks_sampled_per_row, data_sample['number_peaks'])]
        peaks_sampled = [[(i,p) for p in p_list] for i,p_list in enumerate(peaks_sampled)]


        # divide between no flip and flip
        half_size=int(len(peaks_sampled)/2)
        no_flip_peaks = [p[0:half_size] for p in peaks_sampled ]
        flip_peaks = [p[half_size:] for p in peaks_sampled ]

        #flatten
        no_flip_peaks = [item for sublist in no_flip_peaks for item in sublist]
        flip_peaks = [item for sublist in flip_peaks for item in sublist]
        
        
        # Make a copy of the original list
        flip_peaks_flipped = flip_peaks.copy()

        # Shuffle the copied list
        random.shuffle(flip_peaks_flipped)


        # create a flipped list
        flip_peaks_flipped = [ (p0[0],p0[1],p1[0],p1[1])  for p0,p1 in zip(flip_peaks, flip_peaks_flipped)]

        
        output_mz = np.zeros((batch_size, 100 )  )
        output_intensity = np.zeros((batch_size, 100 )  )
        sample_mask = np.zeros((batch_size,100))
        # no interchange the intensities
        for peak_index in (no_flip_peaks):
            i=peak_index[0]
            p=peak_index[1]
            output_mz[i,p] = data_sample[i,'mz_0'][p]
            output_intensity[i,p] = data_sample[i,'intensity_0'][p]
            sample_mask[i,p]=1


        # no interchange the intensities
        for peak_index in (flip_peaks_flipped):
            i0=peak_index[0]
            p0=peak_index[1]
            i1=peak_index[2]
            p1=peak_index[3]
            output_mz[i0,p0] = data_sample['mz_0'][i1, p1]
            output_intensity[i0,p0] = data_sample['intensity_0'][i1, p1]
        
        # modify the flips array accordingly
        data_sample['sampled_mz']=output_mz
        data_sample['sampled_intensity']=output_intensity
        data_sample['no_flip_peaks']=no_flip_peaks
        data_sample['flip_peaks'] = flip_peaks_flipped
        data_sample['flip_mask']= sample_mask #1 if the peak is not exchanged
        
        return data_sample