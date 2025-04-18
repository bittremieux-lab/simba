import numpy as np
import random
import math


class SelfSupervision:
    @staticmethod
    def modify_peaks(
        data_sample, data_total, prob_peaks=0.15, max_peaks=100, prop_no_flips=0.5
    ):
        """
        it receives a dara row, and the total dataset. It must apply the sampling for selecting 15% of the peaks for training.
        """

        # print(f'no_flip_peaks')
        # print(no_flip_peaks)

        # print(f'flip_peaks')
        # print(flip_peaks)

        # create vector of output
        output_mz = np.zeros((max_peaks), dtype=np.float32)
        output_intensity = np.zeros((max_peaks), dtype=np.float32)
        output_mask = np.zeros((max_peaks), dtype=np.int32)

        # select the number of samples taken
        number_peaks = data_sample["number_peaks"][0]
        if number_peaks >= 2:
            number_peaks_sampled = ((prob_peaks) * number_peaks).astype(int)
            # peak at least 2
            number_peaks_sampled = max(number_peaks_sampled, 2)
            # Select the peaks
            peaks_sampled = np.random.choice(
                number_peaks, size=number_peaks_sampled, replace=False
            )

            # print(f'peaks_sampled')
            # print(peaks_sampled)

            # divide between no flip and flip
            no_flip_length = int(prop_no_flips * len(peaks_sampled))
            no_flip_peaks = peaks_sampled[0:no_flip_length]
            flip_peaks = peaks_sampled[no_flip_length:]

            # get mz and intensity
            new_mz = data_sample["mz_0"].copy()[0:number_peaks]
            new_intensity = data_sample["intensity_0"].copy()[0:number_peaks]
            new_mask = np.zeros(number_peaks, dtype=np.int32)

            # for the no flip peaks we only set the mask with the correspoding no flip code=1
            new_mask[no_flip_peaks] = 1
            # no_flip_mz= data_sample['mz_0'][no_flip_peaks]
            # no_flip_int= data_sample['intensity_0'][no_flip_peaks]
            # no_flip_mask = 1*np.ones(len(no_flip_peaks))  # 1 if it was selected as no flip

            # interchange the intensities
            # flip_mz= data_sample['mz_0'][flip_peaks]  # the MZ values are not interchanged
            flip_mz, flip_int = SelfSupervision.get_random_peaks(
                data_total, len(flip_peaks)
            )
            # flip_mask = 2*np.ones(len(flip_peaks)) # 2 if it was selected as flip
            new_mz[flip_peaks] = flip_mz
            new_intensity[flip_peaks] = flip_int
            new_mask[flip_peaks] = 2

            # concatenate
            # mz = np.concatenate([no_flip_mz, flip_mz])
            # intensity = np.concatenate([no_flip_int, flip_int])
            # mask = np.concatenate([no_flip_mask,flip_mask], axis=0)

            # order by mz

            ordered_indexes = np.argsort(new_mz)
            new_mz = new_mz[ordered_indexes]
            new_intensity = new_intensity[ordered_indexes]
            new_mask = new_mask[ordered_indexes]

            # assign values
            # output_mz[no_flip_peaks] = no_flip_mz
            # output_intensity[no_flip_peaks] = no_flip_int
            # sample_mask[no_flip_peaks]=1
            # print('sample_mask')
            # print(sample_mask)
            # assign values
            # output_mz[flip_peaks] = flip_mz
            # output_intensity[flip_peaks] = flip_int
            # sample_mask[flip_peaks]=0

            # print('sample_mask')
            # print(sample_mask)
            # normalize intensity

            # assign values
            output_mz[0:number_peaks] = new_mz
            output_intensity[0:number_peaks] = new_intensity
            output_mask[0:number_peaks] = new_mask

            # normalization
            output_intensity = output_intensity / np.sqrt(
                np.sum(output_intensity**2, axis=0, keepdims=True)
            )

        # modify the flips array accordingly
        data_sample["sampled_mz"] = output_mz
        data_sample["sampled_intensity"] = output_intensity
        # data_sample['no_flip_peaks_indexes']=no_flip_peaks
        # data_sample['flip_peaks_indexes'] = flip_peaks
        data_sample["flips"] = output_mask  # 1 if the peak is not exchanged

        return data_sample

    @staticmethod
    def get_random_peaks(data_total, n_spectrums):
        """
        it returns a random set of peaks from the dataset
        """
        # interchange the intensities
        # get random spectrums from the data
        random_spectrum_indexes = np.random.choice(
            data_total["mz_0"].shape[0], size=n_spectrums
        )

        # first value: index, second value: peak
        random_peak_pairs = [
            (r, np.random.choice(data_total["number_peaks"][r][0]))
            for r in random_spectrum_indexes
        ]

        # get mz, intensity
        intensity = np.array(
            [data_total["intensity_0"][peak[0], peak[1]] for peak in random_peak_pairs]
        )
        mz = np.array(
            [data_total["mz_0"][peak[0], peak[1]] for peak in random_peak_pairs]
        )

        return mz, intensity
