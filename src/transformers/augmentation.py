


class Augmentation:

     @staticmethod
     def augment(data_sample):
          
          new_sample= Augmentation.inversion(data_sample)

          return new_sample
     
     @staticmethod
     def inversion(data_sample):
          # inversion

          new_sample={}
          new_sample["mz_0"]=data_sample['mz_1']
          new_sample["mz_1"]= data_sample['mz_0']

          new_sample["intensity_0"]=data_sample["intensity_1"]
          new_sample["intensity_1"]=data_sample["intensity_0"]

          new_sample['precursor_mass_0']= data_sample['precursor_mass_1']
          new_sample['precursor_mass_1']= data_sample['precursor_mass_0']

          new_sample['precursor_charge_0']= data_sample['precursor_charge_1']
          new_sample['precursor_charge_1']= data_sample['precursor_charge_0']

          new_sample['similarity']=data_sample['similarity']
          return new_sample
