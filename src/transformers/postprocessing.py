import torch.nn.functional as F


class Postprocessing:

    # calculate loss
    @staticmethod
    def compute_cosine_similarity(predictions):
        cosine_similarity = []
        for p in predictions:
            emb0 = p[0, 0:64]
            emb1 = p[0, 64:]
            emb0 = emb0.view(1, -1)
            emb1 = emb1.view(1, -1)
            cos_sim = F.cosine_similarity(emb0, emb1)
            cosine_similarity.append(float(cos_sim[0]))
        return cosine_similarity

    @staticmethod
    def get_similarities(dataloader):
        # calculate similarity
        similarities = []
        for batch in dataloader:
            # similarities.append(float(b['similarity'][0]))
            sim_temp = [float(b) for b in batch["similarity"]]
            # similarities = similarities  +  [b for b in batch]
            similarities = similarities + sim_temp
        return similarities


    @staticmethod
    def get_similarities_multitasking(dataloader):
        # calculate similarity
        #similarities1 = []
        #similarities2 = []
        #for batch in dataloader:
        #    # similarities.append(float(b['similarity'][0]))
        #    sim_temp1 = [float(b) for b in batch["similarity"]]
        #    # similarities = similarities  +  [b for b in batch]
        #    similarities1 = similarities1 + sim_temp1

        #    sim_temp2 = [float(b) for b in batch["similarity2"]]
        #    similarities2 = similarities2 + sim_temp2
            
        similarities1 = [[float(b) for b in batch['similarity']] for batch in dataloader]
        similarities2 = [[float(b) for b in batch['similarity2']] for batch in dataloader]

        return similarities1, similarities2