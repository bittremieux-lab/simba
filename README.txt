## GPU problem solving

Make sure that pytorch library is not cpu one

conda remove pytorch cudatoolkit
conda clean --all

3)Instal things separately and activating tensorflow-

conda install -c anaconda cudatoolkit (11.8)

4)Instal PyTorch (GPU version compatible with CUDA verison):

conda install pytorch=2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

## Wetransfer

wget --user-agent Mozilla/4.0 '[your big address here]' -O dest_file_name

## GLOBUS:

Download the server, login and run it in the background

wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
 tar xzf globusconnectpersonal-latest.tgzwget 


./globusconnectpersonal

 ./globusconnectpersonal -start &

 globus transfer dff8c41a-9419-11ee-83dc-d5484943e99a:/user/antwerpen/209/vsc20939/best_model_20231207.cpkt ddb59aef-6d04-11e5-ba46-22000b92c6ec:~/best_model_gpu_20231207.cpkt



## VSC

Run an interactive session:

srun -p ampere_gpu --gpus=1 --pty bash
