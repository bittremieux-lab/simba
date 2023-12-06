## GLOBUS:

Download the server, login and run it in the background

wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
 tar xzf globusconnectpersonal-latest.tgzwget 


./globusconnectpersonal

 ./globusconnectpersonal -start &

 globus transfer dff8c41a-9419-11ee-83dc-d5484943e99a:/user/antwerpen/209/vsc20939/best_model.ckpt b7f4c648-9415-11ee-be2c-c52a29481bea:~/best_model.cpkt



## VSC

Run an interactive session:

srun -p ampere_gpu --gpus=1 --pty bash
