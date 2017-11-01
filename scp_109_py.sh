#!/bin/bash

# set timeout 30

# spawn scp -r pyscripts liaoshanhe@10.60.43.109:/home/liaoshanhe/RecoSys
# expect "*assword*"

# send "lsh123\r"
# interact

scp -i ~/.ssh/id_rsa -r pyscripts liaoshanhe@10.60.43.109:/home/liaoshanhe/RecoSys
ssh -i ~/.ssh/id_rsa liaoshanhe@10.60.43.109 "mv /home/liaoshanhe/RecoSys/pyscripts/* /home/liaoshanhe/RecoSys/; rmdir /home/liaoshanhe/RecoSys/pyscripts"