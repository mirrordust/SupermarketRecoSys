#!/usr/bin/expect

set timeout 30

spawn scp -r script root@10.60.43.111:/home/reco/data-171021
expect "*assword*"

send "iDataLab\r"
interact