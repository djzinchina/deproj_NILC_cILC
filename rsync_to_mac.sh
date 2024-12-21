#!/bin/bash


des="djz@192.168.4.249:~/Downloads/beamsys/"
src="/home/doujzh/Documents/AliCPT_beamsys/output/"

rsync -azvrP --progress --copy-links --rsh="/usr/bin/sshpass -p 690109 ssh -o StrictHostKeyChecking=no -l djz" $src  $des
des="djz@192.168.4.249:~/Downloads/beamsys/"
src="/media/doujzh/AliCPT_data2/Zirui_beamsys/Mask"

rsync -azvrP --progress --copy-links --exclude *.fits --rsh="/usr/bin/sshpass -p 690109 ssh -o StrictHostKeyChecking=no -l djz" $src  $des
