#!/bin/bash


des="djz@192.168.4.81:~/Downloads/plots_paper/"
src="/home/doujzh/Documents/djzfiles/plots_paper/"

rsync -azvrP --progress --exclude *.fits --copy-links --rsh="/usr/bin/sshpass -p 690109 ssh -o StrictHostKeyChecking=no -l djz" $src  $des
