#!/bin/sh

#python c3o.py Sort 200 14000 
#python c3o.py Grep 160 20000 0.5
python c3o.py SGDLR 120 200000000 5 30
python c3o.py K-Means 300 20000 5 8
python c3o.py 'Page Rank' 330 2000000 3000000 0.0007
