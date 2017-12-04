#!/bin/sh
#{
#python rfclassification.py -s 200 -d amt0svd -a 0 -m 1 > amt0svd.out
#} &
#{
#python rfclassification.py -s 200 -d amt100svd -a 100 -m 1 > amt100svd.out
#} &
#{
#python rfclassification.py -s 200 -d amt300svd -a 300 -m 1 > amt300svd.out
#} &

{
python rfclassification.py -s 200 -d amt200svd -a 200 -m 1 > amt200svd.out
} &

{
python rfclassification.py -s 200 -d amt400svd -a 400 -m 1 > amt400svd.out
} &

{
python rfclassification.py -s 200 -d amt500svd -a 500 -m 1 > amt500svd.out
} &

{
python rfclassification.py -s 200 -d amt600svd -a 600 -m 1 > amt600svd.out
} &

{
python rfclassification.py -s 200 -d plu1svd -p 1 -m 1 > plu1svd.out
} &

{
python rfclassification.py -s 200 -d plu2svd -p 2 -m 1 > plu2svd.out
} &

{
python rfclassification.py -s 200 -d plu6svd -p 6 -m 1 > plu6svd.out
} &

{
python rfclassification.py -s 200 -d plu11svd -p 11 -m 1 > plu11svd.out
} &

{
python rfclassification.py -s 200 -d plu20svd -p 20 -m 1 > plu20svd.out
} &

echo 'all task start...'
