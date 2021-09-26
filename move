#! /data/data/com.termux/files/usr/bin/sh

OLDDIR=/storage/emulated/0/DCIM/

if [ $# -ne 1 ]
then
    echo "Need a directory as an argument."
    exit
fi

if ! [ -d digits/$1 ]
then
    echo "$1 Must be a numeric directory."
    exit
fi

mv $OLDDIR/$1/* digits/$1
