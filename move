#! /data/data/com.termux/files/usr/bin/sh

OLDDIR=/storage/emulated/0/DCIM/digits

if [ $# -ne 1 ]
then
    echo "Need a directory as an argument."
    exit
fi

if ! [ -d $1 ]
then
    echo "$1 Must be a numeric directory."
    exit
fi

mv $OLDDIR/* digits/$1
