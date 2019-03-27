#!/bin/bash

# monitor & aplicacao
$1 >> $2 & eval $3
id=$(echo $!)
kill $id
