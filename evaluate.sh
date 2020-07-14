#!/bin/bash

SRC_PATH='./Data/50_files.src'
GOLD_PATH='./Data/50_files.gold'
CAND_PATH='./Data/50_files.cand'
SRL_PATH='./Data/50_files.srl'

python evaluate.py \
	-src_path ${SRC_PATH} \
	-gold_path ${GOLD_PATH} \
	-cand_path ${CAND_PATH} \
	-srl_path ${SRL_PATH} \
