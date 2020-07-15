#!/bin/bash

SRC_PATH='./Data/50_files.src'
GOLD_PATH='./Data/50_files.gold'
CAND_PATH='./Data/50_files.cand'
TREE_PATH='./Data/bert.tree'

python evaluate.py \
	-src_path ${SRC_PATH} \
	-gold_path ${GOLD_PATH} \
	-cand_path ${CAND_PATH} \
	-tree_path ${TREE_PATH} \
	-run_srl False \
	-run_tree False
