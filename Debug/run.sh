echo "[LOG] Get srl..."
python get_srl.py srl
echo "[LOG] Get tree..."
python get_srl.py tree
echo "[LOG] Get highlight..."
python get_hl.py
echo "[LOG] Get correlation scores..."
python get_score.py
echo "[LOG] Get files for plots..."
python for_plot.py
