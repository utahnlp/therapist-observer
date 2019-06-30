sed -i "s/''/'/g" all_u_lc.txt
sed -i -E "s/\(inaudible\)//g" all_u_lc.txt
sed -r -i "s/\[[0-9]+:[0-9]+*\]//g" all_u_lc.txt
sed -i -E "s/\(ph\)//g" all_u_lc.txt
