bash run_multi.sh hk csjj 0 traineval&
sleep 300s && bash run_multi.sh spo csj 1 traineval&
sleep 600s && bash run_multi.sh mt b 2 traineval
