/tmp2/hhchen/CPR_paper/conda/bin/python ./BiTGCF/main.py \
	--data_path ../input \
	--source_dataset all_vod \
	--target_dataset tv \
	--epoch 200 \
	--batch_size 65536 \
	--embed_size 25

/tmp2/hhchen/CPR_paper/conda/bin/python ./BiTGCF/main.py \
	--data_path ../input \
	--source_dataset all_vod \
	--target_dataset cold_tv \
	--epoch 200 \
	--batch_size 65536 \
	--embed_size 25
