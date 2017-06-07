CUDA_VISIBLE_DEVICES=2 bazel-bin/magenta/models/nsynth/gan/main \
--hps=magenta/models/nsynth/gan/hps3.json \
--mode=train --total_batch_size=8 --num_gpus=1 \
--src_wav_path=/cmsdata/ssd1/cmslab/NSynth_Dataset/GAN_dataset_Brass_and_String/train/String_train_partitioned_1 \
--trg_wav_path=/cmsdata/ssd1/cmslab/NSynth_Dataset/GAN_dataset_Brass_and_String/train/Brass_train_partitioned_1 \
--pretrain_path=pretrain_conv_20_logs \
--train_path=train_hps3_logs \
--log_period=10 \
--ckpt_period=100 \
|& tee ~/train_hps3_log
