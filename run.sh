
# tag emotion_recognition subject_identification
# model eegnet eegchannelnet syncnet


python train.py --task subject_identification --tag my_eegnet --model agnn --epochs 1500 --lr 0.001

# python train.py --task subject_identification --tag subject_identification --model eegchannelnet &

# python train.py --task subject_identification --tag subject_identification --model syncnet 


# python train.py --task emotion_recognition --tag emotion_recognition --model eegnet &

# python train.py --task emotion_recognition --tag emotion_recognition --model eegchannelnet &

# python train.py --task emotion_recognition --tag emotion_recognition --model syncnet