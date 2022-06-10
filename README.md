# quantizer
Requirements:
  pytorch 1.10.0
  numpy
  open3d
  
Usage:
python quantizer.py --root_dir=../dataset_npy --target_dir=../dataset_ori_9 --original_precision=10 --target_precision=9 --original_data_format=ply --target_data_format=npy --sequences="dancer exercise basketballplayer model"
