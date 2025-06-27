# Object-level Cross-view Geo-localization with Location Enhancement and Multi-Head Cross Attention

# Benchmark Dataset
The benchmark dataset we use is [Cross-view Object Geo-localization](https://github.com/sunyuxi/DetGeo)

The download link is [CVOGL](https://drive.google.com/file/d/1WCwnK_rrU--ZOIQtmaKdR0TXcmtzU4cf/view?usp=sharing) 

# Dependencies and Installation
1. Clone repo
```
git clone https://github.com/ZheyangH/OCGNet.git
cd OCGNet
```
2. Install dependent packages
```
conda create -n ocgnet python=3.10 -y
conda activate ocgnet
pip install --upgrade pip
pip install -r requirements.txt
```
# Few-shot
We have annotated a batch of few-shot samples. The usage method is as follows:
1. The download link is [CVOGL-fewshot](https://drive.google.com/drive/folders/1jJdB9erIQg2aLDwvG-rfIdib-sohKAyr?usp=sharing)
2. Place the two .pth files in the following path: 'data/CVOGL_DroneAerial'.

# Train
1. Download the CVOGL dataset and rename the folder to 'data', resulting in 'data/CVOGL_DroneAerial' and 'data/CVOGL_SVI'.
2. Download the pretrained Yolov3 model and place it in the 'saved_models' directory as './saved_models/yolov3.weights'.
3. Run the following command to train the Drone to Satellite model.
```
python src/train.py --emb_size 512 --img_size 1024 --max_epoch 25 --data_root data --data_name CVOGL_DroneAerial --beta 1.0 --savename model_droneaerial --gpu 0 --batch_size 12 --num_workers 6 --print_freq 50 > logs/train_droneaerial.log 2>&1 
```
4. Run the following command to train the Ground to Satellite model.
```
python train.py --emb_size 512 --img_size 1024 --max_epoch 25 --data_root data --data_name CVOGL_SVI --beta 1.0 --savename model_svi --gpu 0,1 --batch_size 12 --num_workers 6 --print_freq 50 > logs/train_svi.log 2>&1 
```
5. Run the following command to train the Drone to Satellite model on few-shot task, use pretrain model on CVOGL "saved_models/model_droneaerial_model_best.pth.tar".
```
python train.py --few_shot_train --pretrain saved_models/model_droneaerial_model_best.pth.tar --emb_size 512 --img_size 1024 --max_epoch 20 --data_root data --data_name CVOGL_DroneAerial --beta 1.0 --savename model_droneaerial_few_shot --gpu 0 --batch_size 6 --num_workers 6 --print_freq 5
```
# Evaluation
1. Run the following command to evaluate the Drone to Satellite model for validation and testing on the CVOGL.
```
python train.py --val --pretrain saved_models/model_droneaerial_model_best.pth.tar --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --savename test_model_droneaerial --gpu 0 --batch_size 8 --num_workers 16 --print_freq 50

python train.py --test --pretrain saved_models/model_droneaerial_model_best.pth.tar --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --savename test_model_droneaerial --gpu 0 --batch_size 8 --num_workers 16 --print_freq 50
```
2. Run the following command to evaluate the Ground to Satellite model for validation and testing on the CVOGL.
```
python train.py --val --pretrain saved_models/model_svi_model_best.pth.tar --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_SVI --savename test_model_svi --gpu 0 --batch_size 8 --num_workers 16 --print_freq 50

python train.py --test --pretrain saved_models/model_svi_model_best.pth.tar --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_SVI --savename test_model_svi --gpu 0 --batch_size 8 --num_workers 16 --print_freq 
```
3. Run the following command to evaluate the Drone to Satellite model for validation and testing on the CVOGL few-shot.
```
python train.py --few_shot_test --pretrain saved_models/model_droneaerial_few_shot_best.pth.tar --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --savename few_shot_model_DetGeo_droneaerial --gpu 0 --batch_size 8 --num_workers 16 --print_freq 5
```

# Gradio demo for Drone to Satellite
We provide two Drone to satellite OCGNet [ckpt1](https://drive.google.com/file/d/1djeXyPwfjLlqE3STJVbdIqKx7oad7QPg/view?usp=sharing) with a standard deviation of 0.1 and [ckpt2](https://drive.google.com/file/d/1ZoFyF4uhuBwfzNVHcp4kwMpd2rlEZJNl/view?usp=drive_link) with a standard deviation of 0.075 for download, place the .pth file in the following path: 'saved_models/'. Meanwhile, place the [DetGeo ckpt](https://drive.google.com/file/d/1UbNQ7bfxX4356jyCrzqfv184aWbT2ZyW/view?usp=sharing) into same path.
Run the following command to use the Gradio UI.
```
python src/demo.py
```

# Others
1. acknowledge the work from [Cross-view Object Geo-localization](https://github.com/sunyuxi/DetGeo) and [YoloV3](https://github.com/patrick013/Object-Detection---Yolov3)
2. This research is supported by the Australian Research Council (ARC).
3. For any questions or issues, open an issue or contact Zheyang Huang(jasonhuang1999cn@gmail.com)
