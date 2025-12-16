# Object-level Cross-view Geo-localization with Location Enhancement and Multi-Head Cross Attention
[Paper link](https://doi.org/10.1109/JSTARS.2025.3603506)
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

# Gradio demo for Drone to Satellite and different backbones
We provide two Drone to satellite OCGNet [ckpt1](https://drive.google.com/file/d/1djeXyPwfjLlqE3STJVbdIqKx7oad7QPg/view?usp=sharing) with a standard deviation of 0.1 and [ckpt2](https://drive.google.com/file/d/1ZoFyF4uhuBwfzNVHcp4kwMpd2rlEZJNl/view?usp=drive_link) with a standard deviation of 0.075 for download, place the .pth file in the following path: 'saved_models/'. Meanwhile, place the [DetGeo ckpt](https://drive.google.com/file/d/1UbNQ7bfxX4356jyCrzqfv184aWbT2ZyW/view?usp=sharing) into same path.
For the different backbones, we provide the ConvNeXT-Tiny version, [drone]([https://drive.google.com/file/d/1djeXyPwfjLlqE3STJVbdIqKx7oad7QPg/view?usp=sharing](https://drive.google.com/file/d/1duJ-WmuEsyv6PReNe5HNPHUfvg0LLrC3/view?usp=sharing)) and [svi]([https://drive.google.com/file/d/1djeXyPwfjLlqE3STJVbdIqKx7oad7QPg/view?usp=sharing](https://drive.google.com/file/d/1uv_SURFVVeVfyzeO_U2IsaPprWr0P_km/view?usp=sharing))
Replace the resnet18 code in OCGNet.py file
```
class MyConvNeXt(nn.Module):
    def __init__(self):
        super(MyConvNeXt, self).__init__()
        # Load pretrained convnext_tiny
        self.base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        self.early_stem = nn.Sequential(
            self.base_model.features[0],  # stem: [B, 96, 64, 64]
        )
        
        self.forward_features = nn.Sequential(
            *self.base_model.features[1:]  # [B, 768, 8, 8]
        )
        
        self.projector = nn.Conv2d(768, 512, kernel_size=1)

    def forward(self, x):
        early_features = self.early_stem(x)           # [B, 96, 64, 64]
        x = self.forward_features(early_features)     # [B, 768, 8, 8]
        x = self.projector(x)                         # [B, 512, 8, 8]
        return x, early_features
 
```
<img width="1310" height="633" alt="image" src="https://github.com/user-attachments/assets/4765a105-1a71-4afe-9478-4f50d73b3d3e" />
<img width="1260" height="638" alt="image" src="https://github.com/user-attachments/assets/664bf322-fff7-4bfd-be0d-899cf891aab0" />

Run the following command to use the Gradio UI, you can click the query image to generate query-image with click-point.

```
python src/demo.py
```

Cite this paper
```
@ARTICLE{11142913,
  author={Huang, Zheyang and Aryal, Jagannath and Nahavandi, Saeid and Lu, Xuequan and Lim, Chee Peng and Wei, Lei and Zhou, Hailing},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Object-Level Cross-View Geolocalization With Location Enhancement and Multihead Cross Attention}, 
  year={2025},
  volume={18},
  number={},
  pages={22880-22890},
  keywords={Location awareness;Satellite images;Drones;Feature extraction;Australia;Accuracy;Satellites;Kernel;Standards;Object recognition;Attention;cross-view matching;geolocalization;object detection},
  doi={10.1109/JSTARS.2025.3603506}}
```

# Others
1. acknowledge the work from [Cross-view Object Geo-localization](https://github.com/sunyuxi/DetGeo) and [YoloV3](https://github.com/patrick013/Object-Detection---Yolov3)
2. This research is supported by the Australian Research Council (ARC).
3. For any questions or issues, open an issue or contact Zheyang Huang(jasonhuang1999cn@gmail.com)
