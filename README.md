2024.4.29 The code and datasets are released.

More details will be added in the coming days.

## Datasets

We use lmdb dataset for training and evaluation dataset.
The datasets can be downloaded in [clova (for validation and evaluation)](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) and [ABINet (for training and evaluation)](https://github.com/FangShancheng/ABINet#datasets).

* Training datasets
    * [MJSynth (MJ)](https://www.robots.ox.ac.uk/~vgg/data/text/)
    * [SynthText (ST)](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)

* Evaluation datasets
    * Regular datasets 
        * [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) (IIIT)
        * [Street View Text](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) (SVT)
        * [ICDAR2013](https://rrc.cvc.uab.es/?ch=2): IC13<sub>S</sub> with 857 images, IC13<sub>L</sub> with 1015 images
    * Irregular dataset
        * [ICDAR2015](https://rrc.cvc.uab.es/?ch=4): IC15<sub>S</sub> with 1811 images, IC15<sub>L</sub> with 2077 images
        * [Street View Text Perspective](https://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf) (SVTP)
        * [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html) (CUTE)

* Training
```
python main.py --config=configs/pretrain_busnet.yaml
python main.py --config=configs/finetune_busnet.yaml
```

## Acknowledgements
This implementation has been based on [ABINet](https://github.com/FangShancheng/ABINet).
