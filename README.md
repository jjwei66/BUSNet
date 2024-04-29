2024.4.29 The code is released.
More details will be added in the coming days.

## Datasets

We use lmdb dataset for training and evaluation dataset.
The datasets can be downloaded in [clova (for validation and evaluation)](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) and [ABINet (for training and evaluation)](https://github.com/FangShancheng/ABINet#datasets).

* Training datasets
    * [MJSynth (MJ)](https://www.robots.ox.ac.uk/~vgg/data/text/)
    * [SynthText (ST)](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
    * [WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)
* Validation datasets
    * The union of the training set of [ICDAR2013](https://rrc.cvc.uab.es/?ch=2), [ICDAR2015](https://rrc.cvc.uab.es/?ch=4), [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), and [Street View Text](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
* Evaluation datasets
    * Regular datasets 
        * [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) (IIIT)
        * [Street View Text](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) (SVT)
        * [ICDAR2013](https://rrc.cvc.uab.es/?ch=2): IC13<sub>S</sub> with 857 images, IC13<sub>L</sub> with 1015 images
    * Irregular dataset
        * [ICDAR2015](https://rrc.cvc.uab.es/?ch=4): IC15<sub>S</sub> with 1811 images, IC15<sub>L</sub> with 2077 images
        * [Street View Text Perspective](https://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf) (SVTP)
        * [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html) (CUTE)
* Tree structure of `data` directory
    ```
    data
    ├── charset_36.txt
    ├── evaluation
    │   ├── CUTE80
    │   ├── IC13_857
    │   ├── IC13_1015
    │   ├── IC15_1811
    │   ├── IC15_2077
    │   ├── IIIT5k_3000
    │   ├── SVT
    │   └── SVTP
    ├── training
    │   ├── MJ
    │   │   ├── MJ_test
    │   │   ├── MJ_train
    │   │   └── MJ_valid
    │   └── ST
    ├── validation
    ├── WikiText-103.csv
    └── WikiText-103_eval_d1.csv
    ```
