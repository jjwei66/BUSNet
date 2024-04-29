import logging
import re

import cv2
import lmdb
import six
from fastai.vision import *
from torchvision import transforms

from utils import CharsetMapper, onehot
from augment import rand_augment_transform


class ImageDataset(Dataset):
    "`ImageDataset` read data from LMDB database."

    def __init__(self,
                 path: PathOrStr,
                 is_training: bool = True,
                 img_h: int = 32,
                 img_w: int = 128,
                 max_length: int = 25,
                 check_length: bool = True,
                 case_sensitive: bool = False,
                 charset_path: str = 'data/charset_36.txt',
                 convert_mode: str = 'RGB',
                 data_aug: bool = True,
                 multiscales: bool = True,
                 one_hot_y: bool = True,
                 return_idx: bool = False,
                 return_raw: bool = False,
                 use_sm=False,
                 **kwargs):
        self.path, self.name = Path(path), Path(path).name
        assert self.path.is_dir() and self.path.exists(), f"{path} is not a valid directory."
        self.convert_mode, self.check_length = convert_mode, check_length
        self.img_h, self.img_w = img_h, img_w
        self.max_length, self.one_hot_y = max_length, one_hot_y
        self.return_idx, self.return_raw = return_idx, return_raw
        self.case_sensitive, self.is_training = case_sensitive, is_training
        self.data_aug, self.multiscales = data_aug, multiscales
        self.charset = CharsetMapper(charset_path, max_length=max_length + 1)
        self.c = self.charset.num_classes
        self.use_sm = use_sm
        if self.is_training and self.use_sm: self.sm = SpellingMutation(pn0=0.5, charset=self.charset)

        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {path}.'
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode()))

        if self.is_training and self.data_aug:
            self.augment_tfs = transforms.Compose([
                rand_augment_transform()
            ])
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return self.length

    def _next_image(self, index):
        next_index = random.randint(0, len(self) - 1)
        return self.get(next_index)

    def _check_image(self, x, pixels=6):
        if x.size[0] <= pixels or x.size[1] <= pixels:
            return False
        else:
            return True

    def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT):
        def _resize_ratio(img, ratio, fix_h=True):
            if ratio * self.img_w < self.img_h:
                if fix_h:
                    trg_h = self.img_h
                else:
                    trg_h = int(ratio * self.img_w)
                trg_w = self.img_w
            else:
                trg_h, trg_w = self.img_h, int(self.img_h / ratio)
            img = cv2.resize(img, (trg_w, trg_h))
            pad_h, pad_w = (self.img_h - trg_h) / 2, (self.img_w - trg_w) / 2
            top, bottom = math.ceil(pad_h), math.floor(pad_h)
            left, right = math.ceil(pad_w), math.floor(pad_w)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            return img

        if self.is_training:
            if random.random() < 0.5:
                base, maxh, maxw = self.img_h, self.img_h, self.img_w
                h, w = random.randint(base, maxh), random.randint(base, maxw)
                return _resize_ratio(img, h / w)
            else:
                return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio
        else:
            return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio

    def resize(self, img):
        if self.multiscales:
            return self.resize_multiscales(img, cv2.BORDER_REPLICATE)
        else:
            return cv2.resize(img, (self.img_w, self.img_h))

    def get(self, idx):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
            try:
                label = str(txn.get(label_key.encode()), 'utf-8')  # label
                label = re.sub('[^0-9a-zA-Z]+', '', label)
                if self.check_length and self.max_length > 0:
                    if len(label) > self.max_length or len(label) <= 0:
                        # logging.info(f'Long or short text image is found: {self.name}, {idx}, {label}, {len(label)}')
                        return self._next_image(idx)
                label = label[:self.max_length]

                imgbuf = txn.get(image_key.encode())  # image
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
                    image = PIL.Image.open(buf).convert(self.convert_mode)
                if self.is_training and not self._check_image(image):
                    # logging.info(f'Invalid image is found: {self.name}, {idx}, {label}, {len(label)}')
                    return self._next_image(idx)
            except:
                import traceback
                traceback.print_exc()
                logging.info(f'Corrupted image is found: {self.name}, {idx}, {label}, {len(label)}')
                return self._next_image(idx)
            return image, label, idx

    def _process_training(self, image):
        if self.data_aug: image = self.augment_tfs(image)
        image = self.resize(np.array(image))
        return image

    def _process_test(self, image):
        return self.resize(np.array(image))  # TODO:move is_training to here

    def __getitem__(self, idx):
        image, text, idx_new = self.get(idx)
        # if not self.is_training: assert idx == idx_new, f'idx {idx} != idx_new {idx_new} during testing.'

        if self.is_training:
            image = self._process_training(image)
        else:
            image = self._process_test(image)
        if self.return_raw: return image, text
        image = self.totensor(image)

        length = tensor(len(text) + 1).to(dtype=torch.long)  # one for end token
        label = self.charset.get_labels(text, case_sensitive=self.case_sensitive)
        label = tensor(label).to(dtype=torch.long)
        if self.one_hot_y: label = onehot(label, self.charset.num_classes)

        if self.return_idx:
            y = [label, length, idx_new]
        else:
            y = [label, length]

        if self.is_training and self.use_sm:
            text_x = self.sm(text)

            length_x = tensor(len(text_x) + 1).to(dtype=torch.long)  # one for end token
            label_x = self.charset.get_labels(text, case_sensitive=self.case_sensitive)
            label_x = tensor(label_x).to(dtype=torch.long)
            if self.one_hot_y: label_x = onehot(label_x, self.charset.num_classes)

            y_x = [label_x, length_x]
            y.append(y_x)
        return image, y



class SpellingMutation(object):
    def __init__(self, pn0=0.7, pn1=0.85, pn2=0.95, pt0=0.7, pt1=0.85, charset=None):
        """ 
        Args:
            pn0: the prob of not modifying characters is (pn0)
            pn1: the prob of modifying one characters is (pn1 - pn0)
            pn2: the prob of modifying two characters is (pn2 - pn1), 
                 and three (1 - pn2)
            pt0: the prob of replacing operation is pt0.
            pt1: the prob of inserting operation is (pt1 - pt0),
                 and deleting operation is (1 - pt1)
        """
        super().__init__()
        self.pn0, self.pn1, self.pn2 = pn0, pn1, pn2
        self.pt0, self.pt1 = pt0, pt1
        self.charset = charset
        logging.info(f'the probs: pn0={self.pn0}, pn1={self.pn1} ' + 
                     f'pn2={self.pn2}, pt0={self.pt0}, pt1={self.pt1}')

    def is_digit(self, text, ratio=0.5):
        length = max(len(text), 1)
        digit_num = sum([t in self.charset.digits for t in text])
        if digit_num / length < ratio: return False
        return True

    def is_unk_char(self, char):
        # return char == self.charset.unk_char
        return (char not in self.charset.digits) and (char not in self.charset.alphabets)

    def get_num_to_modify(self, length):
        prob = random.random()
        if prob < self.pn0: num_to_modify = 0
        elif prob < self.pn1: num_to_modify = 1
        elif prob < self.pn2: num_to_modify = 2
        else: num_to_modify = 3
        
        if length <= 1: num_to_modify = 0
        elif length >= 2 and length <= 4: num_to_modify = min(num_to_modify, 1)
        else: num_to_modify = min(num_to_modify, length // 2)  # smaller than length // 2
        return num_to_modify

    def __call__(self, text, debug=False):
        if self.is_digit(text): return text
        length = len(text)
        num_to_modify = self.get_num_to_modify(length)
        if num_to_modify <= 0: return text

        chars = []
        index = np.arange(0, length)
        random.shuffle(index)
        index = index[: num_to_modify]
        if debug: self.index = index
        for i, t in enumerate(text):
            if i not in index: chars.append(t)
            elif self.is_unk_char(t): chars.append(t)
            else:
                prob = random.random()
                if prob < self.pt0: # replace
                    chars.append(random.choice(self.charset.alphabets))
                elif prob < self.pt1: # insert
                    chars.append(random.choice(self.charset.alphabets))
                    chars.append(t)
                else: # delete
                    continue
        new_text = ''.join(chars[: self.charset.max_length-1])
        return new_text if len(new_text) >= 1 else text