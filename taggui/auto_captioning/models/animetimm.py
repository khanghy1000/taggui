import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

import huggingface_hub
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image as PilImage

import auto_captioning.captioning_thread as captioning_thread
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image

KAOMOJIS = [
    '0_0', '(o)_(o)', '+_+', '+_-', '._.', '<o>_<o>', '<|>_<|>', '=_=',
    '>_<', '3_3', '6_9', '>_o', '@_@', '^_^', 'o_o', 'u_u', 'x_x',
    '|_|', '||_||'
]

BACKUP_REPO = 'Makki2104/animetimm'

INTERPOLATION_MODES = {
    'bicubic': PilImage.Resampling.BICUBIC,
    'bilinear': PilImage.Resampling.BILINEAR,
    'nearest': PilImage.Resampling.NEAREST,
    'box': PilImage.Resampling.BOX,
    'hamming': PilImage.Resampling.HAMMING,
    'lanczos': PilImage.Resampling.LANCZOS,
}


def get_tags_to_exclude(tags_to_exclude_string: str) -> list[str]:
    if not tags_to_exclude_string.strip():
        return []
    tags = re.split(r'(?<!\\),', tags_to_exclude_string)
    tags = [tag.strip().replace(r'\,', ',') for tag in tags]
    return tags


class PadToSize:
    def __init__(self, size: tuple[int, int] | list[int] | int = (512, 512),
                 interpolation: str = 'bilinear',
                 background_color: str | tuple = 'white'):
        if isinstance(size, int):
            size = (size, size)
        self.target_h, self.target_w = int(size[0]), int(size[1])
        if background_color == 'white':
            self.bg_color = (255, 255, 255)
        elif background_color == 'black':
            self.bg_color = (0, 0, 0)
        elif isinstance(background_color, (list, tuple)):
            self.bg_color = tuple(background_color)
        else:
            self.bg_color = (255, 255, 255)

    def __call__(self, img: PilImage.Image) -> PilImage.Image:
        if isinstance(img, np.ndarray):
            img = PilImage.fromarray(img)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGBA')
            canvas = PilImage.new('RGBA', img.size, (*self.bg_color, 255))
            canvas.paste(img, (0, 0), img)
            img = canvas.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        w, h = img.size
        target_aspect = self.target_w / self.target_h
        current_aspect = w / h

        if current_aspect > target_aspect:
            new_w = w
            new_h = int(round(w / target_aspect))
        else:
            new_h = h
            new_w = int(round(h * target_aspect))

        new_img = PilImage.new('RGB', (new_w, new_h), self.bg_color)
        new_img.paste(img, ((new_w - w) // 2, ((new_h - h) // 2)))
        return new_img


class Resize:
    def __init__(self, size: tuple[int, int] | list[int] | int = 512,
                 interpolation: str = 'bicubic', antialias: bool = True,
                 max_size: int | None = None):
        self.size = size
        self.max_size = max_size
        self.interpolation = INTERPOLATION_MODES.get(
            str(interpolation).lower(), PilImage.Resampling.BICUBIC)

    def __call__(self, img: PilImage.Image) -> PilImage.Image:
        if isinstance(img, np.ndarray):
            img = PilImage.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        w, h = img.size
        if isinstance(self.size, (list, tuple)) and len(self.size) >= 2:
            new_w, new_h = int(self.size[1]), int(self.size[0])
        elif isinstance(self.size, int) or (isinstance(self.size, (list, tuple)) and len(self.size) == 1):
            s = int(self.size[0] if isinstance(self.size, (list, tuple)) else self.size)
            if (w <= h and w == s) or (h <= w and h == s):
                new_w, new_h = w, h
            elif w < h:
                new_w = s
                new_h = int(round(h * s / w))
            else:
                new_h = s
                new_w = int(round(w * s / h))
        else:
            new_w, new_h = 512, 512

        if self.max_size is not None and max(new_w, new_h) > self.max_size:
            scale = self.max_size / max(new_w, new_h)
            new_w = int(round(new_w * scale))
            new_h = int(round(new_h * scale))

        return img.resize((new_w, new_h), resample=self.interpolation)


class CenterCrop:
    def __init__(self, size: tuple[int, int] | list[int] | int = 512):
        if isinstance(size, int):
            self.crop_h, self.crop_w = size, size
        elif isinstance(size, (list, tuple)) and len(size) >= 2:
            self.crop_h, self.crop_w = int(size[0]), int(size[1])
        elif isinstance(size, (list, tuple)) and len(size) == 1:
            self.crop_h, self.crop_w = int(size[0]), int(size[0])
        else:
            self.crop_h, self.crop_w = 512, 512

    def __call__(self, img: PilImage.Image) -> PilImage.Image:
        if isinstance(img, np.ndarray):
            img = PilImage.fromarray(img)
        w, h = img.size
        left = max(0, (w - self.crop_w) // 2)
        top = max(0, (h - self.crop_h) // 2)
        right = min(w, left + self.crop_w)
        bottom = min(h, top + self.crop_h)
        return img.crop((left, top, right, bottom))


class MaybeToTensor:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            t = img
            if t.ndim == 3 and t.shape[2] in (1, 3, 4):
                t = t.permute(2, 0, 1)
            if t.dtype == torch.uint8:
                t = t.to(dtype=torch.float32) / 255.0
            else:
                t = t.to(dtype=torch.float32)
            return t
        if isinstance(img, PilImage.Image):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose((2, 0, 1)))
            return tensor
        if isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                arr = arr.transpose((2, 0, 1))
            return torch.from_numpy(arr)
        return img


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor) -> torch.Tensor:
        if isinstance(tensor, PilImage.Image) or isinstance(tensor, np.ndarray):
            tensor = MaybeToTensor()(tensor)
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 3 and tensor.shape[2] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            if tensor.dtype == torch.uint8:
                tensor = tensor.to(dtype=torch.float32) / 255.0
            else:
                tensor = tensor.to(dtype=torch.float32)
            mean = self.mean.to(tensor.device, tensor.dtype)
            std = self.std.to(tensor.device, tensor.dtype)
            return (tensor - mean) / std
        return tensor


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def create_torchvision_transforms(preprocess_config: list | dict | None):
    if isinstance(preprocess_config, dict):
        preprocess_config = preprocess_config.get('test', preprocess_config)

    if not isinstance(preprocess_config, list):
        preprocess_config = [
            {'type': 'pad_to_size', 'size': [512, 512], 'interpolation': 'bilinear', 'background_color': 'white'},
            {'type': 'resize', 'size': 512, 'interpolation': 'bicubic', 'antialias': True},
            {'type': 'center_crop', 'size': [512, 512]},
            {'type': 'maybe_to_tensor'},
            {'type': 'normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        ]

    transforms = []
    has_to_tensor = False
    for item in preprocess_config:
        if not isinstance(item, dict):
            continue
        op_type = str(item.get('type') or item.get('name') or '').lower().replace('_', '').replace('-', '')
        if op_type == 'padtosize':
            transforms.append(PadToSize(
                size=item.get('size', (512, 512)),
                interpolation=item.get('interpolation', 'bilinear'),
                background_color=item.get('background_color', 'white')
            ))
        elif op_type == 'resize':
            transforms.append(Resize(
                size=item.get('size', 512),
                interpolation=item.get('interpolation', 'bicubic'),
                antialias=item.get('antialias', True),
                max_size=item.get('max_size', None)
            ))
        elif op_type == 'centercrop':
            transforms.append(CenterCrop(
                size=item.get('size', 512)
            ))
        elif op_type in ('maybetotensor', 'totensor'):
            transforms.append(MaybeToTensor())
            has_to_tensor = True
        elif op_type == 'normalize':
            if not has_to_tensor:
                transforms.append(MaybeToTensor())
                has_to_tensor = True
            transforms.append(Normalize(
                mean=item.get('mean', [0.485, 0.456, 0.406]),
                std=item.get('std', [0.229, 0.224, 0.225])
            ))

    if not has_to_tensor:
        transforms.append(MaybeToTensor())

    return Compose(transforms)


class AnimeTimmModel:
    def __init__(self, model_id: str, device: torch.device,
                 models_directory_path: Path | None = None):
        self.model_id = model_id
        self.device = device
        self.models_directory_path = models_directory_path

        tags_path = self._resolve_or_download_file('selected_tags.csv')
        preprocess_path = self._resolve_or_download_file('preprocess.json')

        self.tags_df = pd.read_csv(tags_path, keep_default_na=False)
        self.tag_names = self.tags_df['name'].astype(str).tolist()
        self.tag_categories = self.tags_df['category'].astype(int).tolist()
        self.best_thresholds = self.tags_df['best_threshold'].astype(float).to_numpy()

        with open(preprocess_path, 'r', encoding='utf-8') as f:
            preprocess_data = json.load(f)
        preprocess_config = (preprocess_data.get('test', preprocess_data)
                             if isinstance(preprocess_data, dict) else preprocess_data)
        self.preprocessor = create_torchvision_transforms(preprocess_config)

        self.model = self._load_timm_model()
        self.model.to(self.device)
        self.model.eval()

    def _resolve_or_download_file(self, filename: str) -> str:
        model_path = Path(self.model_id)
        if model_path.is_dir() and (model_path / filename).is_file():
            return str(model_path / filename)

        if self.models_directory_path:
            model_name = self.model_id.split('/')[-1]
            candidate = self.models_directory_path / model_name / filename
            if candidate.is_file():
                return str(candidate)
            candidate_full = self.models_directory_path / self.model_id / filename
            if candidate_full.is_file():
                return str(candidate_full)

        try:
            return huggingface_hub.hf_hub_download(
                repo_id=self.model_id,
                filename=filename,
                repo_type='model'
            )
        except Exception as e:
            if self.model_id.startswith('animetimm/'):
                model_name = self.model_id.split('/', 1)[-1]
                return huggingface_hub.hf_hub_download(
                    repo_id=BACKUP_REPO,
                    filename=f'{model_name}/{filename}',
                    repo_type='model'
                )
            raise e

    def _load_timm_model(self):
        model_path = Path(self.model_id)
        if model_path.is_dir():
            return timm.create_model(f'local-dir:{model_path}', pretrained=True)

        if self.models_directory_path:
            model_name = self.model_id.split('/')[-1]
            candidate = self.models_directory_path / model_name
            if candidate.is_dir() and (candidate / 'config.json').is_file():
                return timm.create_model(f'local-dir:{candidate}', pretrained=True)

        try:
            return timm.create_model(f'hf-hub:{self.model_id}', pretrained=True)
        except Exception as e:
            if self.model_id.startswith('animetimm/'):
                model_name = self.model_id.split('/', 1)[-1]
                cache_dir = Path(tempfile.gettempdir()) / 'taggui_animetimm' / model_name
                cache_dir.mkdir(parents=True, exist_ok=True)
                for fname in ['config.json', 'preprocess.json', 'selected_tags.csv', 'pytorch_model.bin']:
                    try:
                        huggingface_hub.hf_hub_download(
                            repo_id=BACKUP_REPO,
                            filename=f'{model_name}/{fname}',
                            local_dir=str(cache_dir.parent),
                            local_dir_use_symlinks=False
                        )
                    except Exception:
                        pass
                return timm.create_model(f'local-dir:{cache_dir}', pretrained=True)
            raise e

    def generate_tags(self, prediction: np.ndarray,
                      animetimm_settings: dict) -> tuple[tuple, tuple]:
        min_probability = animetimm_settings.get('min_probability', 0.35)
        use_custom_threshold = animetimm_settings.get('use_custom_threshold', False)
        max_tags = animetimm_settings.get('max_tags', 30)
        include_general = animetimm_settings.get('include_general', True)
        include_character = animetimm_settings.get('include_character', True)
        include_artist = animetimm_settings.get('include_artist', False)
        include_rating = animetimm_settings.get('include_rating', False)
        replace_underscore = animetimm_settings.get('replace_underscore', True)
        tags_to_exclude = get_tags_to_exclude(
            animetimm_settings.get('tags_to_exclude', ''))
        tags_to_exclude_set = set(tags_to_exclude)

        allowed_categories = set()
        if include_general:
            allowed_categories.add(0)
        if include_artist:
            allowed_categories.add(1)
        if include_character:
            allowed_categories.add(4)
        if include_rating:
            allowed_categories.add(9)

        tags_and_probabilities = []
        for tag_name, category, best_threshold, score in zip(
            self.tag_names, self.tag_categories, self.best_thresholds, prediction
        ):
            if category not in allowed_categories:
                continue

            effective_threshold = (min_probability if use_custom_threshold
                                   else max(min_probability, best_threshold))
            if score < effective_threshold:
                continue

            formatted_tag = tag_name
            if replace_underscore and tag_name not in KAOMOJIS:
                formatted_tag = tag_name.replace('_', ' ')

            if (formatted_tag in tags_to_exclude_set
                    or tag_name in tags_to_exclude_set):
                continue

            tags_and_probabilities.append((formatted_tag, float(score)))

        tags_and_probabilities.sort(key=lambda x: x[1], reverse=True)
        tags_and_probabilities = tags_and_probabilities[:max_tags]

        if tags_and_probabilities:
            tags, probabilities = zip(*tags_and_probabilities)
        else:
            tags, probabilities = (), ()
        return tags, probabilities


class AnimeTimm(AutoCaptioningModel):
    image_mode = 'RGB'
    dtype = torch.float32

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        super().__init__(captioning_thread_, caption_settings)
        self.animetimm_settings = self.caption_settings.get(
            'animetimm_settings', {
                'show_probabilities': True,
                'min_probability': 0.35,
                'use_custom_threshold': False,
                'max_tags': 30,
                'include_general': True,
                'include_character': True,
                'include_artist': False,
                'include_rating': False,
                'replace_underscore': True,
                'tags_to_exclude': ''
            }
        )
        self.show_probabilities = self.animetimm_settings.get(
            'show_probabilities', True)

    def get_error_message(self) -> str | None:
        return None

    def get_processor(self):
        return None

    def get_model(self):
        models_directory_path = self.thread.models_directory_path
        return AnimeTimmModel(
            model_id=self.model_id,
            device=self.device,
            models_directory_path=models_directory_path
        )

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        if are_multiple_images_selected:
            captioning_start_datetime_string = (
                self.get_captioning_start_datetime_string(
                    captioning_start_datetime))
            return (f'Generating tags... (device: {self.device}, start time: '
                    f'{captioning_start_datetime_string})')
        return f'Generating tags... (device: {self.device})'

    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> torch.Tensor:
        pil_image = self.load_image(image)
        input_tensor = self.model.preprocessor(pil_image)
        if isinstance(input_tensor, PilImage.Image):
            input_tensor = MaybeToTensor()(input_tensor)
        elif not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(np.array(input_tensor))

        if input_tensor.ndim == 3 and input_tensor.shape[2] in (1, 3, 4):
            input_tensor = input_tensor.permute(2, 0, 1)

        if input_tensor.dtype == torch.uint8:
            input_tensor = input_tensor.to(dtype=torch.float32) / 255.0
        else:
            input_tensor = input_tensor.to(dtype=torch.float32)

        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.ndim == 4 and input_tensor.shape[3] in (1, 3, 4):
            input_tensor = input_tensor.permute(0, 3, 1, 2)

        input_tensor = input_tensor.to(device=self.device, dtype=torch.float32)
        return input_tensor

    def generate_caption(self, model_inputs: torch.Tensor,
                         image_prompt: str) -> tuple[str, str]:
        if not isinstance(model_inputs, torch.Tensor):
            model_inputs = torch.tensor(np.array(model_inputs))
        if model_inputs.ndim == 3 and model_inputs.shape[2] in (1, 3, 4):
            model_inputs = model_inputs.permute(2, 0, 1)
        if model_inputs.dtype == torch.uint8:
            model_inputs = model_inputs.to(dtype=torch.float32) / 255.0
        else:
            model_inputs = model_inputs.to(dtype=torch.float32)
        if model_inputs.ndim == 3:
            model_inputs = model_inputs.unsqueeze(0)
        elif model_inputs.ndim == 4 and model_inputs.shape[3] in (1, 3, 4):
            model_inputs = model_inputs.permute(0, 3, 1, 2)
        model_inputs = model_inputs.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model.model(model_inputs)
            prediction = torch.sigmoid(output)[0].cpu().numpy()

        tags, probabilities = self.model.generate_tags(
            prediction, self.animetimm_settings)
        caption = self.thread.tag_separator.join(tags)
        if self.show_probabilities:
            console_output_caption = self.thread.tag_separator.join(
                f'{tag} ({probability:.2f})'
                for tag, probability in zip(tags, probabilities)
            )
        else:
            console_output_caption = caption
        return caption, console_output_caption
