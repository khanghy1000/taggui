# Based on https://huggingface.co/pixai-labs/pixai-tagger-v1.0.
# The model uses custom code (`tagger_pipeline.py` in the model repository),
# so it must be loaded with `trust_remote_code=True`.
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

import auto_captioning.captioning_thread as captioning_thread
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image

KAOMOJIS = ['0_0', '(o)_(o)', '+_+', '+_-', '._.', '<o>_<o>', '<|>_<|>',
            '=_=', '>_<', '3_3', '6_9', '>_o', '@_@', '^_^', 'o_o', 'u_u',
            'x_x', '|_|', '||_||']

# Recommended per-category thresholds from the model card. They are the
# per-category macro-F1 operating points from the full-vocabulary evaluation
# and are also the defaults of the model's own pipeline.
RECOMMENDED_THRESHOLDS = {
    'general': 0.17,
    'character': 0.27,
    'style': 0.15,
    'copyright': 0.24,
    'meta': 0.17,
    'rating': 0.41
}


def get_tags_to_exclude(tags_to_exclude_string: str) -> list[str]:
    if not tags_to_exclude_string.strip():
        return []
    tags = re.split(r'(?<!\\),', tags_to_exclude_string)
    tags = [tag.strip().replace(r'\,', ',') for tag in tags]
    return tags


class PixaiTaggerModel:
    def __init__(self, model_id: str, device: torch.device,
                 models_directory_path: Path | None = None):
        self.model_id = model_id
        self.device = device
        resolved_model_id = self._resolve_model_path(model_id,
                                                     models_directory_path)
        self.processor = AutoImageProcessor.from_pretrained(
            resolved_model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            resolved_model_id, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        config = self.model.config
        self.tags: list[str] = list(config.tags)
        self.tags_split: list[tuple[str, int]] = [
            (category, int(count)) for category, count
            in config.tags_split
        ]
        config_thresholds = (dict(config.category_best_threshold)
                             if config.category_best_threshold else {})
        self.category_thresholds: dict[str, float] = {}
        for category, _ in self.tags_split:
            if category in config_thresholds:
                self.category_thresholds[category] = float(
                    config_thresholds[category])
            else:
                self.category_thresholds[category] = float(
                    RECOMMENDED_THRESHOLDS.get(category, 0.2))

    @staticmethod
    def _resolve_model_path(model_id: str,
                            models_directory_path: Path | None) -> str:
        model_path = Path(model_id)
        if model_path.is_dir():
            return str(model_path)
        if models_directory_path:
            model_name = model_id.split('/')[-1]
            candidate = models_directory_path / model_name
            if candidate.is_dir():
                return str(candidate)
            candidate_full = models_directory_path / model_id
            if candidate_full.is_dir():
                return str(candidate_full)
        return model_id

    def generate_tags(self, prediction: np.ndarray,
                      pixai_tagger_settings: dict) -> tuple[tuple, tuple]:
        max_tags = int(pixai_tagger_settings.get('max_tags', 50))
        replace_underscore = pixai_tagger_settings.get('replace_underscore',
                                                       True)
        tags_to_exclude_set = set(get_tags_to_exclude(
            pixai_tagger_settings.get('tags_to_exclude', '')))
        thresholds = {}
        for category, _ in self.tags_split:
            default_threshold = self.category_thresholds.get(
                category, RECOMMENDED_THRESHOLDS.get(category, 0.2))
            try:
                thresholds[category] = float(pixai_tagger_settings.get(
                    f'{category}_threshold', default_threshold))
            except (TypeError, ValueError):
                thresholds[category] = float(default_threshold)
        included_categories = set()
        for category, _ in self.tags_split:
            # Rating tags are excluded by default, like in the AnimeTimm
            # implementation.
            if pixai_tagger_settings.get(f'include_{category}',
                                         category != 'rating'):
                included_categories.add(category)

        tags_and_probabilities = []
        tag_index = 0
        for category, count in self.tags_split:
            if category not in included_categories:
                tag_index += count
                continue
            threshold = thresholds[category]
            for _ in range(count):
                tag_name = self.tags[tag_index]
                score = float(prediction[tag_index])
                tag_index += 1
                if score < threshold:
                    continue
                formatted_tag = tag_name
                if replace_underscore and tag_name not in KAOMOJIS:
                    formatted_tag = tag_name.replace('_', ' ')
                if (formatted_tag in tags_to_exclude_set
                        or tag_name in tags_to_exclude_set):
                    continue
                tags_and_probabilities.append((formatted_tag, score))

        tags_and_probabilities.sort(key=lambda x: x[1], reverse=True)
        tags_and_probabilities = tags_and_probabilities[:max_tags]

        if tags_and_probabilities:
            tags, probabilities = zip(*tags_and_probabilities)
        else:
            tags, probabilities = (), ()
        return tags, probabilities


class PixaiTagger(AutoCaptioningModel):
    image_mode = 'RGBA'
    dtype = torch.float32

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        super().__init__(captioning_thread_, caption_settings)
        self.pixai_tagger_settings = self.caption_settings.get(
            'pixai_tagger_settings', {
                'show_probabilities': True,
                'general_threshold': RECOMMENDED_THRESHOLDS['general'],
                'character_threshold': RECOMMENDED_THRESHOLDS['character'],
                'style_threshold': RECOMMENDED_THRESHOLDS['style'],
                'copyright_threshold': RECOMMENDED_THRESHOLDS['copyright'],
                'meta_threshold': RECOMMENDED_THRESHOLDS['meta'],
                'rating_threshold': RECOMMENDED_THRESHOLDS['rating'],
                'max_tags': 50,
                'include_general': True,
                'include_character': True,
                'include_copyright': True,
                'include_style': True,
                'include_meta': False,
                'include_rating': False,
                'replace_underscore': True,
                'tags_to_exclude': ''
            }
        )
        self.show_probabilities = self.pixai_tagger_settings.get(
            'show_probabilities', True)

    def get_error_message(self) -> str | None:
        return None

    def get_processor(self):
        return None

    def get_model(self):
        models_directory_path = self.thread.models_directory_path
        return PixaiTaggerModel(
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
        pixel_values = self.model.processor(
            pil_image, return_tensors='pt')['pixel_values']
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(np.array(pixel_values))
        pixel_values = pixel_values.to(device=self.device,
                                       dtype=torch.float32)
        return pixel_values

    def generate_caption(self, model_inputs: torch.Tensor,
                         image_prompt: str) -> tuple[str, str]:
        if not isinstance(model_inputs, torch.Tensor):
            model_inputs = torch.tensor(np.array(model_inputs))
        model_inputs = model_inputs.to(device=self.device,
                                       dtype=torch.float32)

        with torch.no_grad():
            output = self.model.model(model_inputs)
            if not isinstance(output, torch.Tensor):
                output = output.logits
            prediction = torch.sigmoid(output)[0].float().cpu().numpy()

        tags, probabilities = self.model.generate_tags(
            prediction, self.pixai_tagger_settings)
        caption = self.thread.tag_separator.join(tags)
        if self.show_probabilities:
            console_output_caption = self.thread.tag_separator.join(
                f'{tag} ({probability:.2f})'
                for tag, probability in zip(tags, probabilities)
            )
        else:
            console_output_caption = caption
        return caption, console_output_caption
