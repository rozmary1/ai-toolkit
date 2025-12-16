# Пример тренировки Z-Image с LyCORIS

Ниже приведён пример конфигурации для тренировки Z-Image с адаптерами LyCORIS (LoHa/LoKr/GLoRA и др.).
Конфиг сохраняет привычную схему нейминга `lora_transformer_*`, поэтому получившиеся веса будут совместимы
с ComfyUI и другими пайплайнами, ожидающими стандартные LoRA названия.

## Минимальный пример YAML
Сохраните файл, например, как `config/examples/train_lycoris_zimage.yaml` и запустите его через CLI/GUI.

```yaml
---
job: extension
config:
  name: "my_zimage_lyco_lora"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0

      # Важная часть: LyCORIS и transformer-only таргетинг для Z-Image
      network:
        type: "lycoris"          # включает спец. сеть LycorisSpecialNetwork
        linear: 16               # ранг LoRA/LoHa/LoKr
        linear_alpha: 16
        transformer_only: true   # Z-Image использует только transformer-блоки
        network_kwargs:
          algo: "loha"           # варианты: loha, lokr, glora, ia3, dylora и т.д.
          # при необходимости можно добавить другие параметры LyCORIS

      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4

      datasets:
        - folder_path: "/path/to/images"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [ 448, 672, 896 ]  # кратно 16*2 (VAE и patch size)

      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        dtype: bf16

      model:
        arch: "zimage"
        name_or_path: "Tongyi-MAI/Z-Image-Turbo"   # базовая модель или локальный путь
        quantize: true
        quantize_te: true
        qtype: "qfloat8"
        assistant_lora_path: "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors"
        low_vram: true

      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 896
        height: 896
        prompts:
          - "studio photo of a person in soft light"
          - "a futuristic cityscape at sunset, wide angle"
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 1
        sample_steps: 8

meta:
  name: "[name]"
  version: '1.0'
```

### Что можно менять
- `network.linear` / `linear_alpha`: ранг адаптера.
- `network.network_kwargs.algo`: выбор алгоритма LyCORIS (LoHa/LoKr/GLoRA/IA3/DyLoRA и др. доступные в LyCORIS).
- `assistant_lora_path`: можно убрать, если не нужен вспомогательный адаптер из Z-Image Turbo.
- `resolution`: числа должны делиться на 32 (требование VAE и патча Z-Image).

### Советы
- Оставляйте `transformer_only: true`, чтобы лоры применялись только к трансформерным блокам Z-Image.
- Для экономии VRAM используйте `quantize: true` и `qtype: "qfloat8"`.
- Обновляйте `prompts` в секции `sample`, чтобы во время тренировки видеть актуальные примеры.
