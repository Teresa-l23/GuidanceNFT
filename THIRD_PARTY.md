# Third-party code

Each experiment builds on an existing code base. `setup_upstream.sh` in the experiment
directory clones it at the pinned commit below and, where listed, applies our patch.

| Used by | Project | Pinned commit | License | What this repository ships |
|---|---|---|---|---|
| `sd35/` | [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) | `cbb14f84` | Apache-2.0 | `sd35/patches/diffusion_nft_upstream.patch`, a modified-files patch; the changes are described in `sd35/baselines/diffusion_nft/LOCAL_PATCHES.md` |
| `toy/` | [NVlabs/edm2](https://github.com/NVlabs/edm2) (`toy_example.py`, `dnnlib`, `torch_utils`) and the Autoguidance toy models | `4bf8162f` | CC BY-NC-SA 4.0 | `toy/patches/toy_example.patch`, which is an adaptation of `toy_example.py` and is distributed under CC BY-NC-SA 4.0 |
| `elf_owt/` | [lillian039/ELF](https://github.com/lillian039/ELF) | `5098bf28` | MIT | `elf_owt/patches/elf_src.patch` |
| `gsm8k/` | [chengxiang/CEDR](https://github.com/chengxiang/CEDR) | `538198dc` | MIT | `gsm8k/patches/cedr.patch` |
| `sit_imagenet/` | [tzco/Diffusion-wo-CFG](https://github.com/tzco/Diffusion-wo-CFG) (SiT model and sampler) | `53c386f4` | not stated | nothing; the files are fetched by the setup script |

Our own files (everything outside `patches/` and the fetched `third_party/` and
`upstream/` directories) are released under the MIT License.
