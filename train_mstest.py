# flake8: noqa: E402
import platform
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Removed: import torch.distributed as dist
# Removed: from torch.nn.parallel import DistributedDataParallel as DDP
# Removed: from torch.cuda.amp import autocast, GradScaler # Accelerate handles this
from tqdm import tqdm
import logging
from config import config
import argparse
import datetime

# +++ Accelerate Imports +++
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    # Removed: DistributedBucketSampler # Accelerate handles distributed sampling
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
    WavLMDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    WavLMLoss,
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

# --- Removed torch backend settings that might conflict or are handled differently ---
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.set_num_threads(1) # This might be better controlled externally or by Accelerate defaults
# torch.set_float32_matmul_precision("medium") # Might be handled by Accelerate mixed precision
# torch.backends.cuda.sdp_kernel("flash") # Keep if desired and compatible
# torch.backends.cuda.enable_flash_sdp(True) # Keep if desired and compatible
# torch.backends.cuda.enable_mem_efficient_sdp(True) # Keep if desired and compatible

global_step = 0


def run():
    # --- Removed Manual Environment Variable Handling ---
    # Accelerate uses 'accelerate launch' which sets these up.

    # --- Initialize Accelerator ---
    # Pass DDP kwargs if needed, like bucket_cap_mb
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, bucket_cap_mb=512) # Set find_unused_parameters based on model complexity if needed
    # Configure mixed precision based on hps
    hps_for_accelerator = utils.get_hparams_from_file(config.train_ms_config.config_path) # Load hps early for accelerator config
    mixed_precision = "bf16" if getattr(hps_for_accelerator.train, "bf16_run", False) else "no"
    # Consider adding gradient_accumulation_steps if you want Accelerate to handle it
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs]
        # cpu=False # Set to True if debugging on CPU
        # log_with="tensorboard", # Enable if using accelerate's tracker integration
        # project_dir=os.path.join(args.model, config.train_ms_config.model) # For tracker integration
    )
    device = accelerator.device # Get the device assigned by Accelerate

    # --- Command line/config parsing (mostly unchanged) ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=config.train_ms_config.config_path,
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="数据集文件夹路径，请注意，数据不再默认放在/logs文件夹下。如果需要用命令行配置，请声明相对于根目录的路径",
        default=config.dataset_path,
    )
    args = parser.parse_args()
    model_dir = os.path.join(args.model, config.train_ms_config.model)

    # Create model directory on main process only
    if accelerator.is_main_process:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    accelerator.wait_for_everyone() # Ensure directory exists before others might need it

    hps = utils.get_hparams_from_file(args.config)
    hps.model_dir = model_dir
    # Config file copy logic (run only on main process)
    if accelerator.is_main_process:
        if os.path.realpath(args.config) != os.path.realpath(config.train_ms_config.config_path):
            with open(args.config, "r", encoding="utf-8") as f:
                data = f.read()
            with open(config.train_ms_config.config_path, "w", encoding="utf-8") as f:
                f.write(data)

    # --- Seed setting (Use Accelerate's utility) ---
    set_seed(hps.train.seed)
    # Removed: torch.manual_seed(hps.train.seed)
    # Removed: torch.cuda.set_device(local_rank) # Accelerate handles device assignment

    global global_step
    logger = None
    writer = None
    writer_eval = None
    # --- Setup logging and TensorBoard only on the main process ---
    if accelerator.is_main_process:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # --- Dataset and DataLoader ---
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    # Removed: train_sampler = DistributedBucketSampler(...) # Accelerate handles this
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=min(config.train_ms_config.num_workers, os.cpu_count() - 1),
        shuffle=True, # Set shuffle=True, Accelerate handles distributed sampling
        pin_memory=True,
        collate_fn=collate_fn,
        batch_size=hps.train.batch_size, # Set batch size directly
        # Removed: batch_sampler=train_sampler
        persistent_workers=True,
        prefetch_factor=6,
    )

    eval_loader = None
    if accelerator.is_main_process: # Create eval loader only on main process if evaluation is rank-specific
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1, # Keep batch size 1 for typical evaluation
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    # --- Model Initialization (without .cuda()) ---
    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas is True
    ):
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator is True
    ):
        print("Using duration discriminator for VITS2")
        # Initialize on CPU first, prepare will move it
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        )
    else:
        net_dur_disc = None

    if (
        "use_wavlm_discriminator" in hps.model.keys()
        and hps.model.use_wavlm_discriminator is True
    ):
        # Initialize on CPU first, prepare will move it
        net_wd = WavLMDiscriminator(
            hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
        )
        wl = WavLMLoss( # WavLMLoss needs the WavLM model path
            hps.model.slm.model, # Make sure this path is correct
            net_wd, # Pass the un-prepared model here, it's used internally
            hps.data.sampling_rate,
            hps.model.slm.sr,
        ) # wl itself might need .to(device) later if it has parameters/buffers
    else:
        net_wd = None
        wl = None


    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        print("Using normal encoder for VITS1")

    # Initialize on CPU first, prepare will move them
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    )
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    # --- Freezing Parameters (do this *before* optimizer creation) ---
    if getattr(hps.train, "freeze_ZH_bert", False):
        print("Freezing ZH bert encoder !!!")
        if hasattr(net_g, 'enc_p') and hasattr(net_g.enc_p, 'bert_proj'):
             for param in net_g.enc_p.bert_proj.parameters():
                param.requires_grad = False
        else:
             print("WARNING: Could not find net_g.enc_p.bert_proj to freeze.")

    if getattr(hps.train, "freeze_emo", False):
        print("Freezing emo vq !!!")
        if hasattr(net_g, 'enc_p') and hasattr(net_g.enc_p, 'emo_vq'):
            for param in net_g.enc_p.emo_vq.parameters():
                param.requires_grad = False
        else:
            print("WARNING: Could not find net_g.enc_p.emo_vq to freeze.")


    # --- Optimizer Initialization ---
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_dur_disc = None
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    optim_wd = None
    if net_wd is not None:
        optim_wd = torch.optim.AdamW(
            net_wd.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

    # --- Load Checkpoints (BEFORE preparing with Accelerate) ---
    # Make sure download happens on main process and others wait
    if accelerator.is_main_process:
        if config.train_ms_config.base["use_base_model"]:
            utils.download_checkpoint(
                hps.model_dir,
                config.train_ms_config.base,
                token=config.openi_token,
                mirror=config.mirror,
            )
    accelerator.wait_for_everyone() # Ensure download completes

    epoch_str = 1
    global_step = 0
    g_resume_lr, d_resume_lr = hps.train.learning_rate, hps.train.learning_rate
    dur_resume_lr, wd_resume_lr = hps.train.learning_rate, hps.train.learning_rate

    # Load logic (needs to handle potential errors gracefully)
    try:
        # Important: Load checkpoints for models and optimizers *before* accelerator.prepare
        # Assumes utils.load_checkpoint loads state_dict and returns epoch/lr info
        # Generator
        g_checkpoint_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        if g_checkpoint_path:
            _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
                g_checkpoint_path, net_g, optim_g,
                skip_optimizer=hps.train.skip_optimizer if "skip_optimizer" in hps.train else False, # Load optimizer state if not skipping
                # Map location to CPU temporarily if loading a GPU checkpoint before prepare
                map_location='cpu'
            )
            global_step = int(utils.get_steps(g_checkpoint_path))
            if not optim_g.param_groups[0].get('initial_lr'):
                 optim_g.param_groups[0]['initial_lr'] = g_resume_lr
            print(f"Loaded Generator from {g_checkpoint_path}, epoch: {epoch_str}, step: {global_step}")
        else:
             print("No Generator checkpoint found, starting from scratch.")


        # Discriminator
        d_checkpoint_path = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        if d_checkpoint_path:
            _, optim_d, d_resume_lr, _ = utils.load_checkpoint(
                d_checkpoint_path, net_d, optim_d,
                skip_optimizer=hps.train.skip_optimizer if "skip_optimizer" in hps.train else False,
                map_location='cpu'
            )
            if not optim_d.param_groups[0].get('initial_lr'):
                 optim_d.param_groups[0]['initial_lr'] = d_resume_lr
            print(f"Loaded Discriminator from {d_checkpoint_path}")
        else:
             print("No Discriminator checkpoint found, starting from scratch.")


        # Duration Discriminator
        if net_dur_disc is not None:
            dur_checkpoint_path = utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth")
            if dur_checkpoint_path:
                _, optim_dur_disc, dur_resume_lr, _ = utils.load_checkpoint(
                    dur_checkpoint_path, net_dur_disc, optim_dur_disc,
                    skip_optimizer=hps.train.skip_optimizer if "skip_optimizer" in hps.train else False,
                     map_location='cpu'
                )
                if not optim_dur_disc.param_groups[0].get('initial_lr'):
                     optim_dur_disc.param_groups[0]['initial_lr'] = dur_resume_lr
                print(f"Loaded Duration Discriminator from {dur_checkpoint_path}")
            else:
                print("No Duration Discriminator checkpoint found, initializing.")
                if optim_dur_disc and not optim_dur_disc.param_groups[0].get('initial_lr'):
                    optim_dur_disc.param_groups[0]['initial_lr'] = dur_resume_lr # Set initial LR if starting fresh

        # WavLM Discriminator
        if net_wd is not None:
            wd_checkpoint_path = utils.latest_checkpoint_path(hps.model_dir, "WD_*.pth")
            if wd_checkpoint_path:
                # Need to handle WavLMDiscriminator loading specifically if utils doesn't cover it
                 _, optim_wd, wd_resume_lr, _ = utils.load_checkpoint(
                    wd_checkpoint_path, net_wd, optim_wd,
                    skip_optimizer=hps.train.skip_optimizer if "skip_optimizer" in hps.train else False,
                    map_location='cpu'
                )
                 if not optim_wd.param_groups[0].get('initial_lr'):
                      optim_wd.param_groups[0]['initial_lr'] = wd_resume_lr
                 print(f"Loaded WavLM Discriminator from {wd_checkpoint_path}")
            else:
                print("No WavLM Discriminator checkpoint found, initializing.")
                if optim_wd and not optim_wd.param_groups[0].get('initial_lr'):
                    optim_wd.param_groups[0]['initial_lr'] = wd_resume_lr # Set initial LR if starting fresh

        # Ensure epoch starts correctly
        epoch_str = max(epoch_str, 1)
        if global_step > 0 and accelerator.is_main_process:
            print(f"Resuming training from epoch {epoch_str} and global step {global_step}")

    except Exception as e:
        # Fallback if loading fails catastrophically
        accelerator.print(f"Error loading checkpoint: {e}. Starting from epoch 1, global step 0.")
        epoch_str = 1
        global_step = 0
        # Set initial LRs if optimizers exist
        if optim_g and not optim_g.param_groups[0].get('initial_lr'):
            optim_g.param_groups[0]['initial_lr'] = hps.train.learning_rate
        if optim_d and not optim_d.param_groups[0].get('initial_lr'):
            optim_d.param_groups[0]['initial_lr'] = hps.train.learning_rate
        if optim_dur_disc and not optim_dur_disc.param_groups[0].get('initial_lr'):
            optim_dur_disc.param_groups[0]['initial_lr'] = hps.train.learning_rate
        if optim_wd and not optim_wd.param_groups[0].get('initial_lr'):
            optim_wd.param_groups[0]['initial_lr'] = hps.train.learning_rate


    # --- Learning Rate Schedulers ---
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2 # last_epoch is the *previous* epoch index
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_dur_disc = None
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    scheduler_wd = None
    if net_wd is not None:
        scheduler_wd = torch.optim.lr_scheduler.ExponentialLR(
            optim_wd, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
        # Move WavLMLoss module to device if it has state
        wl = wl.to(device) # Assuming wl might have buffers/parameters


    # --- Prepare components with Accelerate ---
    # Order matters if components depend on each other, but usually models/optimizers/dataloaders are independent
    # Important: Prepare will move models and optimizers to the correct device(s) and wrap them (e.g., with DDP)
    net_g, net_d, optim_g, optim_d, train_loader = accelerator.prepare(
        net_g, net_d, optim_g, optim_d, train_loader
    )
    # Prepare optional components
    if net_dur_disc is not None:
        net_dur_disc, optim_dur_disc = accelerator.prepare(net_dur_disc, optim_dur_disc)
    if net_wd is not None:
        net_wd, optim_wd = accelerator.prepare(net_wd, optim_wd)

    # Prepare eval_loader only on the main process if it exists
    if accelerator.is_main_process and eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)

    # Schedulers typically don't need `prepare` unless they have state that needs syncing across processes.
    # Standard PyTorch LR schedulers are usually updated based on epoch count, which is synced.

    # Removed: scaler = GradScaler(enabled=hps.train.bf16_run) # Accelerate handles scaling

    # --- Training Loop ---
    for epoch in range(epoch_str, hps.train.epochs + 1):
        # Removed: train_loader.batch_sampler.set_epoch(epoch) # Accelerate handles sampler epoch
        train_and_evaluate( # Pass accelerator instance
            accelerator,
            epoch,
            hps,
            [net_g, net_d, net_dur_disc, net_wd, wl],
            [optim_g, optim_d, optim_dur_disc, optim_wd],
            [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
            # Removed scaler
            [train_loader, eval_loader], # Pass potentially None eval_loader
            logger, # Pass potentially None logger
            [writer, writer_eval], # Pass potentially None writers
            device # Pass device explicitly
        )

        # Step schedulers after epoch (ensure optimizers were prepared)
        # No need for rank check, scheduler steps are usually deterministic based on epoch
        scheduler_g.step()
        scheduler_d.step()
        if scheduler_dur_disc is not None:
            scheduler_dur_disc.step()
        if scheduler_wd is not None:
            scheduler_wd.step()


def train_and_evaluate(
    accelerator, # Pass accelerator instance
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    # Removed scaler
    loaders,
    logger,
    writers,
    device # Receive device
):
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    # Schedulers might not be needed inside if only stepped outside
    # scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers
    train_loader, eval_loader = loaders
    writer, writer_eval = None, None # Initialize to None
    if accelerator.is_main_process and writers is not None:
        writer, writer_eval = writers

    global global_step

    # Set models to train mode
    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    if net_wd is not None:
        net_wd.train()

    # Use tqdm only on the main process for cleaner logs
    progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_main_process)

    for batch_idx, batch in enumerate(train_loader):
        # Unpack batch
        (
            x, x_lengths, spec, spec_lengths, y, y_lengths,
            speakers, tone, language, bert, emo
        ) = batch

        # Update MAS noise scale (accessing .module might be needed if prepare wrapped it)
        # Use accelerator.unwrap_model to be safe
        unwrapped_net_g = accelerator.unwrap_model(net_g)
        if unwrapped_net_g.use_noise_scaled_mas:
            current_mas_noise_scale = (
                unwrapped_net_g.mas_noise_scale_initial
                - unwrapped_net_g.noise_scale_delta * global_step
            )
            unwrapped_net_g.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        # Move data to the device assigned by Accelerate
        x, x_lengths = x.to(device), x_lengths.to(device)
        spec, spec_lengths = spec.to(device), spec_lengths.to(device)
        y, y_lengths = y.to(device), y_lengths.to(device)
        speakers = speakers.to(device)
        tone = tone.to(device)
        language = language.to(device)
        bert = bert.to(device)
        emo = emo.to(device)

        # --- Discriminator Training ---
        # Removed: with autocast(...) # Accelerator handles mixed precision context
        # Generator forward pass for discriminator targets (detach needed)
        with torch.no_grad(): # Ensure generator grads aren't calculated here
             (
                 y_hat, _, _, ids_slice, _, z_mask, _, _, g
             ) = net_g(x, x_lengths, spec, spec_lengths, speakers, tone, language, bert, emo)

        y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            ) # slice GT audio

        # MPD (Multi-Period Discriminator)
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach()) # Pass GT and generated (detached)
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

        # Duration Discriminator
        if net_dur_disc is not None:
            # Need hidden_x, x_mask, logw_, logw from generator forward pass *with grads*
            # Re-run relevant part of generator or get from full pass if structure allows
            # For simplicity, let's assume we need another G forward pass here inside the DurDisc block
             # This is inefficient; ideally restructure G to return needed intermediate vars
             with torch.no_grad(): # Re-get intermediates without tracking grads for G
                 _, _, _, _, x_mask, _, (hidden_x, logw, logw_), _, _ = net_g(
                    x, x_lengths, spec, spec_lengths, speakers, tone, language, bert, emo
                 )

             y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                 hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach(), g.detach()
             )
             loss_dur_disc, _, _ = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
             loss_dur_disc_all = loss_dur_disc # Keep separated for logging if needed
             optim_dur_disc.zero_grad()
             accelerator.backward(loss_dur_disc_all)
             # Removed scaler.unscale_
             if accelerator.sync_gradients: # Only clip when gradients are synced
                  grad_norm_dur = commons.clip_grad_value_(net_dur_disc.parameters(), None) # Or use accelerator.clip_grad_value_
             optim_dur_disc.step()


        # WavLM Discriminator
        if net_wd is not None and wl is not None:
            loss_slm = wl.discriminator(y.detach().squeeze(), y_hat.detach().squeeze()).mean()
            optim_wd.zero_grad()
            accelerator.backward(loss_slm)
            # Removed scaler.unscale_
            if accelerator.sync_gradients:
                grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None) # Or use accelerator.clip_grad_value_
            optim_wd.step()


        # Update MPD
        optim_d.zero_grad()
        accelerator.backward(loss_disc_all) # Use accelerator's backward
        # Removed: scaler.unscale_(optim_d)
        if accelerator.sync_gradients:
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None) # Or use accelerator.clip_grad_value_
            # Optional: Use accelerator's built-in clipping
            # if hps.train.grad_clip_val is not None:
            #    accelerator.clip_grad_value_(net_d.parameters(), hps.train.grad_clip_val)
        optim_d.step() # Optimizer step remains


        # --- Generator Training ---
        # Removed: with autocast(...)
        (
            y_hat, l_length, attn, ids_slice, x_mask, z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (hidden_x, logw, logw_), g, loss_commit
        ) = net_g(x, x_lengths, spec, spec_lengths, speakers, tone, language, bert, emo)


        mel = spec_to_mel_torch(
            spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate,
            hps.data.mel_fmin, hps.data.mel_fmax,
        )
        y_mel = commons.slice_segments(
            mel, ids_slice, hps.train.segment_size // hps.data.hop_length
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(), hps.data.filter_length, hps.data.n_mel_channels,
            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            hps.data.mel_fmin, hps.data.mel_fmax,
        )
        y = commons.slice_segments( # Slice GT audio again for generator loss
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )

        # Generator losses
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat) # Pass GT and generated (attached)

        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)

        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_commit * hps.train.c_commit

        # Add Dur Disc adversarial loss for generator
        if net_dur_disc is not None:
             _, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw, g) # Re-use intermediates
             loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
             loss_gen_all += loss_dur_gen

        # Add WavLM related losses for generator
        if net_wd is not None and wl is not None:
             loss_lm = wl(y.detach().squeeze(), y_hat.squeeze()).mean() # Perceptual loss
             loss_lm_gen = wl.generator(y_hat.squeeze()) # Adversarial loss against WavLM Disc
             loss_gen_all += loss_lm + loss_lm_gen


        optim_g.zero_grad()
        accelerator.backward(loss_gen_all) # Use accelerator's backward
        # Removed: scaler.unscale_(optim_g)
        if accelerator.sync_gradients:
             grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None) # Or use accelerator.clip_grad_norm_
             # Optional: Use accelerator's built-in clipping
             # if hps.train.grad_clip_norm is not None:
             #    accelerator.clip_grad_norm_(net_g.parameters(), hps.train.grad_clip_norm)
        optim_g.step() # Optimizer step remains

        # Removed: scaler.update() # Accelerate handles scaler updates internally

        # Update progress bar on main process
        if accelerator.is_main_process:
             progress_bar.update(1)
             progress_bar.set_description(f"Epoch {epoch} | Step {global_step}")


        # --- Logging and Checkpointing (Main Process Only) ---
        if accelerator.is_main_process:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"] # Get LR from prepared optimizer
                # Ensure losses are tensors on CPU for logging if gathered
                losses = [loss_disc.item(), loss_gen.item(), loss_fm.item(), loss_mel.item(), loss_dur.item(), loss_kl.item()]
                logger.info(f"Train Epoch: {epoch} [{100.0 * batch_idx / len(train_loader):.0f}%]")
                logger.info(f"Losses: {losses}, Step: {global_step}, LR: {lr}")

                # --- Scalar Logging ---
                scalar_dict = {
                    "loss/g/total": loss_gen_all.item(),
                    "loss/d/total": loss_disc_all.item(), # MPD loss
                    "learning_rate": lr,
                    "grad_norm/g": grad_norm_g if 'grad_norm_g' in locals() else 0,
                    "grad_norm/d": grad_norm_d if 'grad_norm_d' in locals() else 0,
                    "loss/g/gan": loss_gen.item(),
                    "loss/g/fm": loss_fm.item(),
                    "loss/g/mel": loss_mel.item(),
                    "loss/g/dur": loss_dur.item(),
                    "loss/g/kl": loss_kl.item(),
                    "loss/g/commit": loss_commit.item() * hps.train.c_commit,
                    # Add individual gan losses if needed
                }
                if net_dur_disc is not None:
                     scalar_dict.update({
                         "loss/dur_disc/total": loss_dur_disc_all.item(),
                         "loss/g/dur_gen": loss_dur_gen.item(),
                         "grad_norm/dur": grad_norm_dur if 'grad_norm_dur' in locals() else 0,
                     })
                if net_wd is not None:
                    scalar_dict.update({
                        "loss/wd/total": loss_slm.item(),
                        "loss/g/lm": loss_lm.item(),
                        "loss/g/lm_gen": loss_lm_gen.item(),
                        "grad_norm/wd": grad_norm_wd if 'grad_norm_wd' in locals() else 0,
                    })

                # --- Image Logging ---
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "all/attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()),
                }
                utils.summarize( # Use the main process writer
                    writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict
                )

            # --- Evaluation and Checkpointing ---
            if global_step % hps.train.eval_interval == 0:
                if eval_loader is not None and writer_eval is not None:
                     evaluate(hps, net_g, eval_loader, writer_eval, device) # Pass device
                     # Ensure model is back in train mode after evaluation
                     net_g.train()
                     if net_dur_disc: net_dur_disc.train()
                     if net_wd: net_wd.train()
                     if net_d: net_d.train()


                # Save checkpoints using unwrapped models
                accelerator.wait_for_everyone() # Ensure all processes finished step before saving
                unwrapped_net_g = accelerator.unwrap_model(net_g)
                unwrapped_net_d = accelerator.unwrap_model(net_d)

                utils.save_checkpoint(
                    unwrapped_net_g, optim_g, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"G_{global_step}.pth")
                )
                utils.save_checkpoint(
                    unwrapped_net_d, optim_d, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"D_{global_step}.pth")
                )
                if net_dur_disc is not None:
                     unwrapped_net_dur_disc = accelerator.unwrap_model(net_dur_disc)
                     utils.save_checkpoint(
                        unwrapped_net_dur_disc, optim_dur_disc, hps.train.learning_rate, epoch,
                        os.path.join(hps.model_dir, f"DUR_{global_step}.pth")
                    )
                if net_wd is not None:
                    unwrapped_net_wd = accelerator.unwrap_model(net_wd)
                    utils.save_checkpoint(
                        unwrapped_net_wd, optim_wd, hps.train.learning_rate, epoch,
                        os.path.join(hps.model_dir, f"WD_{global_step}.pth")
                    )

                # Clean old checkpoints
                keep_ckpts = config.train_ms_config.keep_ckpts
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True
                    )

        global_step += 1
        # End of batch loop

    if accelerator.is_main_process:
        progress_bar.close()
        if logger: # Check if logger exists
            logger.info(f"====> Finished Epoch: {epoch}")


# --- Evaluation Function (Run on Main Process) ---
# Needs device passed to it
def evaluate(hps, generator, eval_loader, writer_eval, device):
    # generator is the *prepared* model from accelerate
    generator.eval() # Set generator to evaluation mode
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...") # This will print only on the main process
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            (
                x, x_lengths, spec, spec_lengths, y, y_lengths,
                speakers, tone, language, bert, emo
            ) = batch

            # Move data to the device
            x, x_lengths = x.to(device), x_lengths.to(device)
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            speakers = speakers.to(device)
            bert = bert.to(device)
            tone = tone.to(device)
            language = language.to(device)
            emo = emo.to(device)

            # Use generator's infer method (access .module if needed, but try direct first)
            # accelerator.unwrap_model might be needed if infer isn't compatible with DDP wrapper
            # Let's assume direct call works or infer handles the wrapper internally
            unwrapped_generator = accelerator.unwrap_model(generator) # Unwrap for inference consistency
            for use_sdp in [True, False]: # Example loop from original code
                y_hat, attn, mask, *_ = unwrapped_generator.infer( # Use unwrapped model's infer
                    x, x_lengths, speakers, tone, language, bert, emo,
                    y=spec, max_len=1000, sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                # Calculations for logging (on the correct device)
                mel = spec_to_mel_torch(
                    spec, hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(), hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax,
                )

                # Update dictionaries (move results to CPU for numpy/plotting)
                image_dict[f"gen/mel_{batch_idx}_{'sdp' if use_sdp else 'nosdp'}"] = utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
                audio_dict[f"gen/audio_{batch_idx}_{'sdp' if use_sdp else 'nosdp'}"] = y_hat[0, :, : y_hat_lengths[0]] # Keep audio on device? Or move to CPU? .cpu()
                if not use_sdp: # Only log ground truth once
                    image_dict[f"gt/mel_{batch_idx}"] = utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
                    audio_dict[f"gt/audio_{batch_idx}"] = y[0, :, : y_lengths[0]] # .cpu()

    # Summarize results using the main process writer_eval
    utils.summarize(
        writer=writer_eval, global_step=global_step, images=image_dict,
        audios=audio_dict, audio_sampling_rate=hps.data.sampling_rate,
    )
    # No need to call generator.train() here, it's done after the evaluate call in the main loop

# --- Main execution guard ---
if __name__ == "__main__":
    run()
