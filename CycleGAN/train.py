"""
    Training for CycleGAN

    Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
    * 2020-11-05: Initial coding
    * 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import config
from utils import save_checkpoint, load_checkpoint, save_scheduler, load_scheduler
from dataset import CycleGAN_Dataset
from discriminator_model import Discriminator
from generator_model import Generator

def train_one_epoch(disc_T, disc_S, gen_S, gen_T, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    ''' GT: S -> T
        GS: T -> S
        DT: distinguish T from fake GT(S) and real T
        DS: distinguish S from fake GS(T) and real S
    '''
    T_reals = 0
    T_fakes = 0
    
    loop = tqdm(loader, leave=True)

    for idx, (source, target) in enumerate(loop):
        source = source.to(config.DEVICE)
        target = target.to(config.DEVICE)

        # Train Discriminators T and S
        with torch.cuda.amp.autocast():
            # T' = GT(S)
            fake_target = gen_T(source)
            
            # Patch: DT(T) and DT(GT(S))
            DT_real = disc_T(target)
            DT_fake = disc_T(fake_target.detach())
            
            # Accumulated scalar: DT(T) and DT(GT(S))
            T_reals += DT_real.mean().item()
            T_fakes += DT_fake.mean().item()
            
            # L_GAN(GT,DT,S,T) = minimize Et[(DT(t) - 1)^2] + Es[DT(GT(s))^2]
            DT_real_loss = mse(DT_real, torch.ones_like(DT_real))
            DT_fake_loss = mse(DT_fake, torch.zeros_like(DT_fake))
            
            # L_GAN(GT,DT,S,T)
            DT_loss = DT_real_loss + DT_fake_loss
            
            # S' = GS(T)
            fake_source = gen_S(target)
            
            # Patch: DS(S) and DS(GS(T)) 
            DS_real = disc_S(source)
            DS_fake = disc_S(fake_source.detach())
            
            # L_GAN(GS,DS,S,T) = minimize Es[(DS(s) - 1)^2] + Et[DS(GS(t))^2]
            DS_real_loss = mse(DS_real, torch.ones_like(DS_real))
            DS_fake_loss = mse(DS_fake, torch.zeros_like(DS_fake))
            
            # L_GAN(GS,DS,S,T) 
            DS_loss = DS_real_loss + DS_fake_loss

            # put it togethor: L_GAN(GT,DT,S,T) + L_GAN(GS,DS,S,T) 
            D_loss = (DT_loss + DS_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators T and Generators S
        with torch.cuda.amp.autocast():
            
            # adversarial loss for both generators
            # Patch: DT(GT(S)) and DS(GS(T)) 
            DT_fake = disc_T(fake_target)
            DS_fake = disc_S(fake_source)
            
            # minimize Et[(DT(t) - 1)^2] and Es[(DS(s) - 1)^2]
            loss_GT = mse(DT_fake, torch.ones_like(DT_fake))
            loss_GS = mse(DS_fake, torch.ones_like(DS_fake))

            # cycle loss
            # GS(GT(S)) -> S'
            cycle_source = gen_S(fake_target)
            # GT(GS(T)) -> T'
            cycle_target = gen_T(fake_source)
            
            # Lcyc(GS,GT) = Es[|GS(GT(s)) - s|] + Et[|GT(GS(t)) - t|]
            cycle_source_loss = l1(source, cycle_source)
            cycle_target_loss = l1(target, cycle_target)

            # add all togethor: L(GS,GT,DS,DT) = LGAN(GT,DT,X,Y) + LGAN(GS,DS,X,Y) + Lcyc(GS,GT)
            λ = config.LAMBDA_CYCLE
            G_loss = loss_GT + loss_GS + cycle_source_loss * λ + cycle_target_loss * λ

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
            
        if idx % 100 == 0:
            # denormalize
            for i in range(source.shape[0]):
                save_image(fake_target[i] * 0.5 + 0.5, f"{config.IMAGE_SAVED_DIR}/trans_source_{idx}_{i}.png")
                save_image(fake_source[i] * 0.5 + 0.5, f"{config.IMAGE_SAVED_DIR}/trans_target_{idx}_{i}.png")
                save_image(target[i] * 0.5 + 0.5, f"{config.IMAGE_SAVED_DIR}/target_{idx}_{i}.png")
                save_image(source[i] * 0.5 + 0.5, f"{config.IMAGE_SAVED_DIR}/source_{idx}_{i}.png")

        loop.set_postfix(H_real=T_reals / (idx + 1), H_fake=T_fakes / (idx + 1))

def main():
    resume   = config.RESUME
    stage    = config.LOAD_MODEL_STAGE
    max_e    = config.NUM_EPOCHS 
    resume_e = 0
    
    # dataset
    dataset = CycleGAN_Dataset(
        root_t=config.TRAIN_DIR + "/targets",
        root_s=config.TRAIN_DIR + "/sources",
        trans=config.transforms,
        start=config.DATA_START,
        end=config.DATA_END
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    # Networks
    disc_T = Discriminator(in_channels=3).to(config.DEVICE)
    disc_S = Discriminator(in_channels=3).to(config.DEVICE)
    
    gen_S = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_T = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    # Learning rate
    opt_disc = optim.Adam(
        list(disc_T.parameters()) + list(disc_S.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_S.parameters()) + list(gen_T.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    
    sch_disc_decay = optim.lr_scheduler.CosineAnnealingLR(opt_disc, max_e)
    sch_gen_decay  = optim.lr_scheduler.CosineAnnealingLR(opt_gen, max_e)

    # Loss function
    L1  = nn.L1Loss()
    mse = nn.MSELoss()

    # Resume?
    if resume:
        if stage == 1:
            load_checkpoint(config.CHECKPOINT_GEN_T, gen_T, opt_gen, config.LEARNING_RATE)
            load_checkpoint(config.CHECKPOINT_GEN_S, gen_S, opt_gen, config.LEARNING_RATE)
            load_checkpoint(config.CHECKPOINT_CRITIC_T, disc_T, opt_disc, config.LEARNING_RATE)
            resume_e = load_checkpoint(config.CHECKPOINT_CRITIC_S, disc_S, opt_disc, config.LEARNING_RATE) % 100
        else:
            load_checkpoint(config.CHECKPOINT_GEN_T, gen_T, opt_gen, config.LEARNING_RATE)
            load_checkpoint(config.CHECKPOINT_GEN_S, gen_S, opt_gen, config.LEARNING_RATE)
            load_checkpoint(config.CHECKPOINT_CRITIC_T, disc_T, opt_disc, config.LEARNING_RATE)
            resume_e = load_checkpoint(config.CHECKPOINT_CRITIC_S, disc_S, opt_disc, config.LEARNING_RATE) % 100
            
            load_scheduler(config.CHECKPOINT_schLR_D, sch_disc_decay)
            load_scheduler(config.CHECKPOINT_schLR_G, sch_gen_decay)

    # float16 setup
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for e in range(resume_e if resume and stage == 1 else 0, 0 if resume and stage == 2 else max_e):
        print(f'stage 1 - epoch {e}')
        train_one_epoch(disc_T, disc_S, gen_S, gen_T, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        if config.SAVE_MODEL:
            save_checkpoint(e + 1, gen_T, opt_gen, filename=config.CHECKPOINT_GEN_T)
            save_checkpoint(e + 1, gen_S, opt_gen, filename=config.CHECKPOINT_GEN_S)
            save_checkpoint(e + 1, disc_T, opt_disc, filename=config.CHECKPOINT_CRITIC_T)
            save_checkpoint(e + 1, disc_S, opt_disc, filename=config.CHECKPOINT_CRITIC_S)
    
    for e in range(resume_e if resume and stage == 2 else 0, max_e):
        print(f'stage 2 - epoch {e}')
        train_one_epoch(disc_T, disc_S, gen_S, gen_T, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        sch_disc_decay.step()
        sch_gen_decay.step()
        if config.SAVE_MODEL:
            save_checkpoint(e + 1, gen_T, opt_gen, filename=config.CHECKPOINT_GEN_T)
            save_checkpoint(e + 1, gen_S, opt_gen, filename=config.CHECKPOINT_GEN_S)
            save_checkpoint(e + 1, disc_T, opt_disc, filename=config.CHECKPOINT_CRITIC_T)
            save_checkpoint(e + 1, disc_S, opt_disc, filename=config.CHECKPOINT_CRITIC_S)
            
            save_scheduler(sch_disc_decay, config.CHECKPOINT_schLR_D)
            save_scheduler(sch_gen_decay, config.CHECKPOINT_schLR_G)

if __name__ == "__main__":
    main()
