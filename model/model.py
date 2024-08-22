"""
File containing the main model.
"""

# Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import random
import torch.nn.functional as F


# Local imports
from model.modules import (
    BaseRGBModel,
    EDSGPMIXERLayers,
    FCLayers,
    step,
    process_prediction,
    MambaBlock,
)
from model.shift import make_temporal_shift


class TDEEDModel(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._modality = args.modality
            assert self._modality == "rgb", "Only RGB supported for now"
            in_channels = {"rgb": 3}[self._modality]
            self._d = 512  # initialized to 512
            self._temp_arch = args.temporal_arch
            # assert self._temp_arch == 'ed_sgp_mixer', 'Only ed_sgp_mixer supported for now'
            self._radi_displacement = args.radi_displacement
            self._feature_arch = args.feature_arch
            assert "rny" in self._feature_arch, "Only rny supported for now"

            if self._feature_arch.startswith(("rny002", "rny008")):
                features = timm.create_model(
                    {
                        "rny002": "regnety_002",
                        "rny008": "regnety_008",
                    }[self._feature_arch.rsplit("_", 1)[0]],
                    pretrained=True,
                )
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            # Add Temporal Shift Modules
            self._require_clip_len = -1
            if self._feature_arch.endswith("_gsm"):
                make_temporal_shift(features, args.clip_len, mode="gsm")
                self._require_clip_len = args.clip_len
            elif self._feature_arch.endswith("_gsf"):
                make_temporal_shift(features, args.clip_len, mode="gsf")
                self._require_clip_len = args.clip_len

            self._features = features
            self._feat_dim = self._d
            feat_dim = self._d

            # Feature extractor for position regression

            self.__cnn_features = None

            def get_feature_map():
                def hook_fn(module, input, output):
                    self.__cnn_features = output

                return hook_fn

            self.__hook = self._features.final_conv.register_forward_hook(
                get_feature_map()
            )
            self._pos_c = 368
            self._P = 7

            # Positional encoding

            if self._temp_arch == "ed_sgp_mixer":
                self.temp_enc = nn.Parameter(
                    torch.normal(
                        mean=0, std=1 / args.clip_len, size=(args.clip_len, self._d)
                    )
                )
                self._temp_fine = EDSGPMIXERLayers(
                    feat_dim,
                    args.clip_len,
                    num_layers=args.n_layers,
                    ks=args.sgp_ks,
                    k=args.sgp_r,
                    concat=True,
                )

            elif self._temp_arch == "transformer_enc_only_base_11m":
                from x_transformers import Encoder
                from positional_encodings.torch_encodings import (
                    PositionalEncoding1D,
                    Summer,
                )

                self.temp_enc = Summer(PositionalEncoding1D(self._feat_dim))
                encoder_layer = nn.TransformerEncoderLayer(
                    self._feat_dim, nhead=8, batch_first=True
                )
                self._temp_fine = nn.TransformerEncoder(encoder_layer, num_layers=5)
                # self._temp_fine = Encoder(
                #     dim = self._feat_dim,
                #     depth = 5,
                #     heads = 8,
                #     attn_flash = True,
                # )

            elif self._temp_arch == "transformer_dec_only_base_11m":
                from x_transformers import Encoder
                from positional_encodings.torch_encodings import (
                    PositionalEncoding1D,
                    Summer,
                )

                h_dim = 256
                self._down_projection = nn.Linear(self._feat_dim, h_dim)
                self.temp_enc = Summer(PositionalEncoding1D(h_dim))
                decoder_layer = nn.TransformerDecoderLayer(
                    h_dim, nhead=8, batch_first=True
                )
                self._temp_fine = nn.TransformerDecoder(decoder_layer, num_layers=5)
                self._temp_queries = nn.Parameter(torch.rand(1, 1, h_dim))

                self._feat_dim = h_dim

            elif self._temp_arch == "mamba_1":
                from mamba_ssm import Mamba

                self.temp_enc = nn.Identity()
                self._temp_fine = MambaBlock(
                    dim=self._feat_dim,
                    dropout=0.25,
                )

                # self._temp_fine = Mamba(
                #     # This module uses roughly 3 * expand * d_model^2 parameters
                #     d_model=self._feat_dim,  # Model dimension d_model
                #     d_state=16,  # SSM state expansion factor
                #     d_conv=4,  # Local convolution width
                #     expand=2,  # Block expansion factor
                # ).to("cuda")

            elif self._temp_arch == "mamba_multi":
                from mamba_ssm import Mamba

                # self.temp_enc = nn.Identity()
                self.temp_enc = nn.Parameter(
                    torch.normal(
                        mean=0, std=1 / args.clip_len, size=(args.clip_len, self._d)
                    )
                )

                self._temp_fine = nn.Sequential(
                    *[
                        MambaBlock(
                            dim=self._feat_dim,
                        )
                        for _ in range(3)
                    ]
                )

            else:
                raise NotImplementedError(self._temp_arch)

            self._pred_fine = FCLayers(self._feat_dim, args.num_classes + 1)

            if args.predict_location:
                # self._pred_loc = FCLayers(self._feat_dim, 2)
                self._pred_loc = nn.Conv2d(
                    in_channels=self._pos_c,
                    out_channels=3,
                    kernel_size=1,
                    stride=1,
                    padding="same",
                )

            if self._radi_displacement > 0:
                self._pred_displ = FCLayers(self._feat_dim, 1)

            # Augmentations and crop
            self.augmentation = T.Compose(
                [
                    T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                    T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                    T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                    T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                    T.RandomApply([T.GaussianBlur(5)], p=0.25),
                    (
                        torch.nn.Identity()
                        if args.predict_location
                        else T.RandomHorizontalFlip()
                    ),
                ]
            )

            # Standarization
            self.standarization = T.Compose(
                [
                    T.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    )  # Imagenet mean and std
                ]
            )

            # Augmentation at test time
            self.augmentationI = T.Compose(
                [
                    (
                        torch.nn.Identity()
                        if args.predict_location
                        else T.RandomHorizontalFlip(p=1.0)
                    )
                ]
            )

            # Croping in case of using it
            self.croping = args.crop_dim
            if self.croping != None and args.predict_location == False:
                self.cropT = T.RandomCrop((self.croping, self.croping))
                self.cropI = T.CenterCrop((self.croping, self.croping))
            else:
                self.cropT = torch.nn.Identity()
                self.cropI = torch.nn.Identity()

            self.img_size = 224  # TODO: parametrize this
            self.resize = T.Resize((self.img_size, self.img_size), antialias=True)

        def get_activation_map(self):
            return self.__cnn_features

        def remove_hook(self):
            self.__hook.remove()

        def forward(self, x, y=None, inference=False, augment_inference=False):

            B, T, C, _, _ = x.shape
            x = self.normalize(x)  # Normalize to 0-1
            x = self.resize(x.flatten(0, 1)).view(B, T, C, self.img_size, self.img_size)

            batch_size, true_clip_len, channels, height, width = x.shape

            if not inference:
                x = x.view(-1, channels, height, width)
                x = self.augment(x)  # augmentation per-batch
                x = self.standarize(x)  # standarization imagenet stats
                # if self.croping != None:
                #     height = self.croping
                #     width = self.croping
                # x = self.cropT(x) #same crop for all frames
                x = x.view(batch_size, true_clip_len, channels, height, width)

            else:
                x = x.view(-1, channels, height, width)
                if augment_inference:
                    x = self.augmentI(x)
                x = self.standarize(x)
                # if self.croping != None:
                #     height = self.croping
                #     width = self.croping
                # x = self.cropI(x) #same center crop for all frames
                x = x.view(batch_size, true_clip_len, channels, height, width)
            clip_len = true_clip_len

            im_feat = self._features(x.view(-1, channels, height, width)).reshape(
                batch_size, clip_len, self._d
            )

            if self._temp_arch == "ed_sgp_mixer" or "mamba_multi" in self._temp_arch:
                im_feat = im_feat + self.temp_enc.expand(batch_size, -1, -1)

            elif "_dec_" not in self._temp_arch:
                im_feat = self.temp_enc(im_feat)

            if "_dec_" in self._temp_arch:
                im_feat = self._down_projection(im_feat)
                im_feat = self.temp_enc(im_feat)
                im_feat = self._temp_fine(
                    self._temp_queries.expand(batch_size, true_clip_len, -1), im_feat
                )
            else:
                im_feat = self._temp_fine(im_feat)

            if hasattr(self, "_pred_loc"):
                cnn_ft = self.get_activation_map()
                loc_feat = self._pred_loc(cnn_ft)
                # print(F"Activation map: {cnn_ft.shape}, loc_feat: {loc_feat.shape}")
                # loc_feat = None
            else:
                loc_feat = None

            if self._radi_displacement > 0:
                displ_feat = self._pred_displ(im_feat).squeeze(-1)
                im_feat = self._pred_fine(im_feat)
                return {
                    "im_feat": im_feat,
                    "displ_feat": displ_feat,
                    "loc_feat": loc_feat,
                }, y

            im_feat = self._pred_fine(im_feat)

            return {"im_feat": im_feat, "loc_feat": loc_feat}, y

            # if self._temp_arch == 'ed_sgp_mixer':
            #     im_feat = self._temp_fine(im_feat)
            #     if self._radi_displacement > 0:
            #         displ_feat = self._pred_displ(im_feat).squeeze(-1)
            #         im_feat = self._pred_fine(im_feat)
            #         return {'im_feat': im_feat, 'displ_feat': displ_feat}, y
            #     im_feat = self._pred_fine(im_feat)
            #     return im_feat, y

            # else:
            #     raise NotImplementedError(self._temp_arch)

        def normalize(self, x):
            return x / 255.0

        def augment(self, x):
            # print(f"Augmenting: {x.shape}")
            return self.augmentation(x)
            # for i in range(x.shape[0]):
            #     x[i] = self.augmentation(x[i])
            # return x

        def augmentI(self, x):
            return self.augmentationI(x)
            # for i in range(x.shape[0]):
            #     x[i] = self.augmentationI(x[i])
            # return x

        def standarize(self, x):
            return self.standarization(x)
            # for i in range(x.shape[0]):
            #     x[i] = self.standarization(x[i])
            # return x

        def print_stats(self):
            print(
                f"Model params: {sum(p.numel() for p in self.parameters()):,}",
            )
            print(
                f"  CNN features: {sum(p.numel() for p in self._features.parameters()):,}",
            )
            print(
                f"  Temporal: {sum(p.numel() for p in self._temp_fine.parameters()):,}",
            )
            print(
                f"  Head: {sum(p.numel() for p in self._pred_fine.parameters()):,}",
            )

    def __init__(self, device="cuda", args=None):
        self.device = device
        self._model = TDEEDModel.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(device)
        self._num_classes = args.num_classes + 1

    def epoch(
        self,
        loader,
        optimizer=None,
        scaler=None,
        lr_scheduler=None,
        acc_grad_iter=1,
        fg_weight=5,
    ):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs["weight"] = torch.FloatTensor(
                [1] + [fg_weight] * (self._num_classes - 1)
            ).to(self.device)

        epoch_loss = 0.0
        epoch_lossD = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_loc = 0.0

        with torch.no_grad() if optimizer is None else nullcontext():
            pbar = tqdm(loader)
            for batch_idx, batch in enumerate(pbar):
                frame = batch["frame"].to(self.device).float()
                label = batch["label"]
                label = label.to(self.device)

                if "labelD" in batch.keys():
                    labelD = batch["labelD"].to(self.device).float()

                if "frame2" in batch.keys():
                    frame2 = batch["frame2"].to(self.device).float()
                    label2 = batch["label2"]
                    label2 = label2.to(self.device)

                    if "labelD2" in batch.keys():
                        labelD2 = batch["labelD2"].to(self.device).float()
                        labelD_dist = torch.zeros((labelD.shape[0], label.shape[1])).to(
                            self.device
                        )

                    l = [random.betavariate(0.2, 0.2) for _ in range(frame2.shape[0])]

                    label_dist = torch.zeros(
                        (label.shape[0], label.shape[1], self._num_classes)
                    ).to(self.device)

                    for i in range(frame2.shape[0]):
                        frame[i] = l[i] * frame[i] + (1 - l[i]) * frame2[i]
                        lbl1 = label[i]
                        lbl2 = label2[i]

                        label_dist[i, range(label.shape[1]), lbl1] += l[i]
                        label_dist[i, range(label2.shape[1]), lbl2] += 1 - l[i]

                        if "labelD2" in batch.keys():
                            labelD_dist[i] = l[i] * labelD[i] + (1 - l[i]) * labelD2[i]

                    label = label_dist
                    if "labelD2" in batch.keys():
                        labelD = labelD_dist

                # Depends on whether mixup is used
                label = (
                    label.flatten()
                    if len(label.shape) == 2
                    else label.view(-1, label.shape[-1])
                )

                with torch.cuda.amp.autocast():
                    pred, y = self._model(frame, y=label, inference=inference)

                    if "labelD" in batch.keys():
                        predD = pred["displ_feat"]

                    if "loc_feat" in pred.keys():
                        pred_loc = pred["loc_feat"]

                    pred = pred["im_feat"]

                    loss = 0.0

                    if len(pred.shape) == 3:
                        pred = pred.unsqueeze(0)

                    for i in range(pred.shape[0]):
                        predictions = pred[i].reshape(-1, self._num_classes)
                        loss_ce = F.cross_entropy(predictions, label, **ce_kwargs)
                        loss += loss_ce
                        epoch_loss_ce += loss_ce.detach().item()

                    if pred_loc is not None:
                        from util.det import convert_target_to_prediction_shape

                        target_xy = convert_target_to_prediction_shape(
                            target=batch["xy"].float(), P=self._model._P
                        )
                        # print(f"pred_loc: {pred_loc.shape}, target_xy: {target_xy.shape}")

                        loss_loc = F.l1_loss(
                            pred_loc.reshape(-1, 3),
                            target_xy.to(self.device).reshape(-1, 3).float(),
                            reduction="mean",
                        )
                        loss += 5 * loss_loc
                        epoch_loss_loc += loss_loc.detach().item()

                    if "labelD" in batch.keys():
                        # print(f"predD: {predD.shape}, labelD: {labelD.shape}")
                        lossD = F.mse_loss(predD, labelD, reduction="none")
                        lossD = (lossD).mean()
                        loss = loss + lossD

                if optimizer is not None:
                    step(
                        optimizer,
                        scaler,
                        loss / acc_grad_iter,
                        lr_scheduler=lr_scheduler,
                        backward_only=(batch_idx + 1) % acc_grad_iter != 0,
                    )

                epoch_loss += loss.detach().item()

                if "labelD" in batch.keys():
                    epoch_lossD += lossD.detach().item()
                # torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                pbar.set_postfix({"loss": epoch_loss / (batch_idx + 1)})

        ret = {"loss": epoch_loss / len(loader), "loss_ce": epoch_loss_ce / len(loader)}
        ret["ce"] = epoch_loss_ce / len(loader)
        ret["lossD"] = epoch_lossD / len(loader)
        ret["loss_loc"] = epoch_loss_loc / len(loader)
        return ret

    def predict(self, seq, use_amp=True, augment_inference=False):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                pred, y = self._model(
                    seq, inference=True, augment_inference=augment_inference
                )
            if isinstance(pred, dict):
                predD = pred["displ_feat"]
                pred = pred["im_feat"]
                pred = process_prediction(pred, predD)
                pred_cls = torch.argmax(pred, axis=2)
                return pred_cls.cpu().numpy(), pred.cpu().numpy()
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            else:
                pred = torch.softmax(pred, axis=2)

            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()
