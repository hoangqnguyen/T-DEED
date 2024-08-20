"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import random
import torch.nn.functional as F


#Local imports
from model.modules import BaseRGBModel, EDSGPMIXERLayers, FCLayers, step, process_prediction, MambaBlock
from model.shift import make_temporal_shift

from util.det import convert_target_to_prediction_shape, visualize_prediction_grid

import model.mamba_core as mamba_core

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)


class MambaSpotModel(BaseRGBModel):

    def __init__(self, device='cuda', args=None):
        self.device = device
        # self._model = MambaSpotModel.Impl(args=args)

        self._model = mamba_core.mambaspotcore_tiny(
            args, num_classes=args.num_classes, num_frames=args.clip_len,
            fused_add_norm=True)

        self.P = 14 # TODO: set it as 14 first, paramterize later

        self._model.print_stats()
        self._args = args

        self._model.to(device)
        self._num_classes = args.num_classes + 1

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
            acc_grad_iter=1, fg_weight=5):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor(
                [1] + [fg_weight] * (self._num_classes - 1)).to(self.device)

        epoch_loss = 0.
        epoch_lossD = 0.
        epoch_loss_ce = 0.
        epoch_loss_loc = 0.

        with torch.no_grad() if optimizer is None else nullcontext():
            pbar = tqdm(loader)
            for batch_idx, batch in enumerate(pbar):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device)

                if 'labelD' in batch.keys():
                    labelD = batch['labelD'].to(self.device).float()
                
                if 'frame2' in batch.keys():
                    frame2 = batch['frame2'].to(self.device).float()
                    label2 = batch['label2']
                    label2 = label2.to(self.device)

                    if 'labelD2' in batch.keys():
                        labelD2 = batch['labelD2'].to(self.device).float()
                        labelD_dist = torch.zeros((labelD.shape[0], label.shape[1])).to(self.device)

                    l = [random.betavariate(0.2, 0.2) for _ in range(frame2.shape[0])]

                    label_dist = torch.zeros((label.shape[0], label.shape[1], self._num_classes)).to(self.device)

                    for i in range(frame2.shape[0]):
                        frame[i] = l[i] * frame[i] + (1 - l[i]) * frame2[i]
                        lbl1 = label[i]
                        lbl2 = label2[i]

                        label_dist[i, range(label.shape[1]), lbl1] += l[i]
                        label_dist[i, range(label2.shape[1]), lbl2] += 1 - l[i]

                        if 'labelD2' in batch.keys():
                            labelD_dist[i] = l[i] * labelD[i] + (1 - l[i]) * labelD2[i]

                    label = label_dist
                    if 'labelD2' in batch.keys():
                        labelD = labelD_dist

                # Depends on whether mixup is used
                label = label.flatten() if len(label.shape) == 2 \
                    else label.view(-1, label.shape[-1])

                with torch.cuda.amp.autocast():
                    pred, y = self._model(frame, y = label, inference=inference)

                    if 'labelD' in batch.keys():
                        predD = pred['displ_feat']
                    
                    if 'loc_feat' in pred.keys():
                        pred_loc = pred['loc_feat']

                    pred = pred['im_feat']

                    loss = 0.

                    if len(pred.shape) == 3:
                        pred = pred.unsqueeze(0)

                    for i in range(pred.shape[0]):
                        predictions = pred[i].reshape(-1, self._num_classes)
                        loss_ce = F.cross_entropy(
                            predictions, label,
                            **ce_kwargs)
                        if torch.isnan(loss_ce).any():
                            breakpoint() # TODO: debug why loss is nan
                        epoch_loss_ce += loss_ce.detach().item()
                        loss += loss_ce
            
                    if pred_loc is not None:
                        target_xy = convert_target_to_prediction_shape(target=batch['xy'].float(), P=self.P).reshape(-1, 3) 
                        loss_loc = F.l1_loss(pred_loc.reshape(-1,3), target_xy.to(self.device).float(), reduction='mean')                         
                        loss += loss_loc
                        epoch_loss_loc += loss_loc.detach().item()
                        
                    if 'labelD' in batch.keys():
                        lossD = F.mse_loss(predD, labelD, reduction = 'none')
                        lossD = (lossD).mean()
                        loss = loss + lossD

                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter,
                        lr_scheduler=lr_scheduler,
                        backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()

                if 'labelD' in batch.keys():
                    epoch_lossD += lossD.detach().item()
                # torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                pbar.set_postfix({"loss": epoch_loss / (batch_idx + 1)})

        ret =  {'loss': epoch_loss / len(loader), 'loss_ce': epoch_loss_ce / len(loader)}
        ret['ce'] = epoch_loss_ce / len(loader)
        ret['lossD'] = epoch_lossD / len(loader)
        ret['loss_loc'] = epoch_loss_loc / len(loader)
        return ret

    def predict(self, seq, use_amp=True, augment_inference = False):
        
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                pred, y = self._model(seq, inference=True, augment_inference = augment_inference)
            if isinstance(pred, dict):
                predD = pred['displ_feat']
                pred = pred['im_feat']
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
        
    def predict2(self, seq, use_amp=True, augment_inference=False):
        # TODO: test this function by visualizing the predictions!!!
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
                pred, y = self._model(seq, inference=True, augment_inference=augment_inference)
            
            # Extract the necessary features, checking if they are None
            predD = pred.get('displ_feat')
            pred_im = pred['im_feat']
            locations = pred.get('loc_feat')

            # Process the prediction if displacement feature is available
            if predD is not None:
                pred_processed = process_prediction(pred_im, predD)
            else:
                pred_processed = torch.softmax(pred_im, axis=2)  # Assuming you want to apply softmax if no processing is needed
            
            # Calculate predicted classes
            pred_cls = torch.argmax(pred_processed, axis=2)

            # Convert all to numpy, handle the case where locations might be None
            pred_cls_np = pred_cls.cpu().numpy()
            pred_processed_np = pred_processed.cpu().numpy()
            locations_np = locations.cpu().numpy() if locations is not None else None
            
            return pred_cls_np, pred_processed_np, locations_np
