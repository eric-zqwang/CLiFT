import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def compute_Perceptual_Loss(self, pred_img, real_img, mask):
    def compute_error(real, fake):
        E = torch.mean(torch.abs(real - fake) )
        return E

    aa = np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
    bb = Variable(torch.from_numpy(aa).float().permute(0,3,1,2).cuda(1))

    real_img_sb = real_img * mask * 255. - bb
    pred_img_sb = pred_img * mask * 255. - bb

    out3_r, out8_r, out13_r, out22_r, out33_r = self.Net(real_img_sb, return_style=0)
    out3_f, out8_f, out13_f, out22_f, out33_f = self.Net(pred_img_sb, return_style=0)

    E0 = compute_error(real_img_sb, pred_img_sb)
    # mask_3 = nn.functional.interpolate(mask, size=[out3_r.size(2), out3_r.size(3)], mode='nearest')
    E1 = compute_error(out3_r, out3_f)/2.6
    # mask_8 = nn.functional.interpolate(mask, size=[out8_r.size(2), out8_r.size(3)], mode='nearest')
    E2 = compute_error(out8_r, out8_f)/4.8
    # mask_13 = nn.functional.interpolate(mask, size=[out13_r.size(2), out13_r.size(3)], mode='nearest')
    E3 = compute_error(out13_r, out13_f)/3.7
    # mask_22 = nn.functional.interpolate(mask, size=[out22_r.size(2), out22_r.size(3)], mode='nearest')
    E4 = compute_error(out22_r, out22_f)/5.6
    # mask_33 = nn.functional.interpolate(mask, size=[out33_r.size(2), out33_r.size(3)], mode='nearest')
    E5 = compute_error(out33_r, out33_f)*10/ 1.5

    total_loss = (E0+E1+E2+E3+E4+E5)/255.

    return total_loss