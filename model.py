import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pywt.data
from functools import partial
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, DropPath


# Note: Please ensure the following custom dependencies are placed in your repository
# from model.lib_mamba.vmambanew import SS2D
# from model import MobileMambaBlock

# ==============================================================================
# 1. Wavelet Spatial-State Model (WSSM) Module
# ==============================================================================

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class WSSM(nn.Module):
    """
    Wavelet Spatial-State Model (WSSM)
    Decouples multi-frequency features and applies dynamic soft-thresholding driven by SSM.
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True,
                 wt_levels=1, wt_type='db1', ssm_ratio=1, forward_type="v05"):
        super(WSSM, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        # --- Wavelet Filters ---
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # --- 2D State Space Model (SS2D) for Global Context ---
        self.global_atten = SS2D(d_model=in_channels, d_state=1, ssm_ratio=ssm_ratio,
                                 initialize="v2", forward_type=forward_type, channel_first=True, k_group=2)

        # --- Dynamic Soft-Thresholding Generators ---
        self.wavelet_convs = nn.ModuleList()
        self.threshold_generators = nn.ModuleList()

        for _ in range(self.wt_levels):
            self.wavelet_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', groups=in_channels * 4,
                              bias=False),
                    nn.BatchNorm2d(in_channels * 4),
                    nn.ReLU(inplace=True)
                )
            )
            # Base multiplier alpha for dynamic thresholding (set to 2.0 empirically)
            self.threshold_generators.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels * 4, kernel_size=1),
                    nn.Sigmoid()
                )
            )

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)])

        # --- Multi-Scale Frequency-Gated Fusion ---
        self.fusion_gate = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.Sigmoid())
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        # Step 1: Establish global continuous meteorological context via SS2D
        global_feat = self.global_atten(x)
        global_feat_scaled = self.base_scale(global_feat)

        # Step 2: Wavelet decomposition and SSM-driven dynamic soft-thresholding
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_convs[i](curr_x_tag)

            # Spatial alignment and dynamic threshold generation
            matched_global = F.interpolate(global_feat, size=curr_x_tag.shape[2:], mode='bilinear', align_corners=False)
            dynamic_tau = self.threshold_generators[i](matched_global) * 2.0  # Base multiplier alpha = 2.0

            # Soft-thresholding to isolate extreme signals from noise
            curr_x_tag = torch.sign(curr_x_tag) * torch.relu(torch.abs(curr_x_tag) - dynamic_tau)

            curr_x_tag = self.wavelet_scale[i](curr_x_tag).reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        # Step 3: Inverse Wavelet Transform
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop() + next_x_ll
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)[:, :, :curr_shape[2], :curr_shape[3]]

        high_freq_features = next_x_ll

        # Step 4: Frequency-Gated Fusion
        gate = self.fusion_gate(high_freq_features)
        out = x + global_feat_scaled * (1 + gate) + high_freq_features

        if self.do_stride is not None: out = self.do_stride(out)
        return out


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)


# ==============================================================================
# 2. Bridge Modules (ECE-EMA & M-ASPP)
# ==============================================================================

class ECE_EMA(nn.Module):
    """ Cross-dimensional interaction attention block """

    def __init__(self, channels, factor=8):
        super(ECE_EMA, self).__init__()
        self.groups = factor if channels // factor != 0 else 1
        self.mid_channels = channels // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.amp = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(self.mid_channels, self.mid_channels)
        self.conv1x1 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x_avg, x_max = self.agp(group_x), self.amp(group_x)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x_global = x_avg + x_max
        x11 = self.softmax(x_global.reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(x_global.reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class SRAB(nn.Module):
    def __init__(self, c_list):
        super().__init__()
        self.attentions = nn.ModuleList(
            [ECE_EMA(c_list[0]), ECE_EMA(c_list[1]), ECE_EMA(c_list[2]), ECE_EMA(c_list[3])])

    def forward(self, t1, t2, t3, t4):
        return self.attentions[0](t1), self.attentions[1](t2), self.attentions[2](t3), self.attentions[3](t4)


class MASPP(nn.Module):
    """ Multi-scale Atrous Spatial Pyramid Pooling (M-ASPP) """

    def __init__(self, in_channels, out_channels):
        super(MASPP, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False), nn.GroupNorm(4, mid_channels),
                                   nn.GELU())
        self.ac1 = self._make_layer(mid_channels, mid_channels, dilation=2)
        self.ac2 = self._make_layer(mid_channels * 2, mid_channels, dilation=4)
        self.ac3 = self._make_layer(mid_channels * 3, mid_channels, dilation=6)
        self.ac4 = self._make_layer(mid_channels * 4, mid_channels, dilation=8)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                                             nn.GroupNorm(4, mid_channels), nn.GELU())
        self.fusion = nn.Sequential(nn.Conv2d(mid_channels * 6, out_channels, 1, bias=False),
                                    nn.GroupNorm(4, out_channels), nn.GELU(), nn.Dropout(0.1))

    def _make_layer(self, in_c, out_c, dilation):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
                             nn.GroupNorm(4, out_c), nn.GELU())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.ac1(x1)
        x3 = self.ac2(torch.cat([x1, x2], dim=1))
        x4 = self.ac3(torch.cat([x1, x2, x3], dim=1))
        x5 = self.ac4(torch.cat([x1, x2, x3, x4], dim=1))
        x_global = F.interpolate(self.global_avg_pool(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        return self.fusion(torch.cat([x1, x2, x3, x4, x5, x_global], dim=1))


# ==============================================================================
# 3. Kinematic-Dynamic Decoupled Predictive Head (KDH)
# ==============================================================================

class HighFreqProjection(nn.Module):
    """ Sub-Pixel High-Frequency Projection to prevent interpolation blurring """

    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        expand_channels = out_channels * (upscale_factor ** 2)
        self.expand_conv = nn.Sequential(nn.Conv2d(in_channels, expand_channels, kernel_size=3, padding=1, bias=False))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.norm = nn.GroupNorm(4, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pixel_shuffle(self.expand_conv(x))))


class SEWeighting(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class TemporalDilatedAdvection(nn.Module):
    """ Physically advects the base frame using synthesized spatial kernels """

    def __init__(self, in_channels, predicted_frames=3, kernel_size=5):
        super().__init__()
        self.predicted_frames = predicted_frames
        self.kernel_size = kernel_size
        self.dynamic_kernel_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.GroupNorm(4, in_channels // 2), nn.SiLU(),
                nn.Conv2d(in_channels // 2, kernel_size ** 2, kernel_size=1)
            ) for _ in range(predicted_frames)
        ])

    def forward(self, feat, base_frame):
        B, _, H, W = feat.shape
        advected_frames = []
        for t in range(self.predicted_frames):
            dilation = t + 1
            pad = dilation * (self.kernel_size - 1) // 2
            spatial_kernel = F.softmax(self.dynamic_kernel_generators[t](feat), dim=1)
            unfolded_base = F.unfold(base_frame, kernel_size=self.kernel_size, padding=pad, dilation=dilation).view(B,
                                                                                                                    self.kernel_size ** 2,
                                                                                                                    H,
                                                                                                                    W)
            advected_t = torch.sum(spatial_kernel * unfolded_base, dim=1, keepdim=True)
            advected_frames.append(advected_t)
        return torch.cat(advected_frames, dim=1)


class KinematicDynamicHead(nn.Module):
    """ Kinematic-Dynamic Decoupled Predictive Head (KDH) """

    def __init__(self, c_list, predicted_frames=3):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.2, dtype=torch.float), requires_grad=True)
        self.highfreq_d1 = HighFreqProjection(c_list[3], c_list[0], upscale_factor=8)
        self.highfreq_d2 = HighFreqProjection(c_list[2], c_list[0], upscale_factor=4)
        self.highfreq_d3 = HighFreqProjection(c_list[1], c_list[0], upscale_factor=2)

        refine_in_dim = c_list[0] * 4
        self.fusion_attention = SEWeighting(refine_in_dim)
        self.refinement = nn.Sequential(nn.Conv2d(refine_in_dim, c_list[0], 3, 1, 1), nn.GroupNorm(4, c_list[0]),
                                        nn.SiLU())

        self.advection_branch = TemporalDilatedAdvection(in_channels=c_list[0], predicted_frames=predicted_frames,
                                                         kernel_size=5)
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[0], kernel_size=5, padding=2, groups=c_list[0]),
            nn.GroupNorm(4, c_list[0]), nn.SiLU(), nn.Conv2d(c_list[0], predicted_frames, kernel_size=3, padding=1))
        self.high_freq_branch = nn.Sequential(nn.Conv2d(c_list[0], c_list[0], kernel_size=1),
                                              nn.GroupNorm(4, c_list[0]), nn.SiLU(),
                                              nn.Conv2d(c_list[0], predicted_frames, kernel_size=3, padding=1))

    def forward(self, d1, d2, d3, d4, base_frame):
        # High-Frequency Projection & Fusion
        feats = self.fusion_attention(
            torch.cat([self.highfreq_d1(d1), self.highfreq_d2(d2), self.highfreq_d3(d3), d4], dim=1))
        feat = self.refinement(feats)

        # Background Advection + Local Residual Dynamics
        advected_base = self.advection_branch(feat, base_frame)
        res = self.low_freq_branch(feat) + self.high_freq_branch(feat)
        final = advected_base + res

        # Adaptive Precipitation Gating
        return final * torch.sigmoid(self.beta * final)


# ==============================================================================
# 4. Final Architecture: WKD-Net
# ==============================================================================

class TransposedUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(4, out_channels)
        self.act = nn.GELU()

    def forward(self, x): return self.act(self.norm(self.up(x)))


# Dummy adapters for missing external dependencies (replace with your actual code)
class MobileMambaAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, stage_idx):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x): return self.conv(x)


class WKD_Net(nn.Module):
    """
    WKD-Net: Wavelet Kinematic-Dynamic State Space Model
    for High-Intensity Precipitation Nowcasting
    """

    def __init__(self, predicted_frames=3, input_frames=5, c_list=[16, 32, 64, 128, 256], bridge=True):
        super().__init__()
        self.bridge = bridge

        # --- Encoder ---
        self.encoder1 = nn.Sequential(nn.Conv2d(input_frames, c_list[0], 3, 1, 1), nn.GroupNorm(4, c_list[0]),
                                      nn.GELU())
        self.encoder2 = nn.Sequential(nn.Conv2d(c_list[0], c_list[1], 3, 2, 1), nn.GroupNorm(4, c_list[1]), nn.GELU())
        self.encoder3 = nn.Sequential(nn.Conv2d(c_list[1], c_list[2], 3, 2, 1), nn.GroupNorm(4, c_list[2]), nn.GELU())
        self.down4 = nn.Conv2d(c_list[2], c_list[2], 3, 2, 1)

        # In a real run, these should be WSSM / Mamba blocks
        self.encoder4 = MobileMambaAdapter(c_list[2], c_list[3], stage_idx=0)
        self.down5 = nn.Conv2d(c_list[3], c_list[3], 3, 2, 1)
        self.encoder5 = MobileMambaAdapter(c_list[3], c_list[4], stage_idx=1)

        # --- Bridge ---
        self.scab = SRAB(c_list) if bridge else None
        self.aspp = MASPP(c_list[4], c_list[4])

        # --- Decoder ---
        self.up1 = TransposedUpBlock(c_list[4], c_list[3])
        self.reduce1 = nn.Conv2d(c_list[3] * 2, c_list[3], 1)
        self.decoder1 = MobileMambaAdapter(c_list[3], c_list[3], stage_idx=1)

        self.up2 = TransposedUpBlock(c_list[3], c_list[2])
        self.reduce2 = nn.Conv2d(c_list[2] * 2, c_list[2], 1)
        self.decoder2 = MobileMambaAdapter(c_list[2], c_list[2], stage_idx=0)

        self.up3 = TransposedUpBlock(c_list[2], c_list[1])
        self.reduce3 = nn.Conv2d(c_list[1] * 2, c_list[1], 1)
        self.decoder3 = nn.Sequential(nn.Conv2d(c_list[1], c_list[1], 3, 1, 1), nn.GroupNorm(4, c_list[1]), nn.GELU())

        self.up4 = TransposedUpBlock(c_list[1], c_list[0])
        self.reduce4 = nn.Conv2d(c_list[0] * 2, c_list[0], 1)
        self.decoder4 = nn.Sequential(nn.Conv2d(c_list[0], c_list[0], 3, 1, 1), nn.GroupNorm(4, c_list[0]), nn.GELU())

        # --- Kinematic-Dynamic Predictive Head ---
        self.prediction_head = KinematicDynamicHead(c_list, predicted_frames)

    def forward(self, x):
        # Base frame for physical advection [B, 1, H, W]
        base_frame = x[:, -1, ...].unsqueeze(1)

        # Encode
        t1 = self.encoder1(x)
        t2 = self.encoder2(t1)
        t3 = self.encoder3(t2)
        t4 = self.encoder4(self.down4(t3))
        out = self.encoder5(self.down5(t4))

        # Bridge
        if self.bridge and self.scab:
            t1, t2, t3, t4 = self.scab(t1, t2, t3, t4)
        out = self.aspp(out)

        # Decode
        out1 = self.up1(out)
        if out1.shape[2:] != t4.shape[2:]: out1 = F.interpolate(out1, size=t4.shape[2:])
        d1_out = self.decoder1(self.reduce1(torch.cat([out1, t4], dim=1)))

        out2 = self.up2(d1_out)
        if out2.shape[2:] != t3.shape[2:]: out2 = F.interpolate(out2, size=t3.shape[2:])
        d2_out = self.decoder2(self.reduce2(torch.cat([out2, t3], dim=1)))

        out3 = self.up3(d2_out)
        if out3.shape[2:] != t2.shape[2:]: out3 = F.interpolate(out3, size=t2.shape[2:])
        d3_out = self.decoder3(self.reduce3(torch.cat([out3, t2], dim=1)))

        out4 = self.up4(d3_out)
        if out4.shape[2:] != t1.shape[2:]: out4 = F.interpolate(out4, size=t1.shape[2:])
        d4_out = self.decoder4(self.reduce4(torch.cat([out4, t1], dim=1)))

        # Predict
        final_prediction = self.prediction_head(d1_out, d2_out, d3_out, d4_out, base_frame)
        return final_prediction