import torch
import torch.nn as nn


# ============================================================
# Patch Embedding
# ============================================================

class PatchEmbed(nn.Module):
    """
    将 224x224 的图片切成 16x16 的 patch，
    每个 patch 再映射成一个 768 维的 token。
    
    224 / 16 = 14
    14 x 14 = 196 patches
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (
            img_size // patch_size,
            img_size // patch_size
        )

        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x:
            [B, 3, 224, 224]

        output:
            [B, 196, 768]
        """

        x = self.proj(x)

        # [B, 768, 14, 14]
        x = x.flatten(2)

        # [B, 768, 196]
        x = x.transpose(1, 2)

        # [B, 196, 768]
        return x


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        drop=0.0
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            in_features,
            hidden_features
        )

        self.act = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_features,
            out_features
        )

        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """
    标准 Vision Transformer Transformer Block。

    结构：

        LayerNorm
             ↓
        Multi-Head Self-Attention
             ↓
        Residual
             ↓
        LayerNorm
             ↓
        MLP
             ↓
        Residual
    """

    def __init__(
        self,
        dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=dim,
            drop=drop
        )

    def forward(self, x):

        # -----------------------------
        # Self-Attention
        # -----------------------------

        residual = x

        x_norm = self.norm1(x)

        attn_output, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False
        )

        x = residual + attn_output

        # -----------------------------
        # MLP
        # -----------------------------

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ============================================================
# MAE Model
# ============================================================

class MAE(nn.Module):
    """
    Standard MAE-style Vision Transformer.

    Encoder:
        Image
          ↓
        Patch Embedding
          ↓
        Random Masking
          ↓
        CLS Token
          ↓
        Transformer Encoder
          ↓
        Encoder Representation

    Decoder:
        Encoder Representation
          ↓
        Insert Mask Tokens
          ↓
        Transformer Decoder
          ↓
        Predict Pixel Values
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        mask_ratio=0.75
    ):
        super().__init__()

        # ====================================================
        # Basic parameters
        # ====================================================

        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # ====================================================
        # Encoder
        # ====================================================

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embed_dim
            ),
            requires_grad=False
        )

        # Encoder Transformer blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])

        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ====================================================
        # Decoder
        # ====================================================

        self.decoder_embed = nn.Linear(
            embed_dim,
            decoder_embed_dim
        )

        # Mask token
        self.mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                decoder_embed_dim
            )
        )

        # Decoder position embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                decoder_embed_dim
            ),
            requires_grad=False
        )

        # Decoder Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(
            decoder_embed_dim
        )

        # ====================================================
        # Decoder prediction head
        #
        # 每一个 patch：
        #
        # 16 x 16 x 3 = 768
        #
        # 所以最后预测 768 个像素值
        # ====================================================

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size * in_chans
        )

        # ====================================================
        # Initialize weights
        # ====================================================

        self.initialize_weights()

    # ========================================================
    # Weight Initialization
    # ========================================================

    def initialize_weights(self):

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        self.pos_embed.data.copy_(
            self.get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.patch_embed.num_patches ** 0.5),
                cls_token=True
            )
        )

        self.decoder_pos_embed.data.copy_(
            self.get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1],
                int(self.patch_embed.num_patches ** 0.5),
                cls_token=True
            )
        )

        # ----------------------------------------------------
        # Patch embedding
        # ----------------------------------------------------

        w = self.patch_embed.proj.weight.data

        torch.nn.init.xavier_uniform_(
            w.view([w.shape[0], -1])
        )

        torch.nn.init.normal_(
            self.patch_embed.proj.bias,
            std=0.02
        )

        # ----------------------------------------------------
        # Tokens
        # ----------------------------------------------------

        torch.nn.init.normal_(
            self.cls_token,
            std=0.02
        )

        torch.nn.init.normal_(
            self.mask_token,
            std=0.02
        )

        # ----------------------------------------------------
        # Linear / LayerNorm
        # ----------------------------------------------------

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):

        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(
                m.weight
            )

            if m.bias is not None:
                nn.init.constant_(
                    m.bias,
                    0
                )

        elif isinstance(m, nn.LayerNorm):

            nn.init.constant_(
                m.bias,
                0
            )

            nn.init.constant_(
                m.weight,
                1.0
            )

    # ========================================================
    # 2D Sine-Cosine Position Embedding
    # ========================================================

    @staticmethod
    def get_2d_sincos_pos_embed(
        embed_dim,
        grid_size,
        cls_token=False
    ):
        """
        创建固定的 2D sine-cosine positional embedding。
        """

        grid_h = torch.arange(
            grid_size,
            dtype=torch.float32
        )

        grid_w = torch.arange(
            grid_size,
            dtype=torch.float32
        )

        grid = torch.meshgrid(
            grid_w,
            grid_h,
            indexing="ij"
        )

        grid = torch.stack(
            grid,
            dim=0
        )

        grid = grid.reshape(
            [2, 1, grid_size, grid_size]
        )

        pos_embed = MAE.get_2d_sincos_pos_embed_from_grid(
            embed_dim,
            grid
        )

        if cls_token:
            cls_embed = torch.zeros(
                1,
                embed_dim
            )

            pos_embed = torch.cat(
                [cls_embed, pos_embed],
                dim=0
            )

        return pos_embed

    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(
        embed_dim,
        grid
    ):

        assert embed_dim % 2 == 0

        emb_h = MAE.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2,
            grid[0]
        )

        emb_w = MAE.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2,
            grid[1]
        )

        return torch.cat(
            [emb_h, emb_w],
            dim=1
        )

    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(
        embed_dim,
        pos
    ):

        assert embed_dim % 2 == 0

        omega = torch.arange(
            embed_dim // 2,
            dtype=torch.float32
        )

        omega /= embed_dim / 2.0

        omega = 1.0 / (
            10000 ** omega
        )

        pos = pos.reshape(-1)

        out = torch.einsum(
            "m,d->md",
            pos,
            omega
        )

        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)

        emb = torch.cat(
            [emb_sin, emb_cos],
            dim=1
        )

        return emb

    # ========================================================
    # Random Masking
    # ========================================================

    def random_masking(
        self,
        x,
        mask_ratio
    ):
        """
        随机遮挡 patch。

        x:
            [B, 196, 768]

        mask_ratio = 0.75

        保留：
            196 x 25% = 49 patches

        遮挡：
            196 x 75% = 147 patches
        """

        B, N, D = x.shape

        len_keep = int(
            N * (1 - mask_ratio)
        )

        # ----------------------------------------------------
        # 每个 patch 生成一个随机数
        # ----------------------------------------------------

        noise = torch.rand(
            B,
            N,
            device=x.device
        )

        # ----------------------------------------------------
        # 根据随机数排序
        # ----------------------------------------------------

        ids_shuffle = torch.argsort(
            noise,
            dim=1
        )

        # ----------------------------------------------------
        # 保留前 len_keep 个 patch
        # ----------------------------------------------------

        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(
                1,
                1,
                D
            )
        )

        # ----------------------------------------------------
        # 创建 mask
        #
        # 0 = visible
        # 1 = masked
        # ----------------------------------------------------

        mask = torch.ones(
            [B, N],
            device=x.device
        )

        mask[:, :len_keep] = 0

        # ----------------------------------------------------
        # 恢复原来的 patch 顺序
        # ----------------------------------------------------

        ids_restore = torch.argsort(
            ids_shuffle,
            dim=1
        )

        mask = torch.gather(
            mask,
            dim=1,
            index=ids_restore
        )

        return x_masked, mask, ids_restore

    # ========================================================
    # Encoder
    # ========================================================

    def forward_encoder(
        self,
        imgs,
        mask_ratio
    ):
        """
        Encoder 只处理没有被 mask 的 patch。
        """

        # ----------------------------------------------------
        # Image -> patches
        # ----------------------------------------------------

        x = self.patch_embed(imgs)

        # [B, 196, 768]

        # ----------------------------------------------------
        # Add positional embedding
        #
        # 不包含 CLS token 的位置
        # ----------------------------------------------------

        x = x + self.pos_embed[:, 1:, :]

        # ----------------------------------------------------
        # Random masking
        # ----------------------------------------------------

        x, mask, ids_restore = self.random_masking(
            x,
            mask_ratio
        )

        # ----------------------------------------------------
        # Add CLS token
        # ----------------------------------------------------

        cls_token = (
            self.cls_token
            + self.pos_embed[:, :1, :]
        )

        cls_tokens = cls_token.expand(
            x.shape[0],
            -1,
            -1
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        for block in self.encoder_blocks:
            x = block(x)

        x = self.encoder_norm(x)

        return x, mask, ids_restore

    # ========================================================
    # Decoder
    # ========================================================

    def forward_decoder(
        self,
        x,
        ids_restore
    ):
        """
        Decoder 把：

            visible patches
            +
            mask tokens

        重新组合起来，然后预测原始图片。
        """

        # ----------------------------------------------------
        # Encoder dimension -> Decoder dimension
        # ----------------------------------------------------

        x = self.decoder_embed(x)

        # ----------------------------------------------------
        # Remove CLS token
        # ----------------------------------------------------

        x_without_cls = x[:, 1:, :]

        B, N_visible, D = x_without_cls.shape

        N_total = ids_restore.shape[1]

        # ----------------------------------------------------
        # 创建 mask tokens
        # ----------------------------------------------------

        mask_tokens = self.mask_token.repeat(
            B,
            N_total - N_visible,
            1
        )

        # ----------------------------------------------------
        # visible + mask tokens
        # ----------------------------------------------------

        x_ = torch.cat(
            [
                x_without_cls,
                mask_tokens
            ],
            dim=1
        )

        # ----------------------------------------------------
        # 恢复原始 patch 顺序
        # ----------------------------------------------------

        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(
                1,
                1,
                D
            )
        )

        # ----------------------------------------------------
        # Add CLS token again
        # ----------------------------------------------------

        x = torch.cat(
            [x[:, :1, :], x_],
            dim=1
        )

        # ----------------------------------------------------
        # Add decoder positional embedding
        # ----------------------------------------------------

        x = x + self.decoder_pos_embed

        # ----------------------------------------------------
        # Decoder Transformer
        # ----------------------------------------------------

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)

        # ----------------------------------------------------
        # Predict pixel values
        # ----------------------------------------------------

        x = self.decoder_pred(x)

        # Remove CLS token
        x = x[:, 1:, :]

        return x

    # ========================================================
    # Patchify
    # ========================================================

    def patchify(self, imgs):
        """
        将原始图片：

            [B, 3, 224, 224]

        转换成：

            [B, 196, 768]
        """

        p = self.patch_size

        assert imgs.shape[2] == imgs.shape[3]
        assert imgs.shape[2] % p == 0

        h = imgs.shape[2] // p

        x = imgs.reshape(
            imgs.shape[0],
            3,
            h,
            p,
            h,
            p
        )

        x = torch.einsum(
            "nchpwq->nhwpqc",
            x
        )

        x = x.reshape(
            imgs.shape[0],
            h * h,
            p * p * 3
        )

        return x

    # ========================================================
    # Unpatchify
    # ========================================================

    def unpatchify(self, x):
        """
        将：

            [B, 196, 768]

        恢复成：

            [B, 3, 224, 224]
        """

        p = self.patch_size

        h = int(
            x.shape[1] ** 0.5
        )

        assert h * h == x.shape[1]

        x = x.reshape(
            x.shape[0],
            h,
            h,
            p,
            p,
            3
        )

        x = torch.einsum(
            "nhwpqc->nchpwq",
            x
        )

        imgs = x.reshape(
            x.shape[0],
            3,
            h * p,
            h * p
        )

        return imgs

    # ========================================================
    # MAE Loss
    # ========================================================

    def forward_loss(
        self,
        imgs,
        pred,
        mask
    ):
        """
        MAE 的核心损失：

        只计算被遮挡 patch 的重建误差。

        不计算 visible patch。
        """

        target = self.patchify(imgs)

        # ----------------------------------------------------
        # Patch normalization
        #
        # 官方 MAE 默认会对每个 patch 做归一化。
        # ----------------------------------------------------

        mean = target.mean(
            dim=-1,
            keepdim=True
        )

        var = target.var(
            dim=-1,
            keepdim=True
        )

        target = (
            target - mean
        ) / (
            var + 1e-6
        ).sqrt()

        # ----------------------------------------------------
        # MSE
        # ----------------------------------------------------

        loss = (
            pred - target
        ) ** 2

        loss = loss.mean(
            dim=-1
        )

        # ----------------------------------------------------
        # 只计算 masked patch
        # ----------------------------------------------------

        loss = (
            loss * mask
        ).sum() / mask.sum()

        return loss

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        imgs,
        mask_ratio=None
    ):

        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        # Encoder
        latent, mask, ids_restore = self.forward_encoder(
            imgs,
            mask_ratio
        )

        # Decoder
        pred = self.forward_decoder(
            latent,
            ids_restore
        )

        # Reconstruction loss
        loss = self.forward_loss(
            imgs,
            pred,
            mask
        )

        return loss, pred, mask


# ============================================================
# Standard MAE ViT-B/16
# ============================================================

def mae_vit_base_patch16_dec512d8b():

    model = MAE(
        img_size=224,
        patch_size=16,
        in_chans=3,

        # Encoder
        embed_dim=768,
        depth=12,
        num_heads=12,

        # Decoder
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,

        mlp_ratio=4.0,

        # 75% masking
        mask_ratio=0.75
    )

    return model


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing MAE model")
    print("=" * 60)

    model = mae_vit_base_patch16_dec512d8b()

    print()
    print("Model created successfully.")
    print()

    # --------------------------------------------------------
    # Test image
    # --------------------------------------------------------

    x = torch.randn(
        2,
        3,
        224,
        224
    )

    print("Input shape:")
    print(x.shape)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        loss, pred, mask = model(x)

    print()
    print("MAE loss:")
    print(loss.item())

    print()
    print("Prediction shape:")
    print(pred.shape)

    print()
    print("Mask shape:")
    print(mask.shape)

    print()
    print("Number of patches:")
    print(model.patch_embed.num_patches)

    print()
    print("Masked patches per image:")
    print(int(mask[0].sum().item()))

    print()
    print("Visible patches per image:")
    print(int((mask[0] == 0).sum().item()))

    print()
    print("=" * 60)
    print("MAE model test finished.")
    print("=" * 60)

