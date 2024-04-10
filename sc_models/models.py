import flax.linen as nn
import jax
import jax.numpy as jnp

class TransformerBlock(nn.Module):
    n_heads: int
    dim: int
    dropout: float
    ff_dropout: float
    deterministic: bool=False
    
    @nn.compact
    def __call__(self, x, *, deterministic=None):
        if deterministic is None:
            deterministic = self.deterministic
        
        x, x_pos, mask = x
        if x_pos is None:
            x_pos = jnp.zeros_like(x)
        
        shortcut = x

        x = nn.MultiHeadDotProductAttention(
            self.n_heads, dropout_rate=self.dropout
        )(x + x_pos, x + x_pos, x, mask=mask, deterministic=deterministic)

        x = nn.LayerNorm()(x+shortcut)
        
        shortcut = x

        x = nn.Dense(self.dim)(x)
        x = jax.nn.gelu(x)
        x = nn.Dropout(self.ff_dropout)(x, deterministic=deterministic)
        x = nn.Dense(shortcut.shape[-1])(x)
        x = nn.Dropout(self.ff_dropout)(x, deterministic=deterministic)
        x = nn.LayerNorm()(x + shortcut)
        
        x = nn.LayerNorm()(x+shortcut)
        
        return x

class SCAtt(nn.Module):
    n_genes: int = 21973
    n_out: int = 68

    hidden_dim: int = 256
    n_att_layers: int = 4
    n_heads: int = 4
    cnts_binning: tuple[int] = (1,2,4,8,16,32,64,128,256)
    dropout: float = 0.2

    n_decoder_layers: int = 6

    @nn.compact
    def __call__(self, gids, cnts, *, training=False):
        embeds = nn.Embed(self.n_genes, self.hidden_dim)(gids)
        mask = gids >= 0 # padding value is -1

        cnts = jnp.digitize(cnts, jnp.asarray(self.cnts_binning))
        embeds = embeds + nn.Embed(len(self.cnts_binning)+1, self.hidden_dim)(cnts)

        query = nn.Embed(1, self.hidden_dim)(jnp.asarray([0]))

        for _ in range(self.n_att_layers):
            shortcut = query
            query = nn.MultiHeadDotProductAttention(
                self.n_heads, dropout_rate=self.dropout
            )(query, embeds, embeds, mask=mask, deterministic=not training)
            query = nn.Dropout(self.dropout)(query, deterministic=not training)
            query = nn.LayerNorm()(shortcut + query)

            shortcut = query
            query = nn.Dense(self.hidden_dim * 2)(query)
            query = jax.nn.gelu(query)
            query = nn.Dropout(self.dropout)(query, deterministic=not training)
            query = nn.Dense(self.hidden_dim)(query)
            query = nn.Dropout(self.dropout)(query, deterministic=not training)
            query = nn.LayerNorm()(shortcut + query)

        x = query[0]
        self.sow("intermediates", "state", x)
        
        for _ in range(self.n_decoder_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.Dropout(self.dropout)(x, deterministic=not training)
            x = nn.relu(x)

        out = nn.Dense(self.n_out)(x)
        
        return out

class SCLinear(nn.Module):
    n_genes: int = 21973
    n_out: int = 68

    n_layers: int = 6
    hidden_dim: int = 256
    dropout: float = 0.2
    
    @nn.compact
    def __call__(self, gids, cnts, *, training=False):
        mask = gids >= 0
        cnts = jnp.where(mask, cnts, 0)
        x = nn.Embed(self.n_genes, self.hidden_dim)(gids)
        cnts = cnts / cnts.sum()
        x = cnts @ x
        
        self.sow("intermediates", "state", x)
        
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.Dropout(self.dropout)(x, deterministic=not training)
            x = nn.relu(x)

        out = nn.Dense(self.n_out)(x)
        
        return out

class ScHybrid(nn.Module):
    n_genes: int = 21973
    n_out: int = 1

    n_layers: int = 6
    hidden_dim: int = 256
    dropout: float = 0.2
    n_heads: int = 4
    n_add_tokens: int = 256

    @nn.compact
    def __call__(self, gids, cnts, *, training=False):
        # encoding
        x = nn.Embed(self.n_genes, self.hidden_dim)(gids)
        cnts = cnts / cnts.sum()
        cell_token = cnts @ x

        # decoding
        x = nn.Embed(self.n_add_tokens, self.hidden_dim)(jnp.arange(self.n_add_tokens))
        x = jnp.r_[x, cell_token]
        
        for _ in range(self.n_layers):
            x = TransformerBlock(
                self.n_heads,
                self.hidden_dim,
                self.dropout,
                self.dropout,
            )((x, None, None), deterministic=not training)

        out = nn.Dense(self.n_out)(x[:-1])

        return out
            
class SCTransformer(nn.Module):
    n_genes: int = 21973
    n_out: int = 68

    n_heads: int = 4
    n_layers: int = 6
    hidden_dim: int = 256
    dropout: float = 0.1
    cnts_binning: tuple[int] = (1,2,4,8,16,32,64,128,256)

    n_add_tokens: int = 1
    
    @nn.compact
    def __call__(self, gids, cnts, *, training=False):
        # cnts = jnp.clip(cnts, 0, 511)
        mask = gids >= 0
        x = nn.Embed(self.n_genes, self.hidden_dim)(gids)
        cnts = jnp.digitize(cnts, jnp.asarray(self.cnts_binning))
        x = x + nn.Embed(len(self.cnts_binning)+1, self.hidden_dim)(cnts)

        # add additional token
        x = jnp.r_[x, jnp.zeros([self.n_add_tokens, self.hidden_dim], dtype=x.dtype)]
        mask = jnp.r_[mask, jnp.asarray([True] * self.n_add_tokens)]

        for _ in range(self.n_layers):
            x = TransformerBlock(
                self.n_heads,
                self.hidden_dim,
                self.dropout,
                self.dropout,
            )((x, None, mask), deterministic=not training)

        self.sow("intermediates", "state", x[-self.n_add_tokens:])

        out = nn.Dense(self.n_out)(x[-self.n_add_tokens:])
        
        return out
