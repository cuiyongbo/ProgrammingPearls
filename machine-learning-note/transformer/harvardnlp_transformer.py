#!/usr/bin/env python3
#coding=utf-8

# original code: https://github.com/harvardnlp/annotated-transformer/the_annotated_transformer.py

import os
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute Scaled Dot Product Attention
    """
    d_k = query.size(-1)
    # query: nbatches, h, -1, d_k
    # key: nbatches, h, -1, d_k
    # key.transpose(-2, -1): nbatches, h, d_k, -1
    # value: nbatches, h, -1, d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LayerNorm(nn.Module):
    def __init__(self, feature_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature_shape))
        self.b_2 = nn.Parameter(torch.zeros(feature_shape))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x-mean)/(std+self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm"""
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        """assume the output of sublayer is the same shape with x"""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder consists of self-attention and feed forward layer"""
    def __init__(self, size, self_attn, ffn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ffn = ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) 
        return self.sublayer[1](x, self.ffn)


class Encoder(nn.Module):
    """Encoder is a stack of layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        """Pass the input and mask through each layer in turn"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attention, src-attention, and feed forward layer"""
    def __init__(self, size, self_attn, src_attn, ffn, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ffn = ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.ffn)


class Decoder(nn.Module):
    """Generic N layer decoder with masking"""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


def subsequent_mask(size):
    """Mask out subsequent positions"""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # assume d_v == d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # do all the linear projections in batch from d_model ==> h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # apply attention all the projected vectors in batch
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        # concatenate using a view and apply a final linear. shape: nbatches, 1, d_model
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lookup_table = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lookup_table(x) * math.sqrt(self.d_model)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.f1 = nn.Linear(d_model, d_ff)
        self.f2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.f2(self.dropout(self.f1(x).relu()))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class EncoderDecoder(nn.Module):
    """Boilerplate for a standard encoder-decoder structure"""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """"Define standard linear + softmax generation step"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )
    # IMPORTANT! initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


class Batch:
    """Object for holding a batch of data with mask during training"""
    def __init__(self, src, tgt=None, pad=2): # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class TrainState:
    step: int = 0 # steps in the current epoch
    accum_step: int = 0 # number of gradient accumulation steps
    samples: int = 0 # total number of examples used
    tokens: int = 0 # total number of tokens processed


def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode="train", accum_iter=1, train_state=TrainState()):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i%40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(f"Epoch Step: {i:6d} | Accumulation Step: {n_accum:3d} | Loss: {loss/batch.ntokens:6.2f} | Tokens / Sec: {tokens/elapsed:7.1f} | Learning Rate: {lr:6.1e}")
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss/total_tokens, train_state


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


def load_tokenizers():
    # load tokenizer for Germany
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")
    # load tokenizer for English
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")
    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )
    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])
    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not os.path.exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("load_vocab Finished.\nVocabulary sizes:")
    print("English Vocabulary size:", len(vocab_src))
    print("Germany Vocabulary size:", len(vocab_tgt))
    return vocab_src, vocab_tgt


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )
    test_iter_map = to_map_style_dataset(test_iter)
    test_sampler = (
        DistributedSampler(test_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(test_sampler is None),
        sampler=test_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader, test_dataloader


def train_worker(gpu, ngpus_per_node, vocab_src, vocab_tgt, spacy_de, spacy_en, config, is_distributed=False):
    print(f"train_worker on GPU {gpu}", flush=True)
    torch.cuda.set_device(gpu)
    pad_idx = vocab_tgt["<blank>"]    
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6, d_model=d_model, d_ff=2048, h=8, dropout=0.1)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0  
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.0)
    criterion.cuda(gpu)
    train_dataloader, valid_dataloader, _ = create_dataloaders(gpu, vocab_src, vocab_tgt, spacy_de, spacy_en, batch_size=config["batch_size"]//ngpus_per_node, max_padding=config["max_padding"], is_distributed=is_distributed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, d_model, factor=1.0, warmup=config["warmup"]))
    train_state = TrainState()
    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)
        print(f"[GPU {gpu}] Epoch {epoch} Training ====")
        model.train()
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )
        GPUtil.showUtilization()
        if is_main_process:
            file_path = "{}{:03d}.pt".format(config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()
        print(f"[GPU {gpu}] Epoch {epoch} Validation ====")
        model.eval()
        sloss, _ = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print("Validation loss:", sloss)
        torch.cuda.empty_cache()
    if is_main_process:
        file_path = "{}final.pt".format(config["file_prefix"])
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from harvardnlp_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    model_path = "multi30k_model_final.pt"
    if os.path.exists(model_path):
        return
    if config["distributed"]:
        train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, is_distributed=False)


def load_trained_model(vocab_src_size, vocab_tgt_size):
    model = make_model(vocab_src_size, vocab_tgt_size, N=6)
    model_path = "multi30k_model_final.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def checkout_outputs(dataloader, model, vocab_src, vocab_tgt, n_examples=15, pad_idx=2, eos_string="</s>"):
    results = [()] * n_examples
    for idx in range(n_examples):
        print(f"\nExample {idx} ==================\n")
        b = next(iter(dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        #print(rb.src.shape, rb.src.dim())
        #print(rb.tgt.shape, rb.tgt.dim())
        #print(rb.src_mask.shape, rb.src_mask.dim())
        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]
        print("Source Text (Input):", " ".join(src_tokens).replace("\n", ""))
        print("Target Text (Ground Truth):", " ".join(tgt_tokens).replace("\n", ""))
        model_output = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        output_token = [vocab_tgt.get_itos()[x] for x in model_output if x != pad_idx]
        model_text = " ".join(output_token).split(eos_string, 1)[0] + eos_string
        print("Target Text (Output):", model_text.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_output, model_text)
    return results


def run_model_example(model, vocab_src, vocab_tgt, spacy_de, spacy_en, n_examples=5):
    print("Preparing Data ...")
    _, _, test_dataloader = create_dataloaders(torch.device("cpu"), vocab_src, vocab_tgt, spacy_de, spacy_en, batch_size=1, is_distributed=False)
    print("Checkin Model Output:")
    example_data = checkout_outputs(test_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples)
    return model, example_data


def predict_unknown_text(input_text, ground_truth_text, model, vocab_src, vocab_tgt, spacy_de, spacy_en):
    def tokenize(text, tokenizer):
        return [tok.text for tok in tokenizer.tokenizer(text)]
    def tokenize_de(text):
        return tokenize(text, spacy_de)
    def tokenize_en(text):
        return tokenize(text, spacy_en)

    device = torch.device("cpu")
    pad_id = vocab_src.get_stoi()["<blank>"]
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id

    #input_text = "Zwei Mädchen sind einkaufen"
    #ground_truth_text = "two girls are shopping"
    #input_text = "Die Mutter füttert ihren Sohn mit Milch"
    #ground_truth_text = "the mother is feeding her son with milk"

    src = torch.cat(
        [
            bs_id,
            torch.tensor(
                vocab_src(tokenize_de(input_text)),
                dtype=torch.int64,
                device=device,
            ),
            eos_id,
        ],
        0,
    )
    tgt = torch.cat(
        [
            bs_id,
            torch.tensor(
                vocab_tgt(tokenize_en(ground_truth_text)),
                dtype=torch.int64,
                device=device,
            ),
            eos_id,
        ],
        0,
    )
    src = src.unsqueeze(0)
    src_mask = (src != pad_id).unsqueeze(-2)
    #print(src_mask.shape)
    #print(src_mask.dim())
    tgt = tgt.unsqueeze(0)

    src_tokens = [vocab_src.get_itos()[x] for x in src[0] if x != pad_id]
    tgt_tokens = [vocab_tgt.get_itos()[x] for x in tgt[0] if x != pad_id]
    print("Source Text (Input):", " ".join(src_tokens).replace("\n", ""))
    print("Target Text (Ground Truth):", " ".join(tgt_tokens).replace("\n", ""))
    eos_string="</s>"
    model_output = greedy_decode(model, src, src_mask, 72, 0)[0]
    output_token = [vocab_tgt.get_itos()[x] for x in model_output if x != pad_id]
    model_text = " ".join(output_token).split(eos_string, 1)[0] + eos_string
    print("Target Text (Output):", model_text.replace("\n", ""))


if __name__ == "__main__":

    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = load_trained_model(len(vocab_src), len(vocab_tgt))
    run_model_example(model, vocab_src, vocab_tgt, spacy_de, spacy_en, n_examples=1)

    input_text = "Eine Mutter füttert ihren Sohn mit Milch"
    ground_truth_text = "a mother is feeding her son with milk"
    predict_unknown_text(input_text, ground_truth_text, model, vocab_src, vocab_tgt, spacy_de, spacy_en)
