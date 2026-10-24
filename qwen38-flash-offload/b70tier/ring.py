"""N137 B70 expert tier: shared host ring between the 3090 engine (client) and the B70 expert server.

One file in a host tmpfs directory shared by both containers (default /run/local-ai/shared/b70.ring, env
QWEN38_B70_RING), mapped MAP_SHARED. No peer-to-peer:
the 3090 side writes x rows + picks here, the B70 server H2Ds them, runs moe_forward, D2Hs the weighted rows back.

Layout (little endian, offsets in bytes):
  0       HDR   int64[512]   control words (below)
  4096    KEYS  int32[16384] resident key list for a LOAD request (key = li * E + e, store record index)
  69632   REQ   one request: int32 np, ntok, li, pad ; int32 ids[MAXP] ; float w[MAXP] ; int32 tok[MAXP]
  73728   X     fp16 [MAXP, H]   one row per pick (the pick's token row), so the server runs np rows x top-1
  598016  OUT   fp16 [MAXP, H]   weighted expert output per pick (w * expert(x)); the client adds row k into hout[tok[k]]
Protocol: client fills REQ/X, then stores HDR[REQ_SEQ] = s (release). Server spins on REQ_SEQ, computes, writes OUT,
then HDR[DONE_SEQ] = s. A LOAD: client writes KEYS + HDR[NKEYS], then HDR[LOAD_SEQ] = s; server answers HDR[LOAD_ACK] = s
with HDR[STATE] = READY (or ERROR + HDR[ERR]).
"""
import mmap, os

DEFAULT_PATH = "/run/local-ai/shared/b70.ring"
import numpy as np

MAGIC = 0x4E31333752494E47      # "N137RING"
VERSION = 1
H, E = 4096, 288
MAXP = 64
MAXKEYS = 16384
OFF_HDR, OFF_KEYS, OFF_REQ = 0, 4096, 69632
OFF_X = 73728
OFF_OUT = OFF_X + MAXP * H * 2
SIZE = (OFF_OUT + MAXP * H * 2 + 4095) // 4096 * 4096

# HDR words
(W_MAGIC, W_VER, W_H, W_E, W_STATE, W_PID, W_LOAD_SEQ, W_LOAD_ACK, W_NKEYS, W_REQ_SEQ, W_DONE_SEQ, W_ERR, W_CALLS,
 W_T_SEEN, W_T_H2D, W_T_KERN, W_T_DONE, W_BUSY_NS, W_NSLOTS, W_HEARTBEAT, W_QUIT) = range(21)
ST_NONE, ST_LOADING, ST_READY, ST_ERROR = 0, 1, 2, 3


class Ring:
    def __init__(self, path, create=False):
        if create:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
            os.ftruncate(fd, SIZE)
            os.fchmod(fd, 0o666)
        else:
            fd = os.open(path, os.O_RDWR)
        self.mm = mmap.mmap(fd, SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        b = np.frombuffer(self.mm, dtype=np.uint8)
        self.buf = b
        self.hdr = b[OFF_HDR:OFF_HDR + 4096].view(np.int64)
        self.keys = b[OFF_KEYS:OFF_KEYS + MAXKEYS * 4].view(np.int32)
        r = b[OFF_REQ:OFF_REQ + 16 + MAXP * 12].view(np.int32)
        self.req = r[:4]
        self.ids = r[4:4 + MAXP]
        self.w = r[4 + MAXP:4 + 2 * MAXP].view(np.float32)
        self.tok = r[4 + 2 * MAXP:4 + 3 * MAXP]
        self.x = b[OFF_X:OFF_X + MAXP * H * 2].view(np.float16).reshape(MAXP, H)
        self.out = b[OFF_OUT:OFF_OUT + MAXP * H * 2].view(np.float16).reshape(MAXP, H)
        import ctypes
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        if create:
            self.hdr[:] = 0
            self.hdr[W_MAGIC], self.hdr[W_VER], self.hdr[W_H], self.hdr[W_E] = MAGIC, VERSION, H, E
        assert int(self.hdr[W_MAGIC]) == MAGIC, "bad ring magic"
