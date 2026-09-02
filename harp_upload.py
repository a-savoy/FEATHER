"""harp_upload.py — upload a sound to a Harp Sound Card memory index from Python.

Self-contained reimplementation of the Sound Card upload wire protocol
(documented in
  C:\\Users\\asavoy\\Documents\\HarpSoundCard\\Interface TCP\\TCP server protocol.txt).
Written with unsigned bytearrays so it runs on numpy 2.x / Python 3.13 — the
manufacturer's own protocol.py breaks on both (int8 overflow + asyncio loop=).

Requires the manufacturer TCP server running first:
    C:\\Users\\asavoy\\Documents\\HarpSoundCard\\Interface TCP\\server_v1.0.exe
It listens on localhost:9999 and bridges to the card over libusb.

Card stores STEREO INT32 at 96000 or 192000 Hz. Wiring here: left -> speaker,
right -> ADC2 audio copy, so both channels carry the stimulus.

NOT yet hardware-tested (needs the server + card). Verify on the first upload:
the card should report OK and PlaySoundOrFrequency on that index should play it.
"""
import os
import socket
import subprocess
import time
from math import gcd
import numpy as np
from scipy.signal import resample_poly

DATA_BLOCK = 32768
META_SIZE = 16
FILEMETA_SIZE = 2048
_REPLY = 12          # 5 header + 6 timestamp + 1 checksum

# Path to the manufacturer's Harp Sound Card TCP server. Override with the
# FEATHER_HARP_SERVER environment variable if yours lives elsewhere.
SERVER_EXE = os.environ.get(
    "FEATHER_HARP_SERVER",
    os.path.join(os.path.expanduser("~"), "Documents", "HarpSoundCard",
                 "Interface TCP", "server_v1.0.exe"))


def _server_up(host="localhost", port=9999):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def ensure_server(host="localhost", port=9999, exe=SERVER_EXE, wait=8):
    """Make sure the Harp TCP bridge server is running; launch it if not.
    Returns True once localhost:9999 accepts connections. The server is only
    needed for uploading (Step 4); triggering (Steps 5-6) uses COM4 directly."""
    if _server_up(host, port):
        return True
    if not os.path.exists(exe):
        raise FileNotFoundError(f"TCP server not found: {exe}")
    subprocess.Popen([exe], cwd=os.path.dirname(exe),
                     creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    for _ in range(int(wait * 4)):
        if _server_up(host, port):
            return True
        time.sleep(0.25)
    return False

# fixed command preambles (length fields verified: 0x8810=34832=16+32768+2048;
# 0x8004=32772=4+32768). See TCP server protocol.txt.
_HEADER_PREAMBLE = bytes([2, 255, 0x10, 0x88, 128, 255, 1])   # write-sound-with-data cmd
_DATA_PREAMBLE   = bytes([2, 255, 0x04, 0x80, 132, 255, 132])  # subsequent data-block cmd


def wav_to_card_wave_int(mono_float, src_sr, card_sr=96000):
    """Resample a mono float waveform in [-1, 1] to the card rate, duplicate to
    stereo (L=speaker, R=ADC2 copy), scale to int32, flatten L,R,L,R,..."""
    if src_sr != card_sr:
        g = gcd(int(card_sr), int(src_sr))
        y = resample_poly(np.asarray(mono_float, dtype=np.float64), card_sr // g, src_sr // g)
    else:
        y = np.asarray(mono_float, dtype=np.float64)
    stereo = np.stack([y, y], axis=1)
    return (np.clip(stereo, -1, 1) * (2 ** 31 - 1)).astype("<i4").flatten()


def _u32(v):
    return int(v).to_bytes(4, "little", signed=False)


def _block(data_bytes, i):
    """32768-byte block i, zero-padded at the tail."""
    b = data_bytes[i * DATA_BLOCK:(i + 1) * DATA_BLOCK]
    return b + bytes(DATA_BLOCK - len(b)) if len(b) < DATA_BLOCK else b


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def upload_sound(wave_int_flat, sound_index, sample_rate=96000, data_type=0,
                 host="localhost", port=9999, timeout=15):
    """Upload one stereo-int32 sound (flattened) to `sound_index` (0..31).
    Returns True on success; raises RuntimeError if the card rejects a packet."""
    if not ensure_server(host, port):
        raise RuntimeError("Could not start/reach the Harp TCP server on "
                           f"{host}:{port}. Check {SERVER_EXE}.")
    data_bytes = np.asarray(wave_int_flat, dtype="<i4").tobytes()
    n_int32 = len(data_bytes) // 4                          # soundLength (both channels)
    n_blocks = (len(data_bytes) + DATA_BLOCK - 1) // DATA_BLOCK

    metadata = _u32(sound_index) + _u32(n_int32) + _u32(sample_rate) + _u32(data_type)
    filemeta = bytes(FILEMETA_SIZE)                          # optional descriptors; zeros OK

    header = bytearray(_HEADER_PREAMBLE + metadata + _block(data_bytes, 0) + filemeta)
    header.append(sum(header) & 0xFF)

    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(header)
        r = _recv_exact(s, _REPLY)
        if not r or r[0] != 2:
            raise RuntimeError(f"header rejected: {r.hex(' ') or '(no reply)'}")
        for i in range(1, n_blocks):
            cmd = bytearray(_DATA_PREAMBLE + _u32(i) + _block(data_bytes, i))
            cmd.append(sum(cmd) & 0xFF)
            s.sendall(cmd)
            r = _recv_exact(s, _REPLY)
            if not r or r[0] != 2:
                raise RuntimeError(f"data block {i}/{n_blocks - 1} rejected: {r.hex(' ') or '(no reply)'}")
        s.shutdown(socket.SHUT_WR)
        final = _recv_exact(s, 2)
    return final == b"OK"
