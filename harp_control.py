"""harp_control.py — drive the Harp Sound Card directly from Python over its
COM serial port (no Bonsai). Implements just the Harp register commands this
experiment needs: identify, configure the "sound-playing" gate, play by index,
and stop.

Confirmed against the device on 2026-07-09: WhoAmI (reg 0) returns 1280.

Register map (harp-tech/device.soundcard):
  0   WhoAmI                 U16  read   -> 1280 for the Sound Card
  32  PlaySoundOrFrequency   U16  write  -> sound index (<32) starts playback
  33  Stop                   U8   write  -> any value stops playback
  65/66/67  ConfigureDO0/1/2 U8   write  -> 2 = "HighWhenSound" (gate HIGH while playing)

Playback (reg 32) produces AUDIBLE output on the speaker — never fire it blindly
if an animal is in the rig.
"""
import time
import serial

PORT = "COM4"
BAUD = 1000000

# Harp payload-type codes
U8, U16 = 0x01, 0x02
READ, WRITE = 0x01, 0x02

# register addresses
WHOAMI = 0
PLAY   = 32
STOP   = 33
CONFIGURE_DO = {0: 65, 1: 66, 2: 67}
DO_HIGH_WHEN_SOUND = 2


def _checksum(frame):
    return sum(frame) & 0xFF


def _read_frame(address, ptype=U16):
    f = [READ, 0x04, address, 0xFF, ptype]
    f.append(_checksum(f))
    return bytes(f)


def _write_frame(address, value, ptype=U16):
    payload = [value & 0xFF] if ptype == U8 else [value & 0xFF, (value >> 8) & 0xFF]
    length = 1 + 1 + 1 + len(payload) + 1     # addr+port+ptype+payload+checksum
    f = [WRITE, length, address, 0xFF, ptype] + payload
    f.append(_checksum(f))
    return bytes(f)


class SoundCard:
    def __init__(self, port=PORT, baud=BAUD, timeout=1.5, retries=3, retry_wait=0.6):
        last = None
        for attempt in range(retries):
            try:
                self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
                return
            except serial.SerialException as e:
                last = e
                # PermissionError(13) = the port is already held by another program.
                # A quick retry rides out a port that a just-closed window is still releasing.
                if attempt < retries - 1:
                    time.sleep(retry_wait)
        raise RuntimeError(
            f"Could not open the Harp Sound Card on {port} — it is being held by "
            "another program.\n\nMost likely causes, in order:\n"
            "  1. Another F.E.A.T.H.E.R. window is still open (only run ONE at a time) "
            "— close the others.\n"
            "  2. A previous F.E.A.T.H.E.R. window was force-closed and left the port "
            "locked — unplug the Harp Sound Card's USB cable, wait 5 seconds, and plug "
            "it back in (this does NOT disturb an Open Ephys recording — the sound card "
            "is a separate USB device).\n"
            "  3. The Harp configuration GUI is open — close it.\n\n"
            f"(underlying error: {last})"
        ) from last

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def whoami(self):
        self.ser.reset_input_buffer()
        self.ser.write(_read_frame(WHOAMI, U16))
        time.sleep(0.2)
        r = self.ser.read(64)
        # value = last two bytes before the 1-byte checksum, little-endian U16
        if len(r) >= 3:
            return int.from_bytes(r[-3:-1], "little"), r
        return None, r

    def configure_gate(self, do_index, mode=DO_HIGH_WHEN_SOUND):
        """Set digital output DO0/1/2 to 'HighWhenSound' so it gates each playback."""
        self.ser.write(_write_frame(CONFIGURE_DO[do_index], mode, U8))
        time.sleep(0.05)
        return self.ser.read(64)

    def play(self, index):
        """Start playback of a stored sound index (AUDIBLE). Returns the device reply."""
        self.ser.write(_write_frame(PLAY, index, U16))
        time.sleep(0.02)
        return self.ser.read(64)

    def stop(self):
        self.ser.write(_write_frame(STOP, 1, U8))
        time.sleep(0.02)
        return self.ser.read(64)


if __name__ == "__main__":
    # Safe self-test: identify only. No sound.
    with SoundCard() as sc:
        val, raw = sc.whoami()
        print("WhoAmI =", val, "(expect 1280)  raw:", raw.hex(" "))
        print("OK — Sound Card reachable from Python." if val == 1280
              else "Unexpected reply; check the port.")
