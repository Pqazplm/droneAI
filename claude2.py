#!/usr/bin/env python3
"""
Object detection on Raspberry Pi Zero 2W (Picamera2 + YOLOv8 INT8 TFLite)
with overlaid bounding boxes rendered to a framebuffer that feeds the
composite (PAL) video output routed into a SpeedyBee F4 / analog VTX.

OPTIONAL add-on included in this same file (previously document.py /
marker_guidance.py):
    Experimental visual-marker guidance to Betaflight over USB MSP.

This single file contains BOTH:
  * The detector / tracker / framebuffer pipeline (original claude2.py).
  * The marker-guidance MSP module (previously a separate module).

WHY IT IS ALL IN ONE FILE
-------------------------
The guidance module is tightly coupled to the tracker output and to the
per-inference timestamping, and it is easier to keep the "only submit on
inference cycles" invariant auditable when the integration points are
visible in one place. The safety warnings below are the same as before and
MUST be read.

IMPORTANT HARDWARE / OS NOTES (read before running)
-----------------------------------------------------
1. Composite (PAL) output on Pi Zero 2W is not exposed as a full-size RCA
   jack; it's available on unpopulated TV test pads (TVDAC+ / TVDAC-) on
   the board. You need to solder wires there, or use a Pi Zero variant
   with the pads.
2. To force PAL composite output at boot, edit /boot/firmware/config.txt:
       enable_tvout=1
       sdtv_mode=2          # 2 = PAL, 0 = NTSC
       sdtv_aspect=1        # 4:3 (use 3 for 16:9)
   (On older Bullseye-based images the file is /boot/config.txt.)
3. This script does NOT try to talk to the GPU composite encoder directly.
   The simplest reliable way to get your own pixels onto the composite
   output is to draw into the Linux framebuffer device (/dev/fb0) that
   the VideoCore firmware mirrors to the TV-out DAC when the above config
   is active. It assumes fb0 is 720x576 (PAL) BGR565 or BGR888 - the
   script auto-detects bit depth from
   /sys/class/graphics/fb0/bits_per_pixel and converts accordingly.
4. Frame skipping is implemented as: run inference every Nth captured
   frame, reuse (and redraw) the last known detections on the frames in
   between, so the *displayed* video stays at full capture rate even
   though the detector itself is much slower than that on a Zero 2W.
5. Expect real-world INT8 320x320 YOLOv8 inference in the several-hundred-
   ms range on this SoC even with 4 threads + XNNPACK; thermal throttling
   after sustained load will slow it further. Heavy passive cooling / a
   small fan is strongly recommended if you need sustained framerate.
6. `ai_edge_litert` (the successor package to `tflite_runtime`) is used if
   present, falling back to `tflite_runtime`. Install one of:
       pip install ai-edge-litert
       # or
       pip install tflite-runtime
7. Guidance requires pyserial:
       sudo apt install python3-serial

GUIDANCE SAFETY WARNING (unchanged, still applies)
--------------------------------------------------
MSP connectivity alone does not establish safe RC arbitration. The
guidance module requires a separately verified Betaflight configuration
in which:

* The physical receiver remains available.
* Physical AUX1 enables/disables MSP override.
* ONLY roll, pitch, yaw and throttle are overridden.
* AUX1 and every other auxiliary channel remain receiver-controlled.
* ARM is NOT assigned to AUX1.
* Stopping MSP updates produces your tested receiver/failsafe behavior.

Do NOT configure MSP as the sole receiver for this arrangement. Reading
MSP_RC would then not provide an independent physical AUX1 switch.
The software flags below cannot verify these firmware requirements.

The program never sends an ARM command, changes firmware settings, or
deliberately changes AUX channels. MSP_SET_RAW_RC still carries auxiliary
values: the primary-only override mask is therefore essential.

Start with propellers removed and dry_run=True. Verify channel order,
directions, override release, receiver loss and USB loss. This is untested
on your particular firmware and is not a flight-ready autopilot.

A monocular box is NOT a range/altitude measurement. Box size is only a
relative approach cue for the SAME known marker. Throttle control below
is a bounded experimental image-error correction around a manually
calibrated throttle value; it does not provide altitude or position hold.
Do not use it for autonomous flight without independent stabilization,
clearance monitoring and a validated safety system.
"""

import argparse
import logging
from logging.handlers import RotatingFileHandler
import math
from pathlib import Path
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# TFLite interpreter import: prefer ai_edge_litert, fall back to tflite_runtime
# ---------------------------------------------------------------------------
_INTERPRETER_BACKEND = None
try:
    from ai_edge_litert.interpreter import Interpreter  # newer Google package
    _INTERPRETER_BACKEND = "ai_edge_litert"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        _INTERPRETER_BACKEND = "tflite_runtime"
    except ImportError:
        print(
            "ERROR: neither ai_edge_litert nor tflite_runtime is installed.\n"
            "  pip install ai-edge-litert   (preferred)\n"
            "  pip install tflite-runtime   (fallback)",
            file=sys.stderr,
        )
        sys.exit(1)

try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: picamera2 not installed (sudo apt install python3-picamera2)",
          file=sys.stderr)
    sys.exit(1)

# Guidance is optional. If pyserial is missing we degrade gracefully so the
# pure detector pipeline still runs, but print a clear warning.
try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    _SERIAL_AVAILABLE = False


# ===========================================================================
# GUIDANCE MODULE (previously marker_guidance.py / document.py)
# ===========================================================================
#
# MSP connectivity alone does not establish safe RC arbitration. See the
# safety warning at the top of this file and in the --help for --guidance.
#
# Dependencies:
#     sudo apt install python3-serial
#
# OPERATION
# ---------
# * AUX1 must first be LOW after startup.
# * Raising AUX1 selects a fresh, confirmed marker-class track.
# * The selected track ID remains locked until AUX1 goes LOW.
# * A missing/stale selected marker, serial error or excessive scheduling
#   delay latches a fault. Toggle AUX1 LOW before another attempt.
# * A fault stops RC transmissions. It does NOT send minimum throttle or
#   claim to land the aircraft. Actual takeover/failsafe is firmware-owned.
# * Marker class IDs refer to YOUR model. The supplied person/car classes
#   are not marker classes. Use a marker-trained model and update --classes.
#
# The default age limit may reject the Zero 2W's slow detections. That is
# intentional. Do not increase it merely to hide inference latency.
#
# Logs include state transitions, observation age, target ID, desired
# channels, MSP timeouts, protocol errors and exception tracebacks.

GUIDANCE_LOG = logging.getLogger("marker_guidance")

MSP_API_VERSION = 1
MSP_FC_VARIANT = 2
MSP_FC_VERSION = 3
MSP_RC = 105
MSP_SET_RAW_RC = 200


def configure_logging(path="logs/marker_guidance.log"):
    """Configure the guidance logger without replacing application handlers."""
    if GUIDANCE_LOG.handlers:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(threadName)s %(message)s"
    )
    console = logging.StreamHandler()
    disk = RotatingFileHandler(
        path, maxBytes=5_000_000, backupCount=4, encoding="utf-8"
    )
    for handler in (console, disk):
        handler.setFormatter(formatter)
        GUIDANCE_LOG.addHandler(handler)
    GUIDANCE_LOG.setLevel(logging.INFO)
    GUIDANCE_LOG.propagate = False


def _clamp(value, low, high):
    return max(low, min(high, value))


def _deadband(value, width):
    if abs(value) <= width:
        return 0.0
    return math.copysign((abs(value) - width) / (1.0 - width), value)


@dataclass(frozen=True)
class GuidanceConfig:
    port: str
    marker_class_id: int
    raw_order: str
    hover_throttle: int
    throttle_min: int
    throttle_max: int

    dry_run: bool = True
    primary_override_verified: bool = False
    baudrate: int = 115200  # Usually nominal for USB CDC.
    frequency_hz: float = 25.0
    serial_timeout: float = 0.08
    max_loop_gap: float = 0.20

    aux_low: int = 1300
    aux_high: int = 1700
    max_observation_age: float = 0.35
    acquire_timeout: float = 3.0
    minimum_confidence: float = 0.4
    minimum_track_age: int = 1

    desired_box_height: float = 0.25
    center_deadband: float = 0.02

    # Microseconds of channel correction per unit normalized image error.
    roll_gain: float = 25.0
    yaw_gain: float = 70.0
    pitch_gain: float = 55.0
    throttle_gain: float = 30.0

    roll_limit: float = 30.0
    yaw_limit: float = 80.0
    pitch_limit: float = 60.0

    roll_sign: int = 1
    pitch_sign: int = 1
    yaw_sign: int = 1
    throttle_sign: int = -1

    # Slew limiting applies to commanded channels while control is active.
    slew_us_per_second: float = 400.0
    # How many consecutive missed inference cycles a LOCKED marker may
    # coast (using its last known box) before guidance declares FAULT.
    # Initial acquisition always requires a fresh track; this grace only
    # applies to a target we have already locked onto, so a brief YOLO
    # dropout cannot cause an instant FAULT-and-stop.
    lock_grace_misses: int = 6
    # When a locked track is lost beyond lock_grace_misses, guidance
    # will look for a fresh detection of the same class within
    # reacquire_radius (normalized frame units) of the last known
    # position before declaring FAULT. 0 disables re-acquisition.
    reacquire_radius: float = 0.25

    # Score threshold for the re-acquisition candidate. Higher than
    # minimum_confidence because we want to be sure the new detection
    # really is the same marker, not a random other object.
    reacquire_min_confidence: float = 0.5

    def validate(self):
        if sorted(self.raw_order) != sorted("AETR"):
            raise ValueError("raw_order must be a permutation of AETR")
        if self.marker_class_id < 0:
            raise ValueError("marker_class_id must be nonnegative")
        if not 1000 <= self.throttle_min <= self.hover_throttle <= self.throttle_max <= 2000:
            raise ValueError("Invalid throttle bounds/calibration")
        if not 1 <= self.frequency_hz <= 50:
            raise ValueError("frequency_hz must be between 1 and 50")
        if not 0 < self.serial_timeout < self.max_loop_gap:
            raise ValueError("Invalid serial timeout / loop-gap limit")
        if self.max_loop_gap <= 1 / self.frequency_hz:
            raise ValueError("max_loop_gap must exceed the command period")
        if not 0 < self.desired_box_height < 1:
            raise ValueError("desired_box_height must be normalized to (0,1)")
        if not 0 <= self.center_deadband < 1:
            raise ValueError("Invalid center_deadband")
        if not 1000 <= self.aux_low < self.aux_high <= 2000:
            raise ValueError("Invalid AUX thresholds")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("Invalid minimum_confidence")
        if self.minimum_track_age < 1:
            raise ValueError("minimum_track_age must be positive")
        if self.reacquire_radius < 0:
            raise ValueError("reacquire_radius must be non-negative")
        if not 0 <= self.reacquire_min_confidence <= 1:
            raise ValueError("Invalid reacquire_min_confidence")
        if min(
            self.max_observation_age,
            self.acquire_timeout,
            self.slew_us_per_second,
        ) <= 0:
            raise ValueError("Age, acquisition and slew limits must be positive")
        if min(
            self.roll_gain, self.pitch_gain, self.yaw_gain, self.throttle_gain,
            self.roll_limit, self.pitch_limit, self.yaw_limit,
        ) < 0:
            raise ValueError("Gains and limits must be nonnegative")
        for direction in (
            self.roll_sign, self.pitch_sign, self.yaw_sign, self.throttle_sign
        ):
            if direction not in (-1, 1):
                raise ValueError("Channel signs must be +1 or -1")
        if not self.dry_run and not self.primary_override_verified:
            raise ValueError(
                "Live operation requires verified primary-only MSP override"
            )


@dataclass(frozen=True)
class Observation:
    track_id: int
    box: tuple
    score: float
    age: int
    misses: int = 0


class _MSP:
    """Synchronous MSPv1 transport; accessed only by the guidance thread."""

    def __init__(self, config):
        if serial is None:
            raise RuntimeError("pyserial not installed (sudo apt install python3-serial)")
        self.timeout = config.serial_timeout
        self.serial = serial.Serial(
            config.port,
            baudrate=config.baudrate,
            timeout=0.005,
            write_timeout=config.serial_timeout,
        )
        self.buffer = bytearray()
        self.serial.reset_input_buffer()

    @staticmethod
    def packet(command, payload=b""):
        if len(payload) > 255:
            raise ValueError("MSPv1 payload too large")
        body = bytes((len(payload), command)) + payload
        checksum = 0
        for value in body:
            checksum ^= value
        return b"$M<" + body + bytes((checksum,))

    def _extract(self):
        while True:
            position = self.buffer.find(b"$M")
            if position < 0:
                # Keep a possible partial header.
                self.buffer[:] = b"$" if self.buffer.endswith(b"$") else b""
                return None
            if position:
                del self.buffer[:position]
            if len(self.buffer) < 6:
                return None
            if self.buffer[2] not in (ord(">"), ord("!")):
                del self.buffer[0]
                continue
            length = self.buffer[3]
            total = length + 6
            if len(self.buffer) < total:
                return None
            frame = bytes(self.buffer[:total])
            checksum = 0
            for value in frame[3:-1]:
                checksum ^= value
            if checksum != frame[-1]:
                GUIDANCE_LOG.warning("MSP checksum mismatch; resynchronizing")
                del self.buffer[0]
                continue
            del self.buffer[:total]
            return frame[2], frame[4], frame[5:-1]

    def request(self, command, payload=b""):
        data = self.packet(command, payload)
        started = time.monotonic()
        written = self.serial.write(data)
        if written != len(data):
            raise IOError("Short MSP serial write")
        deadline = started + self.timeout
        while time.monotonic() < deadline:
            frame = self._extract()
            if frame is None:
                chunk = self.serial.read(
                    max(1, min(512, self.serial.in_waiting))
                )
                self.buffer.extend(chunk)
                continue
            direction, received_command, response = frame
            if received_command != command:
                GUIDANCE_LOG.debug("Ignoring unrelated MSP response %d", received_command)
                continue
            if direction == ord("!"):
                raise RuntimeError(f"Flight controller rejected MSP {command}")
            return response
        raise TimeoutError(f"MSP command {command} response timeout")

    def read_rc(self):
        payload = self.request(MSP_RC)
        if len(payload) < 10 or len(payload) % 2:
            raise ValueError(f"Invalid MSP_RC payload length {len(payload)}")
        channels = struct.unpack("<" + "H" * (len(payload) // 2), payload)
        if any(not 750 <= value <= 2250 for value in channels):
            raise ValueError(f"Invalid RC channel values: {channels}")
        return channels

    def write_rc(self, logical, config):
        # MSP_RC is logical roll, pitch, yaw, throttle, AUX1, ...
        # MSP_SET_RAW_RC feeds receiver channels and may be affected by
        # the configured receiver mapping. raw_order MUST be bench verified.
        values = dict(zip("AERT", logical[:4]))
        raw = [values[letter] for letter in config.raw_order]
        raw.extend(logical[4:])
        if len(raw) > 18:
            raise ValueError("More than 18 RC channels; verify firmware support")
        self.request(MSP_SET_RAW_RC, struct.pack("<" + "H" * len(raw), *raw))

    def close(self):
        self.serial.close()


class Guidance:
    """
    Marker-guidance worker. Runs in its own thread, consumes snapshots
    submitted via submit_tracks() from the detection loop, and (only when
    explicitly enabled) writes RC channels over MSP.

    Integration contract with the detection loop:
      * submit_tracks() MUST be called exactly once per INFERENCE cycle
        (never on skipped display frames).
      * captured_at MUST be a conservative timestamp taken BEFORE
        capture_array(), so capture-call duration is included in the age.
      * Only tracks with misses == 0 are considered; coasting tracks are
        deliberately ignored.
    """

    def __init__(self, config):
        config.validate()
        self.config = config
        self._lock = threading.Lock()
        self._snapshot = None
        self._stop = threading.Event()
        self._thread = None
        self._state = None
        self._target_id = None
        self._last_report = 0.0
        self._previous_output = None
        # Channels (r,p,y,t) as the pilot held them at the moment AUX1 went
        # high. Used as the initial seed and as the base for throttle math,
        # so guidance continues from the pilot's current power level instead
        # of snapping to a hardcoded hover value.
        self._base_channels = None
        # Rate-limit for the "no eligible marker" diagnostic in _select.
        self._last_elig_log = 0.0
        # Last box of the currently locked target. Used by re-acquisition
        # to bound the search for a fresh detection of the same class
        # after the original track is lost.
        self._last_locked_box = None

    def submit_tracks(self, tracks, captured_at):
        """Copy ONLY the current inference result; never retain mutable tracks."""
        captured_at = float(captured_at)
        now = time.monotonic()
        if not math.isfinite(captured_at) or captured_at > now:
            GUIDANCE_LOG.error("Rejected invalid/future observation timestamp")
            return

        observations = []
        for track in tracks:
            # Carry coasting tracks through with their `misses` count so
            # _select can decide: fresh-only for initial acquisition, but
            # allow a locked target to ride out brief detection dropouts
            # up to lock_grace_misses before declaring the lock lost.
            if track.misses > self.config.lock_grace_misses:
                continue
            if int(track.class_id) != self.config.marker_class_id:
                continue
            box = tuple(float(value) for value in track.box)
            score = float(track.score)
            if (
                len(box) != 4
                or not all(math.isfinite(value) for value in box)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                GUIDANCE_LOG.warning("Rejected malformed marker observation")
                continue

            x1, y1, x2, y2 = box
            # YOLO boxes routinely come back 1-3 px outside [0,1] when the
            # object is clipped by the frame edge. Clamping is both correct
            # (the visible part IS the marker) and necessary -- the previous
            # strict 0<=..<=1 check rejected every close-up person detection,
            # which is exactly the case this is meant to handle. Only reject
            # boxes that are entirely off-frame or degenerate after clamping.
            cx1 = min(1.0, max(0.0, x1))
            cy1 = min(1.0, max(0.0, y1))
            cx2 = min(1.0, max(0.0, x2))
            cy2 = min(1.0, max(0.0, y2))
            if not (0.0 <= cx1 < cx2 <= 1.0 and 0.0 <= cy1 < cy2 <= 1.0):
                GUIDANCE_LOG.warning(
                    "Rejected marker box %s: no area inside frame after clamping",
                    box,
                )
                continue
            observations.append(
                Observation(int(track.track_id), (cx1, cy1, cx2, cy2), score, int(track.age), int(track.misses)))
            
        
        with self._lock:
            if self._snapshot is not None and captured_at <= self._snapshot[0]:
                return
            self._snapshot = (captured_at, tuple(observations))

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Guidance instances cannot be restarted")
        self._thread = threading.Thread(
            target=self._run, name="marker-guidance", daemon=False
        )
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                GUIDANCE_LOG.critical(
                    "Guidance thread has not exited; use independent pilot override"
                )

    def _transition(self, state, reason):
        if state != self._state:
            GUIDANCE_LOG.info("state=%s -> %s reason=%s", self._state, state, reason)
            self._state = state

    def _select(self, now):
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return None, None, "no inference observation"
        captured_at, observations = snapshot
        age = now - captured_at
        if age > self.config.max_observation_age:
            return None, age, "stale observation"

        if self._target_id is None:
            # Initial acquisition: fresh tracks only. A coasting track is
            # not a confirmed marker -- never select one as a new lock.
            eligible = [
                item for item in observations
                if item.score >= self.config.minimum_confidence
                and item.age >= self.config.minimum_track_age
                and item.misses == 0
            ]
        else:
            # Already locked: allow coasting up to lock_grace_misses so a
            # brief YOLO dropout does not instantly FAULT the session. The
            # box on a coasting track is its last matched position; score
            # and age likewise reflect the last time it was confirmed.
            eligible = [
                item for item in observations
                if item.score >= self.config.minimum_confidence
                and item.age >= self.config.minimum_track_age
                and item.misses <= self.config.lock_grace_misses
            ]
        # Re-acquisition: original track is gone beyond the grace period
        # (or dropped entirely by the tracker). Before we FAULT, look for
        # a FRESH detection of the same class near the last known position.
        # This handles the case where YOLO momentarily lost the marker and
        # later re-detects it as a NEW track (new track_id) because the
        # tracker also dropped the old one.
        if not eligible and self._target_id is not None and self._last_locked_box is not None:
            cfg = self.config
            if cfg.reacquire_radius > 0:
                lx1, ly1, lx2, ly2 = self._last_locked_box
                lcx = (lx1 + lx2) / 2.0
                lcy = (ly1 + ly2) / 2.0
                best = None
                best_dist = cfg.reacquire_radius ** 2
                for item in observations:
                    if item.misses != 0:
                        continue
                    if item.score < cfg.reacquire_min_confidence:
                        continue
                    bx1, by1, bx2, by2 = item.box
                    bcx = (bx1 + bx2) / 2.0
                    bcy = (by1 + by2) / 2.0
                    d2 = (bcx - lcx) ** 2 + (bcy - lcy) ** 2
                    if d2 < best_dist:
                        best_dist = d2
                        best = item
                if best is not None:
                    GUIDANCE_LOG.info(
                        "Re-acquired marker as track_id=%d score=%.3f "
                        "(was track_id=%d, dist=%.3f)",
                        best.track_id, best.score, self._target_id,
                        best_dist ** 0.5,
                    )
                    self._target_id = best.track_id
                    self._last_locked_box = best.box
                    self._previous_output = None  # re-seed from receiver
                    return best, age, ""
        if self._target_id is not None:
            for item in eligible:
                if item.track_id == self._target_id:
                    return item, age, ""
            return None, age, "locked marker missing or insufficient confidence"

        if not eligible:
            # Rate-limit the diagnostic to ~2 Hz so it doesn't flood the log.
            now_mono = time.monotonic()
            if now_mono - self._last_elig_log >= 0.5:
                scores = [round(o.score, 3) for o in observations]
                ages = [o.age for o in observations]
                GUIDANCE_LOG.info(
                    "no eligible marker: n_obs=%d scores=%s ages=%s "
                    "min_conf=%.2f min_age=%d",
                    len(observations), scores, ages,
                    self.config.minimum_confidence,
                    self.config.minimum_track_age,
                )
                self._last_elig_log = now_mono
            return None, age, "no confirmed marker"

        # Initial acquisition only: prefer the marker nearest the image center.
        target = min(
            eligible,
            key=lambda item: (
                ((item.box[0] + item.box[2]) / 2 - 0.5) ** 2
                + ((item.box[1] + item.box[3]) / 2 - 0.5) ** 2,
                -item.score,
            ),
        )
        self._target_id = target.track_id
        self._last_locked_box = target.box
        GUIDANCE_LOG.info(
            "Locked marker track_id=%d score=%.3f", target.track_id, target.score
        )
        return target, age, ""

    def _command(self, target, receiver, dt):
        cfg = self.config
        x1, y1, x2, y2 = target.box
        error_x = _deadband(x1 + x2 - 1.0, cfg.center_deadband)
        error_y = _deadband(y1 + y2 - 1.0, cfg.center_deadband)

        # Relative image-scale cue, NOT meters.
        approach_error = _clamp(
            (cfg.desired_box_height - (y2 - y1)) / cfg.desired_box_height,
            0.0,
            1.0,
        )

        # Reduce forward input while the marker is off-center.
        alignment = _clamp(1.0 - 2.0 * abs(error_x), 0.0, 1.0)
        alignment *= _clamp(1.0 - 2.0 * abs(error_y), 0.0, 1.0)

        roll = 1500 + cfg.roll_sign * _clamp(
            cfg.roll_gain * error_x, -cfg.roll_limit, cfg.roll_limit
        )
        pitch = 1500 + cfg.pitch_sign * min(
            cfg.pitch_gain * approach_error * alignment, cfg.pitch_limit
        )
        yaw = 1500 + cfg.yaw_sign * _clamp(
            cfg.yaw_gain * error_x, -cfg.yaw_limit, cfg.yaw_limit
        )

        # Base throttle: prefer the value captured at AUX1 rising edge.
        # Falling back to hover_throttle only happens if capture is
        # disabled or somehow missing (should not normally occur once
        # aux_enabled is True). Envelope span is the SAME +/- span the
        # config already defined around hover, so a pilot who took over
        # on hover gets identical behavior to before; a pilot who took
        # over at a different throttle gets that same span around THEIR
        # value, not around hover.
        base_throttle = (self._base_channels[3]
                         if self._base_channels is not None
                         else float(cfg.hover_throttle))

        span_dn = cfg.hover_throttle - cfg.throttle_min
        span_up = cfg.throttle_max - cfg.hover_throttle

        throttle = _clamp(
            base_throttle + cfg.throttle_sign * cfg.throttle_gain * error_y,
            max(1000.0, base_throttle - span_dn),
            min(2000.0, base_throttle + span_up),
        )

        # Remember the last locked box for re-acquisition.
        self._last_locked_box = (x1, y1, x2, y2)

        desired = [roll, pitch, yaw, throttle]
        if self._previous_output is None:
            # Seed from the captured channels so the very first
            # MSP_SET_RAW_RC matches what the pilot was already doing.
            if self._base_channels is not None:
                self._previous_output = list(self._base_channels)
            else:
                self._previous_output = [1500.0, 1500.0, 1500.0, float(cfg.hover_throttle)]

        step = cfg.slew_us_per_second * dt
        output = [
            previous + _clamp(goal - previous, -step, step)
            for previous, goal in zip(self._previous_output, desired)
        ]
        self._previous_output = output

        # AUX values are carried through, never generated as mode/arm commands.
        channels = [int(round(_clamp(value, 1000, 2000))) for value in output]
        channels.extend(receiver[4:])
        return channels, error_x, error_y, approach_error

    def _session(self, link):
        cfg = self.config
        variant = link.request(MSP_FC_VARIANT).decode("ascii", errors="replace")
        version = tuple(link.request(MSP_FC_VERSION))
        api = tuple(link.request(MSP_API_VERSION))
        GUIDANCE_LOG.info(
            "Connected variant=%r firmware=%s api=%s", variant, version, api
        )
        if variant != "BTFL":
            raise RuntimeError("Expected Betaflight FC variant BTFL")

        # Every connection requires an observed LOW before allowing HIGH.
        ready = False
        fault = False
        aux_enabled = False
        acquire_started = None
        self._target_id = None
        self._previous_output = None
        self._base_channels = None
        self._last_locked_box = None
        self._transition("WAIT_AUX_LOW", "startup/reconnection interlock")

        previous_tick = time.monotonic()
        next_tick = previous_tick
        period = 1.0 / cfg.frequency_hz

        while not self._stop.is_set():
            if self._stop.wait(max(0.0, next_tick - time.monotonic())):
                return
            tick = time.monotonic()
            dt = tick - previous_tick
            previous_tick = tick

            receiver = link.read_rc()
            now = time.monotonic()

            # Covers scheduling delay AND delay obtaining receiver telemetry.
            delayed = dt > cfg.max_loop_gap or now - tick > cfg.max_loop_gap
            aux1 = receiver[4]

            if aux1 <= cfg.aux_low:
                ready = True
                fault = False
                aux_enabled = False
                acquire_started = None
                self._target_id = None
                self._previous_output = None
                self._base_channels = None
                self._last_locked_box = None
                self._transition("STANDBY", "physical AUX1 low; no RC transmission")
            else:
                if aux1 >= cfg.aux_high:
                    if not aux_enabled:
                        # Rising edge: freeze the pilot's current channels.
                        # throttle is the important one (prevents the "drops
                        # to hover" jump), but capturing r/p/y as the initial
                        # seed also avoids a snap if the pilot was mid-stick.
                        #
                        # Guard: when Betaflight enters MSP override, MSP_RC
                        # briefly reports the MCU's default/mins (we saw 885)
                        # until the first MSP_SET_RAW_RC is written. Capturing
                        # those values would seed guidance with garbage and
                        # produce a multi-second slew toward neutral. 1000 is
                        # the physical minimum on a real stick; anything below
                        # it is not a stick position and must not be trusted.
                        captured = tuple(float(v) for v in receiver[:4])
                        if min(captured) < 1000.0:
                            GUIDANCE_LOG.warning(
                                "AUX1 high but channels invalid "
                                "(roll=%.0f pitch=%.0f yaw=%.0f throttle=%.0f) -- "
                                "waiting for >=1000 us on all channels "
                                "before capturing",
                                captured[0], captured[1], captured[2], captured[3],
                            )
                            # Do NOT set aux_enabled; retry capture next tick.
                        else:
                            self._base_channels = captured
                            aux_enabled = True
                            GUIDANCE_LOG.info(
                                "AUX1 rising: captured base channels "
                                "roll=%.0f pitch=%.0f yaw=%.0f throttle=%.0f "
                                "(envelope +%d/-%d us around captured throttle)",
                                self._base_channels[0],
                                self._base_channels[1],
                                self._base_channels[2],
                                self._base_channels[3],
                                cfg.throttle_max - cfg.hover_throttle,
                                cfg.hover_throttle - cfg.throttle_min,
                            )

                if not ready:
                    self._transition("WAIT_AUX_LOW", "must observe AUX1 low first")
                elif delayed:
                    fault = True
                    self._transition("FAULT", "control/telemetry deadline exceeded")
                elif fault:
                    self._transition("FAULT", "toggle AUX1 low to reset")
                elif aux_enabled:
                    if acquire_started is None:
                        acquire_started = now
                    target, observation_age, reason = self._select(now)

                    if target is None:
                        if (
                            self._target_id is not None
                            or now - acquire_started > cfg.acquire_timeout
                        ):
                            fault = True
                            self._transition("FAULT", reason)
                        else:
                            self._transition("ACQUIRING", reason)
                    else:
                        channels, ex, ey, approach = self._command(
                            target, receiver, _clamp(dt, 0.0, cfg.max_loop_gap)
                        )

                        # Recheck observation age immediately before transmission.
                        with self._lock:
                            latest = self._snapshot
                        if (
                            latest is None
                            or observation_age + time.monotonic() - now
                            > cfg.max_observation_age
                        ):
                            fault = True
                            self._transition("FAULT", "observation expired before send")
                        elif not self._stop.is_set():
                            self._transition(
                                "DRY_RUN" if cfg.dry_run else "ACTIVE",
                                "fresh locked marker and AUX1 enabled",
                            )
                            if not cfg.dry_run:
                                link.write_rc(channels, cfg)

                            if now - self._last_report >= 0.5:
                                coast_suffix = (
                                    f" coast={target.misses}"
                                    if target.misses else ""
                                )
                                GUIDANCE_LOG.info(
                                    "state=%s target=%d age_ms=%.1f AUX1=%d "
                                    "ex=%.3f ey=%.3f approach=%.3f "
                                    "roll=%d pitch=%d yaw=%d throttle=%d%s",
                                    self._state, target.track_id,
                                    observation_age * 1000, aux1,
                                    ex, ey, approach, *channels[:4],
                                    coast_suffix,
                                )
                                self._last_report = now

            # Never send catch-up bursts.
            next_tick += period
            if next_tick < time.monotonic():
                next_tick = time.monotonic()

    def _run(self):
        GUIDANCE_LOG.warning(
            "Starting dry_run=%s; target class=%d; no automatic arming. "
            "Firmware must own takeover/failsafe.",
            self.config.dry_run, self.config.marker_class_id,
        )
        while not self._stop.is_set():
            link = None
            try:
                link = _MSP(self.config)
                self._session(link)
            except Exception:
                self._transition("LINK_FAULT", "RC transmission stopped")
                GUIDANCE_LOG.exception(
                    "Guidance/MSP failure; reconnect requires AUX1-low interlock"
                )
            finally:
                if link is not None:
                    try:
                        link.close()
                    except Exception:
                        GUIDANCE_LOG.exception("Error closing MSP connection")
            self._stop.wait(1.0)

        self._transition("STOPPED", "no further RC transmissions")
        GUIDANCE_LOG.warning(
            "Stopped. No neutral/throttle/disarm packet was sent; "
            "receiver takeover/failsafe remains the flight controller's responsibility."
        )


# ===========================================================================
# DETECTOR PIPELINE (original claude2.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8-INT8 TFLite detector -> PAL framebuffer overlay "
                    "+ optional marker guidance over MSP"
    )
    p.add_argument("--model", default="yolov8n_int8.tflite", help="Path to INT8 TFLite model")
    p.add_argument("--labels", default="labels.txt", help="Path to newline-separated class names")
    p.add_argument("--input-size", type=int, default=320, help="Model input size (square)")
    p.add_argument("--infer-every", type=int, default=3,
                   help="Run inference on every Nth captured frame (default 3 = skip 2)")
    p.add_argument("--conf-thresh", type=float, default=0.45,
                   help="Confidence threshold. NOTE: INT8-quantized nano models often top out "
                        "well below 0.6 even on clean detections -- if boxes stop appearing "
                        "entirely, check the [debug] max-confidence log line before raising this")
    p.add_argument("--iou-thresh", type=float, default=0.60,
                   help="NMS IoU threshold (raised default: fewer duplicate/flickering overlapping boxes)")
    p.add_argument("--classes", default="0,2",
                   help="Comma-separated COCO class ids to keep, all others discarded "
                        "(default: 0=person,2=car)")
    p.add_argument("--track-iou-thresh", type=float, default=0.30,
                   help="Min IoU to associate a new detection with an existing track between inference cycles")
    p.add_argument("--smooth-alpha", type=float, default=0.5,
                   help="EMA smoothing factor for track box coords: 1.0=snap instantly to new box "
                        "(no smoothing), lower=heavier smoothing/more lag")
    p.add_argument("--track-life", type=int, default=5,
                   help="How many consecutive INFERENCE CYCLES (not display frames) a track is kept "
                        "alive and drawn after it stops being re-detected, before it's dropped")
    p.add_argument("--num-threads", type=int, default=4, help="TFLite interpreter threads")
    p.add_argument("--cam-width", type=int, default=720, help="Camera capture width")
    p.add_argument("--cam-height", type=int, default=576, help="Camera capture height")
    p.add_argument("--fb-device", default="/dev/fb0", help="Framebuffer device for composite output")
    p.add_argument("--no-fb", action="store_true",
                   help="Disable framebuffer writes (debug on desktop with a normal window instead)")
    p.add_argument("--log-every", type=int, default=30, help="Log FPS every N frames")

    # ---- Guidance-related options (all default to OFF / dry-run) ---------
    g = p.add_argument_group(
        "guidance",
        "Optional marker guidance over MSP. Disabled unless --guidance is set. "
        "READ THE SAFETY WARNING AT THE TOP OF THIS FILE BEFORE USING ANY OF THESE."
    )
    g.add_argument("--guidance-min-confidence", type=float, default=0.35,
                   help="Minimum track score to consider as marker. INT8 nano models "
                        "often top out at 0.5-0.6; set below that.")
    g.add_argument("--guidance-min-track-age", type=int, default=1,
                   help="Inference cycles a track must survive before locking. "
                        "Raise for noise rejection; keep low for fast testing.")
    g.add_argument("--guidance-acquire-timeout", type=float, default=5.0,
                   help="Seconds AUX1 may stay high without a lock before FAULT.")
    g.add_argument("--guidance-max-observation-age", type=float, default=0.5,
                   help="Reject observations older than this. Do NOT raise to hide latency.")
    g.add_argument("--guidance-slew", type=float, default=120.0,
                   help="Max us/sec slew on commanded channels. Raise to test responsiveness.")
    g.add_argument("--guidance", action="store_true",
                   help="Enable the marker-guidance worker (still defaults to dry_run)")
    g.add_argument("--guidance-port", default=None,
                   help="Serial port to flight controller, e.g. "
                        "/dev/serial/by-id/usb-...-if00")
    g.add_argument("--guidance-marker-class", type=int, default=-1,
                   help="Class id of the marker in YOUR model. MUST be a class you "
                        "actually train and pass via --classes as well.")
    g.add_argument("--guidance-log", default="logs/marker_guidance.log",
                   help="Path to guidance log file")
    g.add_argument("--guidance-dry-run", action="store_true", default=True,
                   help="Guidance dry-run (default). Live transmission requires BOTH "
                        "--guidance-live AND --guidance-i-verified-primary-override.")
    g.add_argument("--guidance-live", dest="guidance_dry_run", action="store_false",
                   help="Disable dry-run. Requires --guidance-i-verified-primary-override.")
    g.add_argument("--guidance-i-verified-primary-override", action="store_true",
                   help="Explicit acknowledgement that the Betaflight receiver setup has "
                        "been bench-verified to override ONLY roll/pitch/yaw/throttle via "
                        "MSP, with AUX1 still on the physical receiver. This is a software "
                        "flag only and is NOT a substitute for firmware configuration.")
    g.add_argument("--guidance-raw-order", default="AETR",
                   help="Raw MSP_SET_RAW_RC channel order as observed in Betaflight "
                        "Receiver tab (permutation of AETR). MUST be bench-verified.")
    g.add_argument("--guidance-hover-throttle", type=int, default=1350,
                   help="Manually calibrated hover throttle (us). NOT flight calibration.")
    g.add_argument("--guidance-throttle-min", type=int, default=1250)
    g.add_argument("--guidance-throttle-max", type=int, default=1450)
    g.add_argument("--guidance-desired-box-height", type=float, default=0.25,
                   help="Target normalized box height for the marker (relative cue only)")
    g.add_argument("--guidance-roll-sign", type=int, choices=(-1, 1), default=1)
    g.add_argument("--guidance-pitch-sign", type=int, choices=(-1, 1), default=1)
    g.add_argument("--guidance-yaw-sign", type=int, choices=(-1, 1), default=1)
    g.add_argument("--guidance-throttle-sign", type=int, choices=(-1, 1), default=-1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Post-processing: vectorized NMS (numpy only, no per-box Python loops beyond
# the small kept-index loop NMS itself requires)
# ---------------------------------------------------------------------------
def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Standard greedy NMS, vectorized IoU computation. Returns kept indices."""
    if boxes_xyxy.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

        order = rest[iou <= iou_thresh]

    return np.array(keep, dtype=np.int64)


def debug_confidence_stats(raw_output: np.ndarray, num_classes: int, allowed_classes: np.ndarray):
    """
    Computes the raw max confidence per allowed class BEFORE any thresholding,
    so you can see what the model is actually producing regardless of
    --conf-thresh. Use this to tell apart three different failure modes that
    all look like "nothing is highlighted":
      1. conf_thresh set higher than the model's real output range -> here
         you'll see nonzero-but-low numbers (e.g. 0.2-0.4) for the classes
         you care about.
      2. class filter/index mismatch (custom-trained model, different class
         order) -> here the allowed classes will show ~0 while OTHER classes
         (check overall_max_class) show high confidence.
      3. genuinely no target in frame / camera or model input problem ->
         overall_max stays near-zero for every class, all the time.
    """
    out = np.squeeze(raw_output)
    if out.shape[0] == 4 + num_classes:
        out = out.T
    class_scores = out[:, 4:4 + num_classes]
    per_class_max = class_scores.max(axis=0)
    overall_best_class = int(np.argmax(per_class_max))
    allowed_str = ", ".join(f"class{c}={per_class_max[c]:.3f}" for c in allowed_classes.tolist()
                             if 0 <= c < num_classes)
    return (f"{allowed_str} | overall_best=class{overall_best_class}"
            f"({per_class_max[overall_best_class]:.3f})")


def postprocess(raw_output: np.ndarray, conf_thresh: float, iou_thresh: float,
                 num_classes: int, allowed_classes: np.ndarray = None):
    """
    Expects YOLOv8-style output of shape (1, 4+num_classes, num_anchors) or
    (1, num_anchors, 4+num_classes); handles both layouts.
    Returns (boxes_xyxy_norm[N,4], scores[N], class_ids[N]) in normalized [0,1] coords.

    allowed_classes: if given, detections whose class_id is not in this array
    are dropped BEFORE NMS. Filtering early (not after) means NMS only has to
    consider person/car candidates, which is both cheaper and means a car box
    can no longer suppress/interact with a person box of a class we don't care
    about, or vice versa -- previously e.g. a high-confidence "truck" detection
    could still hold IoU-based sway over post-NMS results.
    """
    out = np.squeeze(raw_output)
    if out.shape[0] == 4 + num_classes:
        out = out.T  # -> (num_anchors, 4+num_classes)

    boxes_xywh = out[:, :4]
    class_scores = out[:, 4:4 + num_classes]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= conf_thresh
    if allowed_classes is not None:
        mask &= np.isin(class_ids, allowed_classes)

    if not np.any(mask):
        empty = np.empty((0,), dtype=np.float32)
        return np.empty((0, 4), dtype=np.float32), empty, empty.astype(np.int32)

    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms(boxes_xyxy, scores, iou_thresh)

    return boxes_xyxy[keep], scores[keep], class_ids[keep]


# ---------------------------------------------------------------------------
# Lightweight IoU tracker: fixes box "flicker" without adding real load
# ---------------------------------------------------------------------------
#
# WHY FLICKER HAPPENS HERE SPECIFICALLY
# --------------------------------------
# With ~600-750ms inference and inference only every 3rd captured frame, a
# detection is a rare, noisy event: a person standing still can drop below
# conf_thresh on one inference cycle and reappear on the next just from YOLO's
# own frame-to-frame score jitter -- nothing in the scene actually changed.
# Drawing "only what the current inference cycle says" makes that jitter
# directly visible as boxes popping in and out, which is what you saw.
#
# WHAT FIXES IT, AND WHY THIS COMBINATION
# -----------------------------------------
# 1. Higher conf_thresh (0.6-0.7) + higher NMS IoU (0.6) -- cheap, first line
#    of defense. Raises the bar so borderline/duplicate boxes never appear at
#    all. Doesn't fix disappearance of a *real* target that dips below the
#    threshold for one cycle, so it's necessary but not sufficient alone.
# 2. Track persistence ("life extension") -- a track is only dropped after
#    `track_life` *consecutive missed inference cycles*, not missed display
#    frames. This directly targets the "detector randomly skips one cycle"
#    case: the box stays on screen at its last known place instead of
#    vanishing for exactly one inference interval.
# 3. EMA smoothing of coordinates -- when a track *does* get a fresh match,
#    its displayed box moves toward the new box gradually (`smooth_alpha`)
#    rather than snapping, so the visible motion is a smooth glide instead of
#    a jump -- this is what removes the "shaking"/jitter distinct from the
#    on/off flicker.
# 4. Minimal IoU-based greedy matching between inference cycles is what makes
#    (2) and (3) possible: it's what lets us say "this new detection is the
#    same object as that track" so we know what to smooth toward and what to
#    keep alive. With at most a handful of person/car boxes in frame, the
#    IoU matrix here is a few floats -- negligible next to the ~600ms spent
#    inside the interpreter, so it doesn't cost you FPS.
#
# NOT implemented (and why): adaptive confidence threshold. It adds a feedback
# loop (threshold depends on last frame's average confidence) that's harder to
# reason about and tune than a fixed threshold, and the persistence+smoothing
# combo above already absorbs the single-cycle confidence dips that an
# adaptive threshold would otherwise be compensating for -- so it wasn't worth
# the extra complexity/unpredictability for this use case.
@dataclass
class Track:
    track_id: int
    box: np.ndarray        # smoothed xyxy, normalized [0,1], shape (4,)
    score: float
    class_id: int
    misses: int = 0         # consecutive INFERENCE CYCLES with no matching detection
    age: int = 0            # inference cycles this track has been alive/matched


class SimpleIoUTracker:
    """
    Greedy IoU-matching tracker across inference cycles (NOT display frames).
    Call update() once per inference cycle with the fresh (filtered, NMS'd)
    detections; call get_display_tracks() every display frame to get what to
    draw (including tracks currently "coasting" on persistence).
    """

    def __init__(self, iou_match_thresh: float, smooth_alpha: float, max_misses: int):
        self.iou_match_thresh = iou_match_thresh
        self.smooth_alpha = smooth_alpha
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self._next_id = 1

    @staticmethod
    def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Vectorized pairwise IoU. boxes_a: [N,4], boxes_b: [M,4], both xyxy."""
        if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
            return np.empty((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
        ax1, ay1 = boxes_a[:, 0:1], boxes_a[:, 1:2]
        ax2, ay2 = boxes_a[:, 2:3], boxes_a[:, 3:4]
        bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

        xx1 = np.maximum(ax1, bx1)
        yy1 = np.maximum(ay1, by1)
        xx2 = np.minimum(ax2, bx2)
        yy2 = np.minimum(ay2, by2)

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
        area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
        return inter / (area_a + area_b - inter + 1e-9)

    def update(self, det_boxes: np.ndarray, det_scores: np.ndarray, det_class_ids: np.ndarray):
        track_boxes = (np.stack([t.box for t in self.tracks]) if self.tracks
                       else np.empty((0, 4), dtype=np.float32))
        iou = self._iou_matrix(track_boxes, det_boxes)

        # Greedy match: highest-IoU pairs first, same class only, each track
        # and each detection used at most once. Cheap: a handful of tracks x
        # a handful of detections, not thousands of anchors.
        candidates = []
        for ti in range(iou.shape[0]):
            for di in range(iou.shape[1]):
                if (self.tracks[ti].class_id == det_class_ids[di]
                        and iou[ti, di] >= self.iou_match_thresh):
                    candidates.append((iou[ti, di], ti, di))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            t = self.tracks[ti]
            a = self.smooth_alpha
            t.box = a * det_boxes[di] + (1.0 - a) * t.box  # EMA toward new box
            t.score = float(det_scores[di])
            t.misses = 0
            t.age += 1

        # Unmatched existing tracks: age them; keep coasting until max_misses
        surviving = []
        for ti, t in enumerate(self.tracks):
            if ti in matched_tracks:
                surviving.append(t)
                continue
            t.misses += 1
            if t.misses <= self.max_misses:
                surviving.append(t)  # keep drawing at last known (smoothed) box
        self.tracks = surviving

        # Unmatched detections: brand-new tracks
        for di in range(det_boxes.shape[0]):
            if di in matched_dets:
                continue
            self.tracks.append(Track(
                track_id=self._next_id,
                box=det_boxes[di].copy(),
                score=float(det_scores[di]),
                class_id=int(det_class_ids[di]),
                misses=0,
                age=1,
            ))
            self._next_id += 1

    def get_display_tracks(self):
        return self.tracks


# ---------------------------------------------------------------------------
# Framebuffer writer for composite (PAL) output
# ---------------------------------------------------------------------------
class FramebufferWriter:
    """
    Writes BGR frames into /dev/fb0, converting to the panel's native depth.

    Critically, the fb's actual (width, height, bits-per-pixel, row stride)
    are read from /sys/class/graphics/fb0/*, NOT assumed to equal the camera
    capture resolution. If you write more bytes per line than the real row
    stride, or more lines than the real height, the write() syscall fails
    with ENOSPC ("No space left on device") -- that's what a hardcoded
    720x576 caused when the real panel geometry was different.
    """

    def __init__(self, device: str, requested_width: int, requested_height: int):
        self.device = device
        self.bpp = self._read_int("bits_per_pixel", default=16)
        self.width, self.height = self._read_virtual_size(requested_width, requested_height)
        self.bytes_per_pixel = max(1, self.bpp // 8)
        self.stride = self._read_stride(default=self.width * self.bytes_per_pixel)

        if (self.width, self.height) != (requested_width, requested_height):
            print(f"[fb] note: panel is {self.width}x{self.height}@{self.bpp}bpp "
                  f"(stride={self.stride}B), resizing frames from "
                  f"{requested_width}x{requested_height} to fit", file=sys.stderr)

        self.fb = open(device, "r+b")
        self._rgb_scratch = None     # converted pixel buffer, no stride padding
        self._line_buffer = None     # full stride-padded buffer actually written

    @staticmethod
    def _read_int(name: str, default: int) -> int:
        try:
            with open(f"/sys/class/graphics/fb0/{name}") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return default

    def _read_virtual_size(self, fallback_w: int, fallback_h: int):
        try:
            with open("/sys/class/graphics/fb0/virtual_size") as f:
                w_str, h_str = f.read().strip().split(",")
                return int(w_str), int(h_str)
        except (OSError, ValueError):
            return fallback_w, fallback_h

    def _read_stride(self, default: int) -> int:
        # Not all kernels expose this file; fall back to width*bytes_per_pixel.
        return self._read_int("stride", default=default)

    def write(self, frame_bgr: np.ndarray):
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height),
                                    interpolation=cv2.INTER_LINEAR)

        if self.bpp == 16:
            if self._rgb_scratch is None:
                self._rgb_scratch = np.empty((self.height, self.width), dtype=np.uint16)
            b = (frame_bgr[:, :, 0] >> 3).astype(np.uint16)
            g = (frame_bgr[:, :, 1] >> 2).astype(np.uint16)
            r = (frame_bgr[:, :, 2] >> 3).astype(np.uint16)
            self._rgb_scratch[:] = (r << 11) | (g << 5) | b
            row_bytes = self._rgb_scratch.view(np.uint8).reshape(self.height, -1)
        elif self.bpp == 32:
            if self._rgb_scratch is None:
                self._rgb_scratch = np.empty((self.height, self.width, 4), dtype=np.uint8)
            self._rgb_scratch[:, :, :3] = frame_bgr
            self._rgb_scratch[:, :, 3] = 0
            row_bytes = self._rgb_scratch.reshape(self.height, -1)
        else:  # 24 bpp
            row_bytes = np.ascontiguousarray(frame_bgr).reshape(self.height, -1)

        row_len = row_bytes.shape[1]

        if self.stride == row_len:
            payload = row_bytes.tobytes()
        else:
            # Real panel has row padding (stride > pixel bytes per row) --
            # write into a stride-sized buffer so lines land at the right
            # offsets instead of drifting/wrapping and running off the end.
            if self._line_buffer is None or self._line_buffer.shape != (self.height, self.stride):
                self._line_buffer = np.zeros((self.height, self.stride), dtype=np.uint8)
            self._line_buffer[:, :row_len] = row_bytes
            payload = self._line_buffer.tobytes()

        try:
            self.fb.seek(0)
            self.fb.write(payload)
            self.fb.flush()
        except OSError as e:
            print(f"[fb] write error: {e}", file=sys.stderr)

    def close(self):
        try:
            self.fb.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_detections(frame_bgr: np.ndarray, tracks, labels, frame_w, frame_h):
    """Draws every currently-alive track (fresh or coasting on persistence)."""
    for t in tracks:
        x1, y1, x2, y2 = t.box
        x1 = int(max(0, x1) * frame_w)
        y1 = int(max(0, y1) * frame_h)
        x2 = int(min(1.0, x2) * frame_w)
        y2 = int(min(1.0, y2) * frame_h)

        label = labels[t.class_id] if 0 <= t.class_id < len(labels) else str(t.class_id)
        # misses>0 means this box is coasting (not re-confirmed this cycle) --
        # a dimmer color makes that visible to the pilot instead of hiding it.
        color = (0, 255, 0) if t.misses == 0 else (0, 200, 255)
        text = f"#{t.track_id} {label} {t.score:.2f}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame_bgr, text, (x1 + 2, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame_bgr


def load_labels(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        print(f"[warn] could not read labels file '{path}', using numeric class ids", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False


def _build_guidance(args, allowed_classes: np.ndarray):
    """
    Validate guidance arguments and construct a Guidance instance, or return
    None if guidance is disabled. Kept separate so the constructor/validation
    errors are surfaced before the camera or framebuffer are touched.
    """
    if not args.guidance:
        return None

    if not _SERIAL_AVAILABLE:
        print("ERROR: --guidance requires pyserial (sudo apt install python3-serial)",
              file=sys.stderr)
        sys.exit(1)
    if not args.guidance_port:
        print("ERROR: --guidance requires --guidance-port", file=sys.stderr)
        sys.exit(1)
    if args.guidance_marker_class < 0:
        print("ERROR: --guidance requires --guidance-marker-class "
              "(the class id of YOUR marker in YOUR model)", file=sys.stderr)
        sys.exit(1)

    # Fail fast if the marker class is not even being kept by the detector --
    # otherwise guidance would sit in ACQUIRING forever with no explanation.
    if args.guidance_marker_class not in allowed_classes.tolist():
        print(f"ERROR: --guidance-marker-class {args.guidance_marker_class} is not in "
              f"--classes {allowed_classes.tolist()}; the detector will never emit it",
              file=sys.stderr)
        sys.exit(1)

    if not args.guidance_dry_run and not args.guidance_i_verified_primary_override:
        print("ERROR: --guidance-live requires --guidance-i-verified-primary-override. "
              "Read the safety warning at the top of this file.", file=sys.stderr)
        sys.exit(1)

    configure_logging(args.guidance_log)

    cfg = GuidanceConfig(
        port=args.guidance_port,
        marker_class_id=args.guidance_marker_class,
        raw_order=args.guidance_raw_order,
        hover_throttle=args.guidance_hover_throttle,
        throttle_min=args.guidance_throttle_min,
        throttle_max=args.guidance_throttle_max,
        dry_run=args.guidance_dry_run,
        primary_override_verified=args.guidance_i_verified_primary_override,
        desired_box_height=args.guidance_desired_box_height,
        roll_sign=args.guidance_roll_sign,
        pitch_sign=args.guidance_pitch_sign,
        yaw_sign=args.guidance_yaw_sign,
        throttle_sign=args.guidance_throttle_sign,
        minimum_confidence=args.guidance_min_confidence,
        minimum_track_age=args.guidance_min_track_age,
        acquire_timeout=args.guidance_acquire_timeout,
        max_observation_age=args.guidance_max_observation_age,
        slew_us_per_second=args.guidance_slew,
    )
    return Guidance(cfg)


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    print(f"[init] TFLite backend: {_INTERPRETER_BACKEND}")
    labels = load_labels(args.labels)

    # ---- Load model -------------------------------------------------------
    interpreter = Interpreter(model_path=args.model, num_threads=args.num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_dtype = input_details[0]["dtype"]
    in_scale, in_zero_point = input_details[0].get("quantization", (0.0, 0))
    out_scale, out_zero_point = output_details[0].get("quantization", (0.0, 0))

    # Read the model's ACTUAL input shape rather than assuming NHWC. Some
    # export pipelines (e.g. certain onnx->tflite conversions) emit NCHW
    # (1,3,H,W) instead of the usual TFLite NHWC (1,H,W,3) -- feeding the
    # wrong layout is what caused "expected 3 for dimension 1" earlier.
    in_shape = list(input_details[0]["shape"])
    if len(in_shape) != 4:
        print(f"ERROR: unexpected input tensor rank {in_shape}, expected 4D", file=sys.stderr)
        sys.exit(1)

    if in_shape[1] == 3:
        input_layout = "NCHW"
        model_h, model_w = in_shape[2], in_shape[3]
    elif in_shape[3] == 3:
        input_layout = "NHWC"
        model_h, model_w = in_shape[1], in_shape[2]
    else:
        print(f"ERROR: could not infer input layout from shape {in_shape}", file=sys.stderr)
        sys.exit(1)

    if (model_h, model_w) != (args.input_size, args.input_size):
        print(f"[init] note: model expects {model_w}x{model_h}, overriding --input-size "
              f"({args.input_size})", file=sys.stderr)
        args.input_size = model_h  # assume square input, as documented

    out_shape = output_details[0]["shape"]
    num_classes = (out_shape[1] - 4) if out_shape[1] < out_shape[2] else (out_shape[2] - 4)

    allowed_classes = np.array(sorted({int(c) for c in args.classes.split(",") if c.strip() != ""}),
                                dtype=np.int32)

    print(f"[init] Model loaded ({args.model}), input {args.input_size}x{args.input_size}, "
          f"layout={input_layout}, threads={args.num_threads}, quantized_input={in_dtype}")
    print(f"[init] Keeping classes {allowed_classes.tolist()} "
          f"(conf>={args.conf_thresh}, nms_iou<={args.iou_thresh}); "
          f"tracker: match_iou>={args.track_iou_thresh}, smooth_alpha={args.smooth_alpha}, "
          f"track_life={args.track_life} inference cycles")

    # Preallocate the model input tensor buffer once, reused every inference,
    # in whatever layout the model actually expects.
    model_input = np.empty(tuple(in_shape), dtype=in_dtype)

    # ---- Guidance (validate/construct BEFORE opening camera/framebuffer so
    #      any config error fails fast and leaves no half-open resources). --
    guidance = _build_guidance(args, allowed_classes)

    # ---- Camera -------------------------------------------------------
    picam2 = Picamera2()
    cam_config = picam2.create_video_configuration(
        main={"size": (args.cam_width, args.cam_height), "format": "RGB888"}
    )
    picam2.configure(cam_config)
    picam2.start()
    time.sleep(1.0)  # sensor warm-up
    print(f"[init] Camera started at {args.cam_width}x{args.cam_height}")

    # ---- Framebuffer / PAL output -----------------------------------------
    fb_writer = None
    if not args.no_fb:
        try:
            fb_writer = FramebufferWriter(args.fb_device, args.cam_width, args.cam_height)
            print(f"[init] Writing composite output to {args.fb_device} "
                  f"({fb_writer.bpp} bpp)")
        except OSError as e:
            print(f"[warn] could not open {args.fb_device} ({e}); "
                  f"falling back to on-screen window", file=sys.stderr)
            fb_writer = None

    tracker = SimpleIoUTracker(
        iou_match_thresh=args.track_iou_thresh,
        smooth_alpha=args.smooth_alpha,
        max_misses=args.track_life,
    )
    last_infer_ms = 0.0

    # Start the guidance worker only now, after camera+fb are up, so its
    # serial retry loop can't fight the camera init for USB/CPU time and so
    # shutdown ordering is clean (guidance.close() before camera stop).
    if guidance is not None:
        guidance.start()
        mode = "DRY-RUN" if args.guidance_dry_run else "LIVE"
        print(f"[init] Guidance started ({mode}); marker_class="
              f"{args.guidance_marker_class}, port={args.guidance_port}")

    frame_count = 0
    infer_count = 0
    t_fps_window_start = time.time()
    frames_in_window = 0

    print("[run] entering main loop, Ctrl+C to stop")

    try:
        while _running:
            # Timestamp taken BEFORE capture_array() so the guidance module
            # accounts for capture-call duration as part of observation age.
            capture_started = time.monotonic()
            frame_rgb = picam2.capture_array("main")  # RGB888, no extra copy
            frame_count += 1

            if frame_count % args.infer_every == 0:
                t0 = time.time()

                # Resize + (if needed) requantize into the preallocated buffer
                resized = cv2.resize(frame_rgb, (args.input_size, args.input_size),
                                      interpolation=cv2.INTER_LINEAR)

                if in_dtype in (np.uint8, np.int8):
                    prepared = resized.astype(in_dtype)
                else:
                    prepared = (resized.astype(np.float32) / 255.0)

                if input_layout == "NCHW":
                    prepared = np.transpose(prepared, (2, 0, 1))  # HWC -> CHW

                np.copyto(model_input[0], prepared)

                interpreter.set_tensor(input_details[0]["index"], model_input)
                interpreter.invoke()
                raw_output = interpreter.get_tensor(output_details[0]["index"])

                if out_scale:
                    raw_output = (raw_output.astype(np.float32) - out_zero_point) * out_scale

                if infer_count < 5 or infer_count % 15 == 0:
                    print(f"[debug] pre-threshold confidence: "
                          f"{debug_confidence_stats(raw_output, num_classes, allowed_classes)}")

                boxes, scores, class_ids = postprocess(
                    raw_output, args.conf_thresh, args.iou_thresh, num_classes,
                    allowed_classes=allowed_classes,
                )
                # One tracker update per INFERENCE CYCLE, not per display frame --
                # this is what makes track_life count inference cycles rather
                # than raw display frames (see comment block above the tracker).
                tracker.update(boxes, scores, class_ids)
                last_infer_ms = (time.time() - t0) * 1000.0
                infer_count += 1

                # Submit ONLY on inference cycles, immediately after tracker
                # update: guidance ignores coasting (misses>0) tracks and
                # older snapshots, so doing this on skipped display frames
                # would just re-publish the same snapshot under a newer
                # timestamp and mask real staleness.
                if guidance is not None:
                    guidance.submit_tracks(
                        tracker.get_display_tracks(),
                        captured_at=capture_started,
                    )

            # Draw every currently-alive track on every frame (fresh or coasting)
            active_tracks = tracker.get_display_tracks()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            draw_detections(frame_bgr, active_tracks, labels, args.cam_width, args.cam_height)

            if fb_writer is not None:
                fb_writer.write(frame_bgr)
            else:
                cv2.imshow("preview", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames_in_window += 1
            if frame_count % args.log_every == 0:
                elapsed = time.time() - t_fps_window_start
                fps = frames_in_window / elapsed if elapsed > 0 else 0.0
                avg_conf = float(np.mean([t.score for t in active_tracks])) if active_tracks else 0.0
                coasting = sum(1 for t in active_tracks if t.misses > 0)
                print(f"[stats] display_fps={fps:.1f} last_infer_ms={last_infer_ms:.1f} "
                      f"infer_count={infer_count} active_tracks={len(active_tracks)} "
                      f"(coasting={coasting}) avg_conf={avg_conf:.2f}")
                t_fps_window_start = time.time()
                frames_in_window = 0

    finally:
        print("[shutdown] stopping guidance...")
        # Stop guidance BEFORE the camera so any in-flight MSP write finishes
        # while the main loop is still able to service SIGINT.
        if guidance is not None:
            try:
                guidance.close()
            except Exception:
                print("[shutdown] guidance.close() raised", file=sys.stderr)

        print("[shutdown] stopping camera and cleaning up...")
        try:
            picam2.stop()
        except Exception:
            pass
        if fb_writer is not None:
            fb_writer.close()
        else:
            cv2.destroyAllWindows()
        print("[shutdown] done")


if __name__ == "__main__":
    main()
