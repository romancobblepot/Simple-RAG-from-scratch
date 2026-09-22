/**
 * Tiny WebAudio sound effects — no audio assets needed.
 * Soft tick-triplet loops while the bot is "thinking",
 * and a two-note chime when the answer lands.
 */

let audioCtx: AudioContext | null = null;

function getContext(): AudioContext | null {
  try {
    if (!audioCtx) {
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!Ctor) return null;
      audioCtx = new Ctor();
    }
    if (audioCtx.state === "suspended") void audioCtx.resume();
    return audioCtx;
  } catch {
    return null;
  }
}

function tone(
  ctx: AudioContext,
  freq: number,
  startAt: number,
  duration: number,
  peakGain: number,
  type: OscillatorType = "sine",
) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0, startAt);
  gain.gain.linearRampToValueAtTime(peakGain, startAt + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, startAt + duration);
  osc.connect(gain).connect(ctx.destination);
  osc.start(startAt);
  osc.stop(startAt + duration + 0.05);
}

/**
 * Plays a soft three-tick pattern (synced with the three dots) on repeat.
 * Returns a stop function — call it when loading finishes.
 */
export function startThinkingSound(): () => void {
  const ctx = getContext();
  if (!ctx) return () => {};

  let cancelled = false;
  let timer: number | undefined;

  const playTriplet = () => {
    if (cancelled) return;
    const t = ctx.currentTime;
    tone(ctx, 523.25, t, 0.08, 0.03); // C5
    tone(ctx, 392.0, t + 0.22, 0.08, 0.025); // G4
    tone(ctx, 523.25, t + 0.44, 0.1, 0.03); // C5
    timer = window.setTimeout(playTriplet, 1500);
  };

  playTriplet();

  return () => {
    cancelled = true;
    if (timer !== undefined) window.clearTimeout(timer);
  };
}

/** Two-note chime when an answer is generated. */
export function playAnswerSound() {
  const ctx = getContext();
  if (!ctx) return;
  const t = ctx.currentTime;
  tone(ctx, 659.25, t, 0.12, 0.045, "triangle"); // E5
  tone(ctx, 880.0, t + 0.11, 0.18, 0.045, "triangle"); // A5
}
