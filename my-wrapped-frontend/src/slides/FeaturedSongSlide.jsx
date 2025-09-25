import { useEffect, useMemo, useRef, useState } from "react";

const defaultData = {
  title: "Get Lucky",
  artist: "Daft Punk",
  album: "Random Access Memories",
  albumArtUrl:
    "https://upload.wikimedia.org/wikipedia/en/a/a7/Random_Access_Memories.jpg",
  audioUrl:
    "https://cdn.pixabay.com/download/audio/2022/03/15/audio_3b0e4b2d55.mp3?filename=disco-110134.mp3",
};

function clamp(n, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, n));
}

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const v = (samples[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / samples.length);
}

function buildMirroredWavePath({ width, height, amp, t }) {
  const steps = 64;
  const margin = 72;
  const points = [];
  const baseStrength = 0.28 + amp * 0.55;
  const rippleStrength = 0.12 + amp * 0.18;

  for (let i = 0; i <= steps; i += 1) {
    const progress = i / steps;
    const y = progress * (height + margin * 2) - margin;
    const bell = Math.pow(Math.sin(progress * Math.PI), 1.4);
    const ripple =
      Math.sin(t * 0.0011 + progress * 11) * rippleStrength +
      Math.sin(t * 0.0007 + progress * 5.5) * rippleStrength * 0.55;
    const offset = width * (baseStrength * bell + 0.05) + ripple * width;

    points.push({
      y,
      offset: Math.max(width * 0.14, Math.min(width * 0.98, offset)),
    });
  }

  let leftPath = `M ${width} ${-margin}`;
  let rightPath = `M 0 ${-margin}`;

  for (let i = 1; i < points.length; i += 1) {
    const prev = points[i - 1];
    const curr = points[i];

    const yMid = (prev.y + curr.y) / 2;
    const leftPrevX = width - prev.offset;
    const leftCurrX = width - curr.offset;
    const rightPrevX = prev.offset;
    const rightCurrX = curr.offset;

    const leftCtrlX = width - (prev.offset + curr.offset) / 2;
    const rightCtrlX = (prev.offset + curr.offset) / 2;

    leftPath += ` C ${leftPrevX} ${yMid}, ${leftCtrlX} ${yMid}, ${leftCurrX} ${curr.y}`;
    rightPath += ` C ${rightPrevX} ${yMid}, ${rightCtrlX} ${yMid}, ${rightCurrX} ${curr.y}`;
  }

  leftPath += ` L 0 ${height + margin} L ${width} ${height + margin} Z`;
  rightPath += ` L ${width} ${height + margin} L 0 ${height + margin} Z`;

  return { leftPath, rightPath };
}

function MirroredWave({ level }) {
  const svgRef = useRef(null);
  const leftRef = useRef(null);
  const rightRef = useRef(null);

  useEffect(() => {
    let raf = 0;

    const update = (now) => {
      if (!svgRef.current || !leftRef.current || !rightRef.current) {
        raf = requestAnimationFrame(update);
        return;
      }

      const { clientWidth, clientHeight } = svgRef.current;
      const halfWidth = clientWidth / 2;
      const { leftPath, rightPath } = buildMirroredWavePath({
        width: halfWidth,
        height: clientHeight,
        amp: level,
        t: now,
      });

      leftRef.current.setAttribute("d", leftPath);
      rightRef.current.setAttribute("d", rightPath);
      raf = requestAnimationFrame(update);
    };

    raf = requestAnimationFrame(update);
    return () => cancelAnimationFrame(raf);
  }, [level]);

  const leftGradientId = useMemo(
    () => `wave-left-${Math.random().toString(36).slice(2)}`,
    [],
  );
  const rightGradientId = useMemo(
    () => `wave-right-${Math.random().toString(36).slice(2)}`,
    [],
  );

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      <svg ref={svgRef} className="absolute inset-0 h-full w-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id={leftGradientId} x1="1" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="rgba(129,140,248,0.9)" />
            <stop offset="100%" stopColor="rgba(168,85,247,0.45)" />
          </linearGradient>
          <linearGradient id={rightGradientId} x1="0" x2="1" y1="0" y2="1">
            <stop offset="0%" stopColor="rgba(168,85,247,0.9)" />
            <stop offset="100%" stopColor="rgba(129,140,248,0.45)" />
          </linearGradient>
        </defs>
        <g>
          <path ref={leftRef} fill={`url(#${leftGradientId})`} />
          <path ref={rightRef} fill={`url(#${rightGradientId})`} />
        </g>
      </svg>
      <div className="absolute inset-y-0 left-1/2 h-full w-[28rem] -translate-x-1/2 bg-gradient-to-r from-transparent via-white/10 to-transparent blur-3xl" />
      <div className="absolute inset-0 bg-[radial-gradient(120%_130%_at_12%_40%,rgba(99,102,241,0.28),transparent)]" />
      <div className="absolute inset-0 bg-[radial-gradient(120%_130%_at_88%_60%,rgba(192,132,252,0.24),transparent)]" />
    </div>
  );
}

export default function FeaturedSongSlide({ data = defaultData }) {
  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [level, setLevel] = useState(0);

  const audioRef = useRef(null);
  const ctxRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);

  const ensureAudioGraph = () => {
    if (!audioRef.current) return null;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = ctxRef.current ?? new AudioCtx();
    ctxRef.current = ctx;

    if (!sourceRef.current) {
      const source = ctx.createMediaElementSource(audioRef.current);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.88;
      source.connect(analyser);
      analyser.connect(ctx.destination);
      sourceRef.current = source;
      analyserRef.current = analyser;
    }

    return ctx;
  };

  const togglePlayback = async () => {
    try {
      if (!audioRef.current) return;
      const ctx = ensureAudioGraph();
      if (!ctx) return;

      if (!isReady) {
        setIsReady(true);
      }

      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        await ctx.resume();
        await audioRef.current.play();
        setIsPlaying(true);
      }
    } catch (err) {
      console.warn("Audio playback failed", err);
    }
  };

  useEffect(() => {
    const buffer = new Uint8Array(1024);
    let raf = 0;

    const loop = () => {
      const analyser = analyserRef.current;
      if (analyser) {
        analyser.getByteTimeDomainData(buffer);
        const value = clamp(rms(buffer) * 1.9);
        setLevel(value);
      }
      raf = requestAnimationFrame(loop);
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return undefined;

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handlePause);

    return () => {
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handlePause);
    };
  }, []);

  const depth = 22 + level * 14;
  const background = `linear-gradient(115deg, hsl(249 68% ${18 + depth / 2}%), hsl(270 75% ${30 + depth}%))`;

  return (
    <div
      className="relative flex min-h-screen w-full items-center justify-center overflow-hidden px-6 py-12 text-white md:px-16"
      style={{ background }}
    >
      <MirroredWave level={level} />

      <div className="relative z-10 flex w-full max-w-6xl flex-col items-center gap-16 md:flex-row md:items-center md:justify-between">
        <div className="relative flex shrink-0 items-center justify-center">
          <div className="relative h-64 w-64 overflow-hidden rounded-[2.5rem] border border-white/15 bg-white/10 shadow-[0_30px_60px_-25px_rgba(15,22,86,0.7)] backdrop-blur-sm transition-transform duration-700 ease-out hover:scale-[1.02] md:h-72 md:w-72">
            <img
              src={data.albumArtUrl}
              alt={`${data.title} album cover`}
              className="h-full w-full object-cover"
            />
            <button
              type="button"
              onClick={togglePlayback}
              className="absolute bottom-6 left-1/2 flex h-14 w-14 -translate-x-1/2 items-center justify-center rounded-full bg-white/90 text-[0.6rem] font-semibold uppercase tracking-[0.35em] text-indigo-950 shadow-xl transition hover:bg-white"
            >
              {isPlaying ? "Pause" : isReady ? "Play" : "Start"}
            </button>
          </div>
        </div>

        <div className="flex max-w-xl flex-col items-center text-center md:items-start md:text-left">
          <span className="text-[0.75rem] uppercase tracking-[0.55em] text-white/60">Your top song is</span>
          <h1 className="mt-6 text-5xl font-black tracking-tight md:text-6xl lg:text-7xl">{data.title}</h1>
          <p className="mt-6 text-2xl font-medium text-white/80 md:text-[1.75rem]">{data.artist}</p>
          <p className="mt-2 text-sm uppercase tracking-[0.5em] text-white/50">{data.album}</p>
        </div>
      </div>

      <audio ref={audioRef} src={data.audioUrl} preload="auto" />
    </div>
  );
}
