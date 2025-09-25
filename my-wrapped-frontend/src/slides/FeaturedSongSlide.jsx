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

function buildMirroredWavePath({
  width,
  height,
  amp,
  t,
}) {
  const steps = 32;
  const margin = 40;
  const points = [];
  const intensity = 0.18 + amp * 0.82;

  for (let i = 0; i <= steps; i += 1) {
    const y = (i / steps) * (height + margin * 2) - margin;
    const envelope = Math.sin((i / steps) * Math.PI);
    const wobble = Math.sin(t * 0.0015 + i * 0.6) * width * 0.08;
    const offset = envelope * width * intensity + wobble;
    points.push({ y, offset: Math.max(0, Math.min(width * 0.95, offset)) });
  }

  let leftPath = `M ${width} ${-margin}`;
  let rightPath = `M 0 ${-margin}`;

  for (let i = 1; i < points.length; i += 1) {
    const prev = points[i - 1];
    const curr = points[i];

    const leftCurrX = width - curr.offset;
    const rightCurrX = curr.offset;
    const midY = (prev.y + curr.y) / 2;
    const leftMidX = width - (prev.offset + curr.offset) / 2;
    const rightMidX = (prev.offset + curr.offset) / 2;

    leftPath += ` Q ${leftMidX} ${midY} ${leftCurrX} ${curr.y}`;
    rightPath += ` Q ${rightMidX} ${midY} ${rightCurrX} ${curr.y}`;
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
      const sideWidth = clientWidth / 2;
      const { leftPath, rightPath } = buildMirroredWavePath({
        width: sideWidth,
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
            <stop offset="0%" stopColor="rgba(59,130,246,0.75)" />
            <stop offset="100%" stopColor="rgba(168,85,247,0.5)" />
          </linearGradient>
          <linearGradient id={rightGradientId} x1="0" x2="1" y1="0" y2="1">
            <stop offset="0%" stopColor="rgba(168,85,247,0.75)" />
            <stop offset="100%" stopColor="rgba(59,130,246,0.5)" />
          </linearGradient>
        </defs>
        <g>
          <path ref={leftRef} fill={`url(#${leftGradientId})`} />
          <path ref={rightRef} fill={`url(#${rightGradientId})`} />
        </g>
      </svg>
      <div className="absolute inset-y-0 left-0 w-1/2 opacity-70 mix-blend-screen"
        style={{
          background:
            "radial-gradient(70% 60% at 80% 50%, rgba(59,130,246,0.6), rgba(59,130,246,0))",
        }}
      />
      <div className="absolute inset-y-0 right-0 w-1/2 opacity-70 mix-blend-screen"
        style={{
          background:
            "radial-gradient(70% 60% at 20% 50%, rgba(168,85,247,0.6), rgba(168,85,247,0))",
        }}
      />
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

  const bootAudio = async () => {
    try {
      if (!audioRef.current) return;
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

      await audioRef.current.play();
      setIsReady(true);
      setIsPlaying(true);
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

  const bgLightness = 12 + level * 18;
  const background = `linear-gradient(135deg, hsl(248 68% ${bgLightness + 8}%), hsl(268 72% ${bgLightness}%))`;

  return (
    <div
      className="relative flex min-h-screen w-full flex-col items-center justify-center overflow-hidden px-6 py-16 text-white"
      style={{ background }}
    >
      <MirroredWave level={level} />

      <div className="relative z-10 flex max-w-3xl flex-col items-center gap-10 text-center">
        <div className="flex flex-col gap-2 text-sm uppercase tracking-[0.4em] text-white/70">
          <span>Your top song is</span>
        </div>

        <div className="flex w-full flex-col items-center justify-center gap-10 sm:flex-row sm:text-left">
          <div className="relative w-60 max-w-xs overflow-hidden rounded-3xl border border-white/20 bg-white/5 shadow-2xl shadow-indigo-900/40">
            <img
              src={data.albumArtUrl}
              alt={`${data.title} album cover`}
              className="h-full w-full object-cover"
            />
            {!isPlaying && (
              <button
                onClick={bootAudio}
                className="absolute inset-0 flex items-center justify-center bg-black/50 text-sm font-semibold uppercase tracking-widest text-white transition hover:bg-black/40"
              >
                {isReady ? "Resume" : "Play"}
              </button>
            )}
          </div>

          <div className="flex max-w-md flex-col items-center sm:items-start">
            <h1 className="text-4xl font-extrabold sm:text-5xl md:text-6xl">{data.title}</h1>
            <p className="mt-4 text-lg font-medium text-white/80">{data.artist}</p>
            <p className="text-sm uppercase tracking-widest text-white/60">{data.album}</p>

            {isPlaying && (
              <button
                onClick={() => {
                  if (!audioRef.current) return;
                  audioRef.current.pause();
                  setIsPlaying(false);
                }}
                className="mt-8 rounded-full border border-white/30 px-5 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-white/80 transition hover:border-white/60 hover:text-white"
              >
                Pause
              </button>
            )}
          </div>
        </div>
      </div>

      <audio ref={audioRef} src={data.audioUrl} preload="auto" />
    </div>
  );
}
