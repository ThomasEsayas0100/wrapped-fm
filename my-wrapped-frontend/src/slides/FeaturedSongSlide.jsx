import { useEffect, useMemo, useRef, useState } from "react";

const defaultData = {
  title: "Get Lucky",
  artist: "Daft Punk",
  album: "Random Access Memories",
  albumArtUrl:
    "https://upload.wikimedia.org/wikipedia/en/a/a7/Random_Access_Memories.jpg",
  youtubeUrl: "https://www.youtube.com/watch?v=5NV6Rdv1a3I",
  previewAudioUrl:
    "https://cdn.pixabay.com/download/audio/2022/03/15/audio_3b0e4b2d55.mp3?filename=disco-110134.mp3",
};

function getYouTubeEmbedUrl(url) {
  if (!url) return null;

  const patterns = [
    /(?:v=|\/v\/|youtu\.be\/|embed\/)([A-Za-z0-9_-]{11})/, // common formats
  ];

  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match && match[1]) {
      return `https://www.youtube.com/embed/${match[1]}?autoplay=1&controls=0&playsinline=1`;
    }
  }

  return null;
}

function clamp(n, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, n));
}

function distributeFrequencies(
  data,
  startIndex,
  endIndex,
  buckets,
  curve = 0.85
) {
  const span = Math.max(1, endIndex - startIndex);
  const bucketSize = span / buckets;
  const result = [];

  for (let bucket = 0; bucket < buckets; bucket += 1) {
    const bucketStart = Math.floor(startIndex + bucket * bucketSize);
    const bucketEnd = Math.floor(startIndex + (bucket + 1) * bucketSize);
    let sum = 0;
    let count = 0;

    for (let i = bucketStart; i < bucketEnd; i += 1) {
      const sample = data[i];
      if (typeof sample === "number") {
        sum += sample;
        count += 1;
      }
    }

    const average = count > 0 ? sum / count : 0;
    const normalized = clamp(average / 255);
    const curved = Math.pow(normalized, curve);
    result.push(clamp(curved * 1.35));
  }

  return result;
}

function smoothLevels(previous, next, inertia = 0.65) {
  const result = [];
  let hasChanged = false;

  const maxLength = Math.max(previous.length, next.length);

  for (let i = 0; i < maxLength; i += 1) {
    const prev = previous[i] ?? previous[previous.length - 1] ?? 0;
    const target = next[i] ?? next[next.length - 1] ?? 0;
    const value = prev * inertia + target * (1 - inertia);
    if (!hasChanged && Math.abs(value - prev) > 0.003) {
      hasChanged = true;
    }
    result[i] = clamp(value);
  }

  return hasChanged ? result : previous;
}

function generateSyntheticLevels(count, now, speed = 0.0028) {
  const levels = [];

  for (let i = 0; i < count; i += 1) {
    const phase = i * 0.85;
    const base = 0.35 + Math.sin(now * speed + phase) * 0.25;
    const accent = Math.sin(now * speed * 1.7 + phase * 1.8) * 0.18;
    const wobble = Math.sin(now * speed * 0.4 + phase * 2.6) * 0.12;
    levels.push(clamp(base + accent + wobble));
  }

  return levels;
}

function useAudioSpectrum(audioRef, isActive, { lowBars, highBars, onModeChange }) {
  const [lowLevels, setLowLevels] = useState(() => new Array(lowBars).fill(0));
  const [highLevels, setHighLevels] = useState(() => new Array(highBars).fill(0));

  const analyserRef = useRef(null);
  const contextRef = useRef(null);
  const sourceRef = useRef(null);
  const bufferRef = useRef(null);
  const modeRef = useRef("idle");
  const fallbackRef = useRef(true);
  const setupAttemptedRef = useRef(false);

  useEffect(() => {
    setLowLevels(new Array(lowBars).fill(0));
  }, [lowBars]);

  useEffect(() => {
    setHighLevels(new Array(highBars).fill(0));
  }, [highBars]);

  useEffect(() => {
    const audioEl = audioRef.current;
    if (audioEl) {
      audioEl.crossOrigin = "anonymous";
    }
  }, [audioRef]);

  useEffect(() => {
    let raf = 0;
    let mounted = true;

    async function ensureAnalyser() {
      const audioEl = audioRef.current;
      if (!isActive || !audioEl || !audioEl.src) {
        fallbackRef.current = true;
        updateMode("synthetic");
        return;
      }

      try {
        if (!contextRef.current) {
          const AudioCtx = window.AudioContext || window.webkitAudioContext;
          contextRef.current = new AudioCtx();
        }
        const ctx = contextRef.current;
        if (ctx.state === "suspended") {
          await ctx.resume();
        }

        if (!sourceRef.current) {
          const source = ctx.createMediaElementSource(audioEl);
          const analyser = ctx.createAnalyser();
          analyser.fftSize = 1024;
          analyser.smoothingTimeConstant = 0.82;
          const gain = ctx.createGain();
          gain.gain.value = 0;

          source.connect(analyser);
          analyser.connect(gain);
          gain.connect(ctx.destination);

          sourceRef.current = source;
          analyserRef.current = analyser;
          bufferRef.current = new Uint8Array(analyser.frequencyBinCount);
        }

        fallbackRef.current = false;
        updateMode("audio");
      } catch (error) {
        console.warn("Falling back to synthetic equalizer", error);
        fallbackRef.current = true;
        updateMode("synthetic");
      }
    }

    function updateMode(nextMode) {
      if (modeRef.current !== nextMode) {
        modeRef.current = nextMode;
        if (onModeChange) onModeChange(nextMode);
      }
    }

    function tick() {
      if (!mounted) return;

      const now = performance.now();

      if (isActive) {
        if (!setupAttemptedRef.current) {
          setupAttemptedRef.current = true;
          void ensureAnalyser();
        }

        if (!fallbackRef.current && analyserRef.current && bufferRef.current) {
          const analyser = analyserRef.current;
          const buffer = bufferRef.current;
          analyser.getByteFrequencyData(buffer);

          const midpoint = Math.floor(buffer.length / 2);
          const low = distributeFrequencies(buffer, 0, midpoint, lowBars);
          const high = distributeFrequencies(
            buffer,
            midpoint,
            buffer.length,
            highBars
          );

          setLowLevels((prev) => smoothLevels(prev, low, 0.58));
          setHighLevels((prev) => smoothLevels(prev, high, 0.6));
        } else {
          updateMode("synthetic");
          const syntheticLow = generateSyntheticLevels(lowBars, now, 0.0024);
          const syntheticHigh = generateSyntheticLevels(highBars, now + 260, 0.0029);
          setLowLevels((prev) => smoothLevels(prev, syntheticLow, 0.8));
          setHighLevels((prev) => smoothLevels(prev, syntheticHigh, 0.8));
        }
      } else {
        setupAttemptedRef.current = false;
        fallbackRef.current = false;
        updateMode("idle");
        const zerosLow = new Array(lowBars).fill(0);
        const zerosHigh = new Array(highBars).fill(0);
        setLowLevels((prev) => smoothLevels(prev, zerosLow, 0.7));
        setHighLevels((prev) => smoothLevels(prev, zerosHigh, 0.7));
      }

      raf = requestAnimationFrame(tick);
    }

    tick();

    return () => {
      mounted = false;
      cancelAnimationFrame(raf);
    };
  }, [audioRef, isActive, lowBars, highBars, onModeChange]);

  return useMemo(
    () => ({
      lowLevels,
      highLevels,
    }),
    [lowLevels, highLevels]
  );
}

function EqualizerSide({ side, levels }) {
  const isLeft = side === "left";

  return (
    <div
      className={`pointer-events-none absolute inset-y-[-10%] ${
        isLeft ? "left-[-10%] justify-start" : "right-[-10%] justify-end"
      } flex w-[26vw] min-w-[200px]`}
    >
      <div className="relative flex h-full w-full items-center justify-center">
        <div
          className={`absolute inset-0 ${
            isLeft ? "bg-gradient-to-tr" : "bg-gradient-to-tl"
          } from-sky-500/18 via-cyan-400/8 to-transparent blur-3xl`}
        />
        <div
          className={`relative z-10 flex h-[78%] items-end gap-[clamp(0.45rem,1vw,0.85rem)] ${
            isLeft ? "" : "flex-row-reverse"
          }`}
        >
          {levels.map((value, index) => (
            <div key={index} className="flex-1">
              <div className="relative mx-auto flex h-full w-[clamp(0.55rem,1.15vw,1rem)] overflow-hidden rounded-full bg-white/10">
                <div
                  className="absolute bottom-0 left-0 right-0 origin-bottom rounded-full bg-gradient-to-t from-sky-500 via-cyan-400 to-sky-200 shadow-[0_0_24px_rgba(56,189,248,0.35)]"
                  style={{
                    transform: `scaleY(${0.15 + value * 0.85})`,
                    transition: "transform 0.1s ease-out",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function FeaturedSongSlide({ data = defaultData }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [analysisMode, setAnalysisMode] = useState("idle");
  const previewAudioRef = useRef(null);

  const embedUrl = isPlaying ? getYouTubeEmbedUrl(data.youtubeUrl) : null;

  const { lowLevels, highLevels } = useAudioSpectrum(previewAudioRef, isPlaying, {
    lowBars: 9,
    highBars: 9,
    onModeChange: setAnalysisMode,
  });

  useEffect(() => {
    const audio = previewAudioRef.current;
    if (!audio) return;

    audio.muted = true;
    audio.loop = true;
    audio.preload = "auto";
    audio.playsInline = true;
  }, []);

  async function handleTogglePlayback() {
    if (isPlaying) {
      setIsPlaying(false);
      if (previewAudioRef.current) {
        previewAudioRef.current.pause();
        previewAudioRef.current.currentTime = 0;
      }
      return;
    }

    try {
      if (previewAudioRef.current && previewAudioRef.current.src) {
        previewAudioRef.current.currentTime = 0;
        await previewAudioRef.current.play();
      }
    } catch (error) {
      console.warn("Preview audio playback blocked", error);
    }

    setIsPlaying(true);
  }

  return (
    <div className="relative flex min-h-screen w-full items-center justify-center overflow-hidden bg-[#070c1f] px-6 py-16 text-white">
      <EqualizerSide side="left" levels={lowLevels} />
      <EqualizerSide side="right" levels={highLevels} />

      <div className="relative z-10 flex max-w-3xl flex-col items-center gap-12 text-center">
        <h1 className="text-4xl font-bold tracking-tight text-white md:text-5xl">
          Your top song was...
          <span className="ml-4 text-3xl align-middle md:text-4xl" aria-hidden="true">
            🥁🥁🥁
          </span>
        </h1>

        <div className="flex flex-col items-center gap-8">
          <div className="rounded-[2.5rem] border-[14px] border-[#191f3c] bg-[#0c1430] p-4 shadow-[0_25px_65px_rgba(10,12,35,0.55)]">
            <img
              src={data.albumArtUrl}
              alt={data.album ? `${data.title} album cover for ${data.album}` : `${data.title} album cover`}
              className="h-64 w-64 rounded-[1.75rem] object-cover md:h-72 md:w-72"
            />
          </div>

          <div className="space-y-2">
            <p className="text-4xl font-bold md:text-5xl">{data.title}</p>
            <p className="text-xl font-medium text-white/90 md:text-2xl">{data.artist}</p>
            <p className="text-sm uppercase tracking-[0.35em] text-white/60 md:text-base">
              {data.album}
            </p>
          </div>
        </div>

        {data.youtubeUrl ? (
          <div className="flex flex-col items-center gap-4 text-sm text-white/80">
            <button
              type="button"
              onClick={handleTogglePlayback}
              className="rounded-full bg-white/10 px-6 py-2 font-medium tracking-wide transition hover:bg-white/20"
            >
              {isPlaying ? "Stop playback" : "Play track"}
            </button>

            {embedUrl && (
              <iframe
                key={embedUrl}
                src={embedUrl}
                title="Top song audio player"
                allow="autoplay"
                aria-hidden="true"
                tabIndex={-1}
                className="absolute h-px w-px overflow-hidden"
                style={{ clip: "rect(0 0 0 0)" }}
              />
            )}
            {analysisMode === "synthetic" && (
              <p className="text-xs text-white/60">
                Visualizer running in ambient mode.
              </p>
            )}
          </div>
        ) : null}
      </div>

      {data.previewAudioUrl ? (
        <audio ref={previewAudioRef} src={data.previewAudioUrl} />
      ) : (
        <audio ref={previewAudioRef} />
      )}
    </div>
  );
}
