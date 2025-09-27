import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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

function extractYouTubeVideoId(url) {
  if (!url) return null;

  const patterns = [
    /(?:v=|\bv\/|youtu\.be\/|embed\/)([A-Za-z0-9_-]{11})/,
  ];

  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match && match[1]) {
      return match[1];
    }
  }

  return null;
}

function getYouTubeEmbedUrl(url) {
  const videoId = extractYouTubeVideoId(url);
  if (!videoId) return null;
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const params = new URLSearchParams({
    autoplay: "1",
    controls: "0",
    playsinline: "1",
    enablejsapi: "1",
    modestbranding: "1",
  });
  if (origin) params.set("origin", origin);
  return `https://www.youtube.com/embed/${videoId}?${params.toString()}`;
}

async function resolveYouTubeAudioStream(videoId, { signal } = {}) {
  if (!videoId) return null;

  const controller = signal ? undefined : new AbortController();
  const finalSignal = signal ?? controller?.signal;

  const endpoint = `https://piped.video/api/v1/streams/${videoId}`;

  const response = await fetch(endpoint, { signal: finalSignal, headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`YouTube audio stream request failed with ${response.status}`);
  }

  const payload = await response.json();
  const audioStreams = Array.isArray(payload?.audioStreams) ? payload.audioStreams : [];

  if (!audioStreams.length) {
    throw new Error("No audio streams available for YouTube source");
  }

  const sorted = [...audioStreams].sort((a, b) => {
    const bitrateA = typeof a.bitrate === "number" ? a.bitrate : parseInt(a.bitrate ?? "0", 10) || 0;
    const bitrateB = typeof b.bitrate === "number" ? b.bitrate : parseInt(b.bitrate ?? "0", 10) || 0;
    return bitrateB - bitrateA;
  });

  for (const stream of sorted) {
    if (stream?.url) {
      return stream.url;
    }
  }

  throw new Error("Unable to resolve playable YouTube audio stream");
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

function generateSyntheticLevels(count, now, speed = 0.0026) {
  const levels = [];

  for (let i = 0; i < count; i += 1) {
    const phase = i * 0.75;
    const base = 0.28 + Math.sin(now * speed + phase) * 0.22;
    const accent = Math.sin(now * speed * 1.45 + phase * 1.6) * 0.16;
    const wobble = Math.sin(now * speed * 0.35 + phase * 2.2) * 0.12;
    levels.push(clamp(base + accent + wobble));
  }

  return levels;
}

function useAudioSpectrum(
  audioRef,
  isActive,
  { lowBars, highBars, onModeChange, sourceLabel, sourceVersion },
  tabAudioStream
) {
  const [lowLevels, setLowLevels] = useState(() => new Array(lowBars).fill(0));
  const [highLevels, setHighLevels] = useState(() => new Array(highBars).fill(0));

  const analyserRef = useRef(null);
  const contextRef = useRef(null);
  const sourceRef = useRef(null);
  const elementSourceRef = useRef(null);
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
    setupAttemptedRef.current = false;
  }, [sourceVersion]);

  useEffect(() => {
    setupAttemptedRef.current = false;
  }, [tabAudioStream]);

  useEffect(() => {
    let raf = 0;
    let mounted = true;

    async function ensureAnalyser() {
      const audioEl = audioRef.current;
      const shouldAnalyse = isActive || Boolean(tabAudioStream);
      const hasElementSource = Boolean(audioEl && audioEl.src);
      const hasTabStream = Boolean(tabAudioStream);

      if (!shouldAnalyse || (!hasElementSource && !hasTabStream)) {
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

        const desiredType = hasTabStream ? "tab" : "element";
        const needsRebuild =
          !sourceRef.current ||
          sourceRef.current.type !== desiredType ||
          (desiredType === "tab" && sourceRef.current.stream !== tabAudioStream);

        if (needsRebuild) {
          if (sourceRef.current) {
            try {
              sourceRef.current.node.disconnect();
            } catch (error) {
              console.warn("Unable to disconnect previous audio node", error);
            }
            if (sourceRef.current.gain) {
              try {
                sourceRef.current.gain.disconnect();
              } catch (error) {
                console.warn("Unable to disconnect previous gain node", error);
              }
            }
            sourceRef.current = null;
          }

          if (analyserRef.current) {
            try {
              analyserRef.current.disconnect();
            } catch {
              // ignore
            }
            analyserRef.current = null;
          }

          const analyser = ctx.createAnalyser();
          analyser.fftSize = 1024;
          analyser.smoothingTimeConstant = 0.82;
          const gain = ctx.createGain();
          gain.gain.value = 0;

          let sourceNode;

          if (desiredType === "tab") {
            sourceNode = ctx.createMediaStreamSource(tabAudioStream);
            sourceRef.current = {
              node: sourceNode,
              type: "tab",
              stream: tabAudioStream,
              gain,
            };
          } else {
            if (!elementSourceRef.current) {
              elementSourceRef.current = ctx.createMediaElementSource(audioEl);
            }
            sourceNode = elementSourceRef.current;
            sourceRef.current = {
              node: sourceNode,
              type: "element",
              gain,
            };
          }

          sourceNode.connect(analyser);
          analyser.connect(gain);
          gain.connect(ctx.destination);

          analyserRef.current = analyser;
          bufferRef.current = new Uint8Array(analyser.frequencyBinCount);
        }

        fallbackRef.current = false;
        if (hasTabStream) {
          updateMode("audio:tab-capture");
        } else {
          updateMode(sourceLabel ? `audio:${sourceLabel}` : "audio");
        }
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
      const audioEl = audioRef.current;
      const hasElementSource = Boolean(audioEl && audioEl.src);
      const hasTabStream = Boolean(tabAudioStream);
      const shouldAnalyse = isActive || hasTabStream;

      if (!hasElementSource && !hasTabStream) {
        fallbackRef.current = true;
        updateMode("synthetic");
        const syntheticLow = generateSyntheticLevels(lowBars, now, 0.0024);
        const syntheticHigh = generateSyntheticLevels(highBars, now + 260, 0.0029);
        setLowLevels((prev) => smoothLevels(prev, syntheticLow, 0.8));
        setHighLevels((prev) => smoothLevels(prev, syntheticHigh, 0.8));
        raf = requestAnimationFrame(tick);
        return;
      }

      if (!shouldAnalyse) {
        setupAttemptedRef.current = false;
        fallbackRef.current = true;
        updateMode("synthetic");
        const syntheticLow = generateSyntheticLevels(lowBars, now, 0.0024);
        const syntheticHigh = generateSyntheticLevels(highBars, now + 260, 0.0029);
        setLowLevels((prev) => smoothLevels(prev, syntheticLow, 0.8));
        setHighLevels((prev) => smoothLevels(prev, syntheticHigh, 0.8));
        raf = requestAnimationFrame(tick);
        return;
      }

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

      raf = requestAnimationFrame(tick);
    }

    tick();

    return () => {
      mounted = false;
      cancelAnimationFrame(raf);
    };
  }, [audioRef, isActive, lowBars, highBars, onModeChange, sourceLabel, tabAudioStream]);

  return useMemo(
    () => ({
      lowLevels,
      highLevels,
    }),
    [lowLevels, highLevels]
  );
}

function EqualizerRibbon({ side, levels }) {
  const canvasRef = useRef(null);
  const latestLevelsRef = useRef(levels);
  latestLevelsRef.current = levels;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return undefined;

    let animationFrame;

    function draw() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const ribbonLevels = latestLevelsRef.current;
      if (!ribbonLevels || ribbonLevels.length < 2) {
        animationFrame = requestAnimationFrame(draw);
        return;
      }

      const width = rect.width;
      const height = rect.height;
      const outerX = side === "left" ? width : 0;
      const direction = side === "left" ? -1 : 1;
      const verticalPadding = height * 0.12;
      const baseThickness = width * 0.22;
      const liveAmplitude = width * 0.58;

      const gradient = ctx.createLinearGradient(
        side === "left" ? width : 0,
        0,
        side === "left" ? 0 : width,
        height
      );
      gradient.addColorStop(0, "rgba(59, 130, 246, 0.9)");
      gradient.addColorStop(0.4, "rgba(14, 165, 233, 0.75)");
      gradient.addColorStop(1, "rgba(125, 211, 252, 0.35)");

      ctx.beginPath();
      ctx.moveTo(outerX, -verticalPadding);

      const step = (height + verticalPadding * 2) / (ribbonLevels.length - 1);
      let previousInnerX = outerX;
      let previousY = -verticalPadding;

      for (let i = 0; i < ribbonLevels.length; i += 1) {
        const level = ribbonLevels[i];
        const thickness = baseThickness + level * liveAmplitude;
        const innerX = outerX + direction * thickness;
        const y = -verticalPadding + step * i;

        const controlX = previousInnerX + (innerX - previousInnerX) * 0.68;
        const controlY = previousY + step * 0.68;

        ctx.quadraticCurveTo(controlX, controlY, innerX, y);

        previousInnerX = innerX;
        previousY = y;
      }

      ctx.lineTo(outerX, height + verticalPadding);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();

      animationFrame = requestAnimationFrame(draw);
    }

    animationFrame = requestAnimationFrame(draw);

    return () => cancelAnimationFrame(animationFrame);
  }, [side]);

  return (
    <div
      className={`pointer-events-none absolute inset-y-[-14%] ${
        side === "left" ? "left-[-10%]" : "right-[-10%]"
      } w-[clamp(240px,25vw,340px)]`}
    >
      <canvas ref={canvasRef} className="h-full w-full" />
      <div
        className={`absolute inset-0 ${
          side === "left" ? "bg-gradient-to-tr" : "bg-gradient-to-tl"
        } from-sky-500/25 via-cyan-400/10 to-transparent blur-3xl mix-blend-screen`}
      />
    </div>
  );
}

export default function FeaturedSongSlide({ data = defaultData }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [analysisMode, setAnalysisMode] = useState("idle");
  const [tabAudioStream, setTabAudioStream] = useState(null);
  const [isTabCaptureLoading, setIsTabCaptureLoading] = useState(false);
  const [tabCaptureError, setTabCaptureError] = useState(null);
  const previewAudioRef = useRef(null);

  const [audioSource, setAudioSource] = useState(() => ({
    url: data.previewAudioUrl ?? null,
    label: data.previewAudioUrl ? "preview" : "none",
    status: data.previewAudioUrl ? "ready" : "idle",
  }));

  const stopTabAudioCapture = useCallback(() => {
    setTabAudioStream((current) => {
      if (current) {
        current.getTracks().forEach((track) => track.stop());
      }
      return null;
    });
    setTabCaptureError(null);
  }, []);

  const startTabAudioCapture = useCallback(async () => {
    if (isTabCaptureLoading) return;

    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getDisplayMedia !== "function"
    ) {
      setTabCaptureError("Tab audio capture is not supported in this browser.");
      return;
    }

    try {
      setTabCaptureError(null);
      setIsTabCaptureLoading(true);

      const stream = await navigator.mediaDevices.getDisplayMedia({
        audio: true,
        video: false,
      });

      if (!stream) {
        setTabCaptureError("No stream was returned when starting tab capture.");
        setIsTabCaptureLoading(false);
        return;
      }

      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) {
        stream.getTracks().forEach((track) => track.stop());
        setTabCaptureError("No audio tracks found in the shared stream.");
        setIsTabCaptureLoading(false);
        return;
      }

      stream.getVideoTracks().forEach((track) => track.stop());

      setTabAudioStream((previous) => {
        if (previous && previous !== stream) {
          previous.getTracks().forEach((track) => track.stop());
        }
        return stream;
      });

      setIsPlaying(false);
      const audio = previewAudioRef.current;
      if (audio) {
        audio.pause();
        audio.currentTime = 0;
      }
    } catch (error) {
      if (error?.name !== "AbortError") {
        console.warn("Tab audio capture failed", error);
      }
      if (error?.name === "NotAllowedError") {
        setTabCaptureError("Tab audio capture was blocked.");
      } else if (error?.name === "NotFoundError") {
        setTabCaptureError("No audio source was available for the selected tab.");
      } else {
        setTabCaptureError("Unable to start tab audio capture.");
      }
    } finally {
      setIsTabCaptureLoading(false);
    }
  }, [isTabCaptureLoading, previewAudioRef, setIsPlaying]);

  useEffect(() => {
    let cancelled = false;
    const previewUrl = data.previewAudioUrl?.trim();
    if (previewUrl) {
      setAudioSource({ url: previewUrl, label: "preview", status: "ready" });
      return undefined;
    }

    const videoId = extractYouTubeVideoId(data.youtubeUrl);
    if (!videoId) {
      setAudioSource({ url: null, label: "none", status: "idle" });
      return undefined;
    }

    setAudioSource({ url: null, label: "youtube", status: "loading" });

    const controller = new AbortController();

    async function load() {
      try {
        const streamUrl = await resolveYouTubeAudioStream(videoId, {
          signal: controller.signal,
        });
        if (!cancelled) {
          setAudioSource({ url: streamUrl, label: "youtube", status: "ready" });
        }
      } catch (error) {
        if (!cancelled) {
          console.warn("Unable to load YouTube audio stream", error);
          setAudioSource({ url: null, label: "youtube", status: "error" });
        }
      }
    }

    void load();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [data.previewAudioUrl, data.youtubeUrl]);

  useEffect(() => {
    if (!tabAudioStream) return undefined;

    const handleEnded = () => {
      stopTabAudioCapture();
    };

    tabAudioStream.getTracks().forEach((track) => {
      track.addEventListener("ended", handleEnded);
      track.addEventListener("inactive", handleEnded);
    });

    return () => {
      tabAudioStream.getTracks().forEach((track) => {
        track.removeEventListener("ended", handleEnded);
        track.removeEventListener("inactive", handleEnded);
      });
    };
  }, [tabAudioStream, stopTabAudioCapture]);

  useEffect(
    () => () => {
      stopTabAudioCapture();
    },
    [stopTabAudioCapture]
  );

  const embedUrl = isPlaying ? getYouTubeEmbedUrl(data.youtubeUrl) : null;
  const isTabCaptureActive = Boolean(tabAudioStream);

  const { lowLevels, highLevels } = useAudioSpectrum(
    previewAudioRef,
    isPlaying,
    {
      lowBars: 48,
      highBars: 48,
      onModeChange: setAnalysisMode,
      sourceLabel: audioSource.label && audioSource.label !== "none" ? audioSource.label : null,
      sourceVersion: audioSource.url ?? audioSource.status,
    },
    tabAudioStream
  );

  const analysisMessage = useMemo(() => {
    if (tabAudioStream) {
      if (analysisMode === "audio:tab-capture") {
        return "Visualizer synced to shared tab audio.";
      }
      return "Tab audio sharing active — waiting for audio from the shared tab.";
    }

    if (analysisMode === "audio") {
      return "Visualizer synced to audio source.";
    }

    if (typeof analysisMode === "string" && analysisMode.startsWith("audio:")) {
      const [, label] = analysisMode.split(":");
      if (label === "youtube") {
        return "Visualizer synced to YouTube audio stream.";
      }
      if (label === "preview") {
        return "Visualizer synced to preview audio clip.";
      }
    }

    if (analysisMode === "synthetic") {
      if (audioSource.status === "loading") {
        return "Fetching audio stream... equalizer running in ambient mode.";
      }
      if (audioSource.status === "error" && audioSource.label === "youtube") {
        return "Visualizer in ambient mode — unable to access YouTube audio.";
      }
      return "Visualizer running in ambient mode.";
    }

    if (audioSource.status === "loading") {
      return "Fetching audio stream...";
    }

    return null;
  }, [analysisMode, audioSource.status, audioSource.label, tabAudioStream]);

  useEffect(() => {
    const audio = previewAudioRef.current;
    if (!audio) return;

    audio.muted = true;
    audio.loop = true;
    audio.preload = "auto";
    audio.playsInline = true;
  }, []);

  useEffect(() => {
    const audio = previewAudioRef.current;
    if (!audio) return;

    if (audioSource.url) {
      if (audio.src !== audioSource.url) {
        audio.src = audioSource.url;
        audio.load();
      }
    } else if (audio.src) {
      audio.removeAttribute("src");
      audio.load();
    }
  }, [audioSource.url]);

  useEffect(() => {
    async function playIfReady() {
      const audio = previewAudioRef.current;
      if (!audio || !isPlaying) return;
      if (tabAudioStream) return;
      if (!audioSource.url || audioSource.status !== "ready") return;

      try {
        await audio.play();
      } catch (error) {
        console.warn("Autoplay for analysis track blocked", error);
      }
    }

    void playIfReady();
  }, [audioSource.status, audioSource.url, isPlaying, tabAudioStream]);

  async function handleTogglePlayback() {
    if (tabAudioStream) {
      return;
    }
    if (!audioSource.url || audioSource.status !== "ready") {
      return;
    }
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
    <div className="relative flex min-h-screen w-full items-center justify-center overflow-hidden bg-gradient-to-b from-[#050816] via-[#070c1f] to-[#050816] px-6 py-16 text-white">
      <EqualizerRibbon side="left" levels={lowLevels} />
      <EqualizerRibbon side="right" levels={highLevels} />

      <div className="relative z-10 flex max-w-3xl flex-col items-center gap-12 text-center">
        <h1 className="text-4xl font-bold tracking-tight text-white md:text-5xl">
          Your top song was...
          <span className="ml-4 text-3xl align-middle md:text-4xl" aria-hidden="true">
            🥁🥁🥁
          </span>
        </h1>

        <div className="flex flex-col items-center gap-8">
          <div className="rounded-[2.5rem] border-[14px] border-[#151b35] bg-[#0c1430] p-4 shadow-[0_32px_90px_rgba(7,11,28,0.65)]">
            <img
              src={data.albumArtUrl}
              alt={data.album ? `${data.title} album cover for ${data.album}` : `${data.title} album cover`}
              className="h-64 w-64 rounded-[1.75rem] object-cover md:h-72 md:w-72"
            />
          </div>

          <div className="space-y-3">
            <p className="text-4xl font-bold md:text-5xl">{data.title}</p>
            <p className="text-xl font-medium text-white/90 md:text-2xl">{data.artist}</p>
            <p className="text-sm uppercase tracking-[0.4em] text-white/55 md:text-base">{data.album}</p>
          </div>
        </div>

        <div className="flex flex-col items-center gap-4 text-sm text-white/80">
          <div className="flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              onClick={handleTogglePlayback}
              disabled={!audioSource.url || audioSource.status !== "ready" || isTabCaptureActive}
              className={`rounded-full px-6 py-2 font-medium tracking-wide transition ${
                !audioSource.url || audioSource.status !== "ready" || isTabCaptureActive
                  ? "cursor-not-allowed bg-white/10 text-white/40"
                  : "bg-white/10 hover:bg-white/20"
              }`}
            >
              {isPlaying ? "Stop playback" : "Play track"}
            </button>

            {isTabCaptureActive ? (
              <button
                type="button"
                onClick={stopTabAudioCapture}
                className="rounded-full bg-emerald-400/90 px-6 py-2 font-semibold text-emerald-950 transition hover:bg-emerald-300"
              >
                Stop sharing tab audio
              </button>
            ) : (
              <button
                type="button"
                onClick={startTabAudioCapture}
                disabled={isTabCaptureLoading}
                className={`rounded-full px-6 py-2 font-semibold transition ${
                  isTabCaptureLoading
                    ? "cursor-wait bg-sky-400/40 text-sky-100/70"
                    : "bg-sky-400/80 text-sky-950 hover:bg-sky-300"
                }`}
              >
                {isTabCaptureLoading ? "Starting tab capture..." : "Share tab audio for visualizer"}
              </button>
            )}
          </div>

          {tabCaptureError ? (
            <p className="max-w-sm text-center text-xs text-rose-200/90">{tabCaptureError}</p>
          ) : null}

          {embedUrl && (
            <iframe
              key={embedUrl}
              src={embedUrl}
              title="Top song audio player"
              allow="autoplay; encrypted-media"
              aria-hidden="true"
              tabIndex={-1}
              className="absolute h-px w-px overflow-hidden"
              style={{ clip: "rect(0 0 0 0)" }}
            />
          )}

          {analysisMessage ? (
            <p className="max-w-sm text-center text-xs text-white/60">{analysisMessage}</p>
          ) : null}
        </div>
      </div>

      <audio ref={previewAudioRef} />
    </div>
  );
}
