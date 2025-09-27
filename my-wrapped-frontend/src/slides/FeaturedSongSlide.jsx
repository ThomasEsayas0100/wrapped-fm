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

    function drawRoundedRect(x, y, width, height, radius) {
      const clampedRadius = Math.max(
        0,
        Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2)
      );

      ctx.beginPath();
      ctx.moveTo(x + clampedRadius, y);
      ctx.lineTo(x + width - clampedRadius, y);
      ctx.quadraticCurveTo(x + width, y, x + width, y + clampedRadius);
      ctx.lineTo(x + width, y + height - clampedRadius);
      ctx.quadraticCurveTo(
        x + width,
        y + height,
        x + width - clampedRadius,
        y + height
      );
      ctx.lineTo(x + clampedRadius, y + height);
      ctx.quadraticCurveTo(x, y + height, x, y + height - clampedRadius);
      ctx.lineTo(x, y + clampedRadius);
      ctx.quadraticCurveTo(x, y, x + clampedRadius, y);
      ctx.closePath();
    }

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
      const isLeft = side === "left";
      const outerX = isLeft ? width : 0;
      const baseThickness = width * 0.14;
      const amplitude = width * 0.56;
      const rowHeight = height / ribbonLevels.length;
      const glowColorStops = isLeft
        ? [
            "rgba(56, 189, 248, 0.18)",
            "rgba(14, 165, 233, 0.28)",
            "rgba(12, 74, 110, 0.05)",
          ]
        : [
            "rgba(168, 85, 247, 0.18)",
            "rgba(129, 140, 248, 0.28)",
            "rgba(76, 29, 149, 0.06)",
          ];

      const washGradient = ctx.createLinearGradient(
        isLeft ? width : 0,
        0,
        isLeft ? 0 : width,
        height
      );
      glowColorStops.forEach((color, index) => {
        washGradient.addColorStop(index / (glowColorStops.length - 1 || 1), color);
      });

      ctx.save();
      ctx.globalAlpha = 0.45;
      ctx.fillStyle = washGradient;
      ctx.fillRect(0, 0, width, height);
      ctx.restore();

      const beamGradient = ctx.createLinearGradient(
        isLeft ? width : 0,
        0,
        isLeft ? 0 : width,
        height
      );
      beamGradient.addColorStop(0, isLeft ? "#38bdf8" : "#c084fc");
      beamGradient.addColorStop(0.4, isLeft ? "#22d3ee" : "#a855f7");
      beamGradient.addColorStop(1, isLeft ? "#0ea5e9" : "#6366f1");

      ctx.save();
      ctx.globalCompositeOperation = "lighter";

      for (let i = 0; i < ribbonLevels.length; i += 1) {
        const level = ribbonLevels[i];
        const normalized = Math.max(0, Math.min(1, level));
        const bias = Math.sin((i / ribbonLevels.length) * Math.PI) * 0.12;
        const pulse = Math.sin(performance.now() * 0.0018 + i * 0.45) * 0.06;
        const thickness = baseThickness + amplitude * (normalized * 0.9 + bias * 0.6 + pulse);
        const heightScale = 0.6 + normalized * 0.55;
        const barHeight = rowHeight * heightScale;
        const y = i * rowHeight + (rowHeight - barHeight) / 2;
        const x = isLeft ? width - thickness : outerX;

        ctx.globalAlpha = 0.7 + normalized * 0.25;
        ctx.fillStyle = beamGradient;
        drawRoundedRect(x, y, thickness, barHeight, barHeight * 0.35);
        ctx.fill();

        const highlightWidth = thickness * (isLeft ? 0.22 : 0.18);
        const highlightX = isLeft ? width - highlightWidth : x;
        const highlightGradient = ctx.createLinearGradient(
          highlightX,
          y,
          highlightX + highlightWidth,
          y + barHeight
        );
        if (isLeft) {
          highlightGradient.addColorStop(0, "rgba(224, 242, 254, 0.65)");
          highlightGradient.addColorStop(1, "rgba(125, 211, 252, 0)");
        } else {
          highlightGradient.addColorStop(0, "rgba(236, 233, 254, 0.65)");
          highlightGradient.addColorStop(1, "rgba(199, 210, 254, 0)");
        }

        ctx.globalAlpha = 0.95;
        ctx.fillStyle = highlightGradient;
        drawRoundedRect(
          highlightX,
          y + barHeight * 0.12,
          highlightWidth,
          barHeight * 0.76,
          barHeight * 0.32
        );
        ctx.fill();
      }

      ctx.restore();

      animationFrame = requestAnimationFrame(draw);
    }

    animationFrame = requestAnimationFrame(draw);

    return () => cancelAnimationFrame(animationFrame);
  }, [side]);

  return (
    <div
      className={`pointer-events-none absolute inset-y-[-14%] ${
        side === "left" ? "left-[-10%]" : "right-[-10%]"
      } w-[clamp(220px,24vw,340px)]`}
    >
      <canvas ref={canvasRef} className="h-full w-full" />
      <div
        className={`absolute inset-0 ${
          side === "left" ? "bg-gradient-to-tr" : "bg-gradient-to-tl"
        } from-sky-400/20 via-cyan-300/10 to-transparent blur-3xl mix-blend-screen`}
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
        // Some browsers require a video track in order to enable tab audio
        // capture. Request the smallest possible surface and immediately drop
        // the track after access is granted.
        video: {
          cursor: "never",
          frameRate: 1,
          width: 1,
          height: 1,
          displaySurface: "browser",
        },
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
      } else if (error?.name === "NotSupportedError") {
        setTabCaptureError(
          "This browser requires sharing a tab (with video) to enable audio capture."
        );
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
    <div className="relative flex min-h-screen w-full items-center justify-center overflow-hidden bg-[#030712] px-6 py-16 text-white">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_15%_20%,rgba(59,130,246,0.22),transparent_55%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_25%,rgba(124,58,237,0.18),transparent_52%)]" />
        <div className="absolute inset-x-[10%] top-[-30%] h-[55%] rounded-full bg-gradient-to-b from-sky-500/30 via-transparent to-transparent blur-3xl" />
        <div className="absolute inset-x-[5%] bottom-[-35%] h-[60%] rounded-full bg-gradient-to-t from-indigo-500/25 via-transparent to-transparent blur-3xl" />
      </div>
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
          <div className="rounded-[2.5rem] border border-white/10 bg-white/5 p-3 backdrop-blur-xl shadow-[0_35px_110px_rgba(8,14,35,0.55)]">
            <img
              src={data.albumArtUrl}
              alt={data.album ? `${data.title} album cover for ${data.album}` : `${data.title} album cover`}
              className="h-64 w-64 rounded-[1.75rem] object-cover ring-1 ring-white/10 md:h-72 md:w-72"
            />
          </div>

          <div className="space-y-3">
            <p className="text-4xl font-bold tracking-tight md:text-5xl">{data.title}</p>
            <p className="text-xl font-medium text-white/90 md:text-2xl">{data.artist}</p>
            <p className="text-xs uppercase tracking-[0.55em] text-white/50 md:text-sm">{data.album}</p>
          </div>
        </div>

        <div className="flex flex-col items-center gap-4 text-sm text-white/80">
          <div className="flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              onClick={handleTogglePlayback}
              disabled={!audioSource.url || audioSource.status !== "ready" || isTabCaptureActive}
              className={`rounded-full px-6 py-2 font-semibold tracking-wide transition ${
                !audioSource.url || audioSource.status !== "ready" || isTabCaptureActive
                  ? "cursor-not-allowed bg-white/10 text-white/35"
                  : "bg-white/80 text-slate-950 shadow-lg shadow-sky-500/30 transition hover:shadow-sky-400/40"
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
                    : "bg-sky-400/80 text-sky-950 shadow-lg shadow-sky-500/40 hover:bg-sky-300"
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
            <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-[11px] font-medium uppercase tracking-[0.3em] text-white/70">
              {analysisMessage}
            </div>
          ) : null}
        </div>
      </div>

      <audio ref={previewAudioRef} />
    </div>
  );
}
