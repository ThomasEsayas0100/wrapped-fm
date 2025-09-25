import { useState } from "react";

const defaultData = {
  title: "Get Lucky",
  artist: "Daft Punk",
  album: "Random Access Memories",
  albumArtUrl:
    "https://upload.wikimedia.org/wikipedia/en/a/a7/Random_Access_Memories.jpg",
  youtubeUrl: "https://www.youtube.com/watch?v=5NV6Rdv1a3I",
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

function DecorativeRibbon({ side }) {
  const isLeft = side === "left";

  return (
    <div
      className={`pointer-events-none absolute inset-y-[-12%] ${
        isLeft ? "left-[-22%] justify-start" : "right-[-22%] justify-end"
      } flex w-[48vw]`}
    >
      <div
        className="h-full w-[38vw] max-w-[520px]"
        style={{
          background:
            "linear-gradient(135deg, rgba(59,130,246,0.95), rgba(125,211,252,0.85))",
          borderRadius: "46% 54% 60% 40% / 38% 52% 48% 62%",
          transform: isLeft
            ? "translateX(-12%) rotate(-12deg)"
            : "translateX(12%) rotate(12deg)",
          filter: "drop-shadow(0 0 90px rgba(56,189,248,0.35))",
        }}
      />
    </div>
  );
}

export default function FeaturedSongSlide({ data = defaultData }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const embedUrl = isPlaying ? getYouTubeEmbedUrl(data.youtubeUrl) : null;

  return (
    <div className="relative flex min-h-screen w-full items-center justify-center overflow-hidden bg-[#070c1f] px-6 py-16 text-white">
      <DecorativeRibbon side="left" />
      <DecorativeRibbon side="right" />

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
              onClick={() => setIsPlaying((prev) => !prev)}
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
          </div>
        ) : null}
      </div>
    </div>
  );
}
