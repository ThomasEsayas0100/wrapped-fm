import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import * as topojson from "topojson-client";
import worldData from "../data/world-110m.json";

export default function CountryHeatmapSlide({ countryCounts, countryTopArtists }) {
  const svgRef = useRef();
  const [tooltip, setTooltip] = useState(null);
  const [cursor, setCursor] = useState({ x: -100, y: -100 });

  const countries = useMemo(
    () => topojson.feature(worldData, worldData.objects.countries).features,
    []
  );

  const maxCount = useMemo(() => d3.max(Object.values(countryCounts)), [countryCounts]);
  const colorScale = useMemo(
    () =>
      d3
        .scaleSequentialLog(d3.interpolatePlasma)
        .domain([1, maxCount || 1]),
    [maxCount]
  );

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const { width, height } = svgRef.current.getBoundingClientRect();
    svg.selectAll("*").remove();
    svg.attr("viewBox", [0, 0, width, height]);
    const projection = d3
      .geoNaturalEarth1()
      .scale(width / 6)
      .translate([width / 2, height / 2]);
    const path = d3.geoPath().projection(projection);

    svg
      .append("path")
      .datum({ type: "Sphere" })
      .attr("d", path)
      .attr("fill", "#0f172a")
      .attr("stroke", "#475569")
      .attr("stroke-width", 0.5);

    svg
      .append("g")
      .selectAll("path")
      .data(countries)
      .join("path")
      .attr("d", path)
      .attr("fill", d => {
        const count = countryCounts[d.properties.name] || 0;
        return count ? colorScale(count) : "#222";
      })
      .attr("stroke", "#00000055")
      .attr("stroke-width", 0.5)
      .on("mousemove", function (event, d) {
        const [mx, my] = d3.pointer(event);
        const name = d.properties.name;
        const count = countryCounts[name] || 0;
        const topArtists = countryTopArtists[name] || [];
        const artistList = topArtists.length ? topArtists.join(", ") : "No data";
        const tooltipWidth = 220;
        const tooltipHeight = 80;
        const x = Math.min(mx + 20, width - tooltipWidth);
        const y = Math.min(my + 20, height - tooltipHeight);
        d3.select(this).attr("stroke", "#fff").attr("stroke-width", 1.2);
        setTooltip({ x, y, name, count, artistList });
      })
      .on("mouseout", function () {
        d3.select(this).attr("stroke", "#00000055").attr("stroke-width", 0.5);
        setTooltip(null);
      });
  }, [countries, countryCounts, countryTopArtists, colorScale]);

  return (
    <div
      className="relative w-full h-screen overflow-hidden text-white"
      onMouseMove={e => setCursor({ x: e.clientX, y: e.clientY })}
    >
      <div className="absolute inset-0 -z-20 bg-gradient-to-br from-indigo-950 via-purple-900 to-sky-950 bg-[length:300%_300%] animate-gradient-slow" />
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute top-1/3 left-1/4 w-[40rem] h-[40rem] bg-fuchsia-500/20 rounded-full blur-3xl animate-float" />
        <div
          className="absolute top-1/4 left-1/2 w-[30rem] h-[30rem] bg-indigo-500/20 rounded-full blur-3xl animate-float"
          style={{ animationDelay: "-5s" }}
        />
        <div
          className="absolute top-2/3 left-1/3 w-[35rem] h-[35rem] bg-rose-500/10 rounded-full blur-3xl animate-float"
          style={{ animationDelay: "-2s" }}
        />
      </div>

      <div className="relative z-0 flex items-center justify-center w-full h-full px-4">
        <div className="relative w-full max-w-7xl h-[90vh] p-[3px] rounded-3xl bg-gradient-to-r from-yellow-400 via-pink-500 to-orange-500 bg-[length:400%_400%] animate-border shadow-[0_0_25px_rgba(255,200,100,0.3)]">
          <div className="relative w-full h-full rounded-3xl bg-slate-900/30 backdrop-blur-xl overflow-hidden">
            <svg ref={svgRef} className="absolute inset-0 w-full h-full" />

            {tooltip && (
              <div
                className="pointer-events-none absolute z-20 bg-slate-900/90 border border-white/20 rounded-lg px-3 py-2 text-xs shadow-lg backdrop-blur"
                style={{ left: tooltip.x, top: tooltip.y }}
              >
                <p className="font-semibold">{tooltip.name}</p>
                <p>{tooltip.count} plays</p>
                <p className="mt-1 text-[10px] text-violet-200">Top Artists: {tooltip.artistList}</p>
              </div>
            )}

            <header className="absolute top-6 left-1/2 -translate-x-1/2 text-center z-10">
              <h1 className="text-5xl font-bold tracking-tight bg-gradient-to-r from-yellow-200 via-pink-200 to-orange-200 bg-clip-text text-transparent drop-shadow-lg">
                Where Your Music Comes From
              </h1>
              <p className="mt-2 text-sm text-slate-300 max-w-lg mx-auto">
                Hover over a country to reveal your play count and top artists from that region.
              </p>
            </header>

            <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex items-center text-xs text-slate-300 z-10">
              <span className="mr-2">Less</span>
              <div className="h-2 w-48 bg-gradient-to-r from-violet-900 via-fuchsia-600 to-yellow-300 rounded" />
              <span className="ml-2">More</span>
            </div>
          </div>
        </div>
      </div>

      <div
        className="pointer-events-none fixed z-30 w-24 h-24 rounded-full bg-violet-400/10 mix-blend-screen blur-3xl -translate-x-1/2 -translate-y-1/2 transition-all duration-300"
        style={{ left: cursor.x, top: cursor.y }}
      />
    </div>
  );
}
