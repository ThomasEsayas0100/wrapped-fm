import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import * as topojson from "topojson-client";
import worldData from "../data/world-110m.json";

export default function CountryHeatmapSlide({ countryCounts, countryTopArtists }) {
  const svgRef = useRef();
  const [tooltip, setTooltip] = useState(null);

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
    const width = window.innerWidth;
    const height = window.innerHeight;
    svg.selectAll("*").remove();
    svg.attr("viewBox", [0, 0, width, height]);

    const projection = d3.geoNaturalEarth1().scale(width / 6.5).translate([width / 2, height / 2]);
    const path = d3.geoPath().projection(projection);

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
    <div className="relative w-full h-screen overflow-hidden bg-gradient-to-b from-slate-950 via-violet-950 to-slate-900 text-white">
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

      <header className="absolute top-8 left-1/2 -translate-x-1/2 text-center z-10">
        <h1 className="text-5xl font-bold tracking-tight bg-gradient-to-r from-violet-400 to-rose-400 bg-clip-text text-transparent drop-shadow-lg">
          Where Your Music Comes From
        </h1>
        <p className="mt-2 text-sm text-slate-300 max-w-lg mx-auto">
          Hover over a country to reveal your play count and top artists from that region.
        </p>
      </header>

      <div className="absolute bottom-12 left-1/2 -translate-x-1/2 flex items-center text-xs text-slate-300 z-10">
        <span className="mr-2">Less</span>
        <div className="h-2 w-48 bg-gradient-to-r from-violet-900 via-fuchsia-600 to-yellow-300 rounded" />
        <span className="ml-2">More</span>
      </div>
    </div>
  );
}
