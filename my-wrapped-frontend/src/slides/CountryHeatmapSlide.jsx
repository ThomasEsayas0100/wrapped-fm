import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import * as topojson from "topojson-client";
import worldData from "../data/world-110m.json"; // ✅ relative path from slides folder

export default function CountryHeatmapSlide({ countryCounts, countryTopArtists }) {
  const svgRef = useRef();
  const [tooltipContent, setTooltipContent] = useState(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const width = window.innerWidth;
    const height = window.innerHeight;

    svg.selectAll("*").remove();
    svg.attr("viewBox", [0, 0, width, height]);

    const projection = d3.geoNaturalEarth1().scale(width / 6.5).translate([width / 2, height / 2]);
    const path = d3.geoPath().projection(projection);

    const maxCount = d3.max(Object.values(countryCounts));
    const colorScale = d3
      .scaleSequentialSqrt(d3.interpolateTurbo)
      .domain([0, maxCount || 1]);

    const countries = topojson.feature(worldData, worldData.objects.countries);

    svg
      .append("g")
      .selectAll("path")
      .data(countries.features)
      .join("path")
      .attr("d", path)
      .attr("fill", d => {
        const name = d.properties.name;
        const count = countryCounts[name] || 0;
        return count > 0 ? colorScale(count) : "#1e1e2e";
      })
      .attr("stroke", "#ffffff33")
      .attr("stroke-width", 0.5)
      .on("mousemove", (event, d) => {
        const name = d.properties.name;
        const count = countryCounts[name] || 0;
        const topArtists = countryTopArtists[name] || [];
        const artistList = topArtists.length > 0 ? topArtists.join(", ") : "No data";
        setTooltipContent(`${name} — ${count} plays\nTop Artists: ${artistList}`);
      })
      .on("mouseout", () => {
        setTooltipContent(null);
      });
  }, [countryCounts, countryTopArtists]);

  return (
    <div className="relative w-full h-screen overflow-hidden bg-gradient-to-b from-[#050026] to-[#120933]">
    <svg ref={svgRef} className="absolute inset-0 w-full h-full" />

    {tooltipContent && (
      <div className="fixed top-1/2 right-6 transform -translate-y-1/2 w-72 bg-white/10 text-white p-4 rounded-xl backdrop-blur-md shadow-xl text-sm font-mono whitespace-pre-wrap z-50">
        {tooltipContent}
      </div>
    )}

    <div className="absolute top-10 left-10 text-white z-10">
      <h1 className="text-4xl font-bold text-cyan-300 drop-shadow">Where Your Music Comes From</h1>
      <p className="text-sm text-magenta-100 mt-2 max-w-md">
        This map visualizes the total number of scrobbles from artists based in each country. Hover over a country to see details.
      </p>
    </div>
  </div>


  );
}
