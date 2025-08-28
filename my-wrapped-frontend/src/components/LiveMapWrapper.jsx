import { useEffect, useState } from "react";
import CountryHeatmapSlide from "../slides/CountryHeatmapSlide";

export default function LiveMapWrapper() {
  const [data, setData] = useState({
    countryCounts: {},
    countryTopArtists: {},
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("http://localhost:8000/country-summary");
        const json = await res.json();
        setData(json);
      } catch (err) {
        console.error("Error fetching heatmap data:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  const isLoading = Object.keys(data.countryCounts).length === 0;

  return (
    isLoading ? (
      <div className="flex items-center justify-center w-full h-screen bg-slate-950 text-slate-300">
        <span className="animate-pulse text-violet-400 mr-2">♪</span>
        Loading map data...
      </div>
    ) : (
      <CountryHeatmapSlide
        countryCounts={data.countryCounts}
        countryTopArtists={data.countryTopArtists}
      />
    )
  );
}
