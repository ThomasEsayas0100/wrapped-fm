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

  return (
    Object.keys(data.countryCounts).length === 0
        ? <p>Loading map data...</p>
        : <CountryHeatmapSlide
            countryCounts={data.countryCounts}
            countryTopArtists={data.countryTopArtists}
          />
      
    // <CountryHeatmapSlide
    //   countryCounts={data.countryCounts}
    //   countryTopArtists={data.countryTopArtists}
    // />
  );
}
