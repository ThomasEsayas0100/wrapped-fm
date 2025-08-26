const fs = require('fs');
const d3 = require('d3');
const { JSDOM } = require('jsdom');

function generateCalendar(streakDates, year = 2025, outputPath = 'calendar.svg') {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
  const body = d3.select(dom.window.document).select('body');

  const boxSize = 16;
  const boxSpacing = 4;
  const monthSpacingX = 170;
  const monthSpacingY = 170;

  const streakSet = new Set(streakDates.map(d => new Date(d).toDateString()));
  const sortedStreak = streakDates.slice().sort();
  const totalDelay = 4; // seconds
  const N = sortedStreak.length;

  // Generate delay map using sine easing
  const delayMap = new Map();
  const maxDelay = 0.2;
  const minDelay = 0.05;
  const midIndex = (N - 1) / 2;
  const slope = (maxDelay - minDelay) / midIndex;  
  let cumulativeDelay = 0;
  sortedStreak.forEach((isoDate, i) => {
    const delay = maxDelay - Math.abs(i - midIndex) * slope;
    cumulativeDelay += delay;
    delayMap.set(isoDate, cumulativeDelay);
  });

  // Fix sorted streak
  
  console.log(delayMap);

  const cols = 4;
  const rows = 3;
  const margin = 40;
  const maxWeeks = 6;

  const svgWidth = cols * monthSpacingX + margin * 2;
  const svgHeight = rows * monthSpacingY + margin + maxWeeks * (boxSize + boxSpacing);

  const svg = body.append('svg')
    .attr('xmlns', 'http://www.w3.org/2000/svg')
    .attr('width', svgWidth)
    .attr('height', svgHeight)
    .attr('style', 'background:#111');

  // CSS animation for color shifting
  svg.append('style').text(`
    .streak-box {
      animation: colorShift 6s linear infinite;
    }

    @keyframes colorShift {
      0%   { fill: #8b5cf6; }
      25%  { fill: #3b82f6; }
      50%  { fill: #06b6d4; }
      75%  { fill: #3b82f6; }
      100% { fill: #8b5cf6; }
    }
  `);

  const monthNames = d3.timeFormat('%b');

  for (let month = 0; month < 12; month++) {
    const col = month % cols;
    const row = Math.floor(month / cols);
    const offsetX = col * monthSpacingX + margin;
    const offsetY = row * monthSpacingY + margin;

    const firstDay = new Date(year, month, 1);
    const daysInMonth = new Date(year, month + 1, 0).getDate();

    svg.append('text')
      .attr('x', offsetX + 2)
      .attr('y', offsetY - 7)
      .attr('fill', '#4169E1')
      .attr('font-size', '22px')
      .attr('font-family', 'Segoe UI, Roboto, sans-serif')
      .attr('font-weight', 'bold')
      .text(monthNames(new Date(year, month, 1)).toUpperCase());

    for (let day = 1; day <= daysInMonth; day++) {
      const date = new Date(year, month, day);
      const dayOfWeek = date.getDay();
      const weekOfMonth = Math.floor((day + firstDay.getDay() - 1) / 7);
      const x = offsetX + dayOfWeek * (boxSize + boxSpacing);
      const y = offsetY + weekOfMonth * (boxSize + boxSpacing);

      const dateStr = date.toDateString();
      const isoDate = date.toISOString().slice(0, 10);
      const inStreak = streakSet.has(dateStr);

      // Base layer: default box background for all
      svg.append('rect')
        .attr('x', x)
        .attr('y', y)
        .attr('width', boxSize)
        .attr('height', boxSize)
        .attr('rx', 3)
        .attr('ry', 3)
        .attr('fill', '#2a2a2a');

      // Animated overlay for streak days
      if (inStreak && delayMap.has(isoDate)) {
        const delay = delayMap.get(isoDate);

        const rect = svg.append('rect')
          .attr('class', 'streak-box')
          .attr('x', x)
          .attr('y', y)
          .attr('width', boxSize)
          .attr('height', boxSize)
          .attr('rx', 3)
          .attr('ry', 3)
          .attr('opacity', 0);

        rect.append('animate')
          .attr('attributeName', 'opacity')
          .attr('from', 0)
          .attr('to', 1)
          .attr('begin', `${delay.toFixed(1)}s`)
          .attr('dur', '0.3s')
          .attr('fill', 'freeze');
      }
    }
  }

  fs.writeFileSync(outputPath, body.html());
  console.log(`✅ Nonlinear animated calendar saved to ${outputPath}`);
}

// Example streak
const streakDates = [
    '2025-04-29', '2025-04-30',
    '2025-05-01', '2025-05-02', '2025-05-03',
    '2025-05-04', '2025-05-05', '2025-05-06',
    '2025-05-07', '2025-05-08', '2025-05-09',
    '2025-05-10', '2025-05-11', '2025-05-12',
    '2025-05-13', '2025-05-14', '2025-05-15',
    '2025-05-16', '2025-05-17', '2025-05-18',
    '2025-05-19', '2025-05-20', '2025-05-21',
    '2025-05-22', '2025-05-23', '2025-05-24',
    '2025-05-25', '2025-05-26', '2025-05-27',
    '2025-05-28', '2025-05-29', '2025-05-30', '2025-05-31'
  ];

generateCalendar(streakDates);
